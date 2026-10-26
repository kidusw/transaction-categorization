"""Amount anomalies: is this amount unusual for this counterparty in this account?

Each transaction is scored against the account's *earlier* transactions only (no look-
ahead), so the flag is what a live system would have said on that day. The score is a
robust z-score of log |amount| against the same series (merchant/category + direction);
when the series is too short it falls back to the account's history in that category.
Log amounts make "10x the usual" equally surprising for a 50 ETB coffee and a 5,000 ETB
rent payment.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from txcat.insights.recurring import series_key

MAD_TO_STD = 1.4826  # MAD of a normal distribution x 1.4826 = its standard deviation


@dataclass(frozen=True)
class AnomalyConfig:
    # Flag when robust z >= threshold (unusually *large* amounts only). 5.0 is the best F1
    # on the dev half of the synthetic accounts (`txcat insights evaluate`).
    threshold: float = 5.0
    min_history: int = 5  # earlier transactions needed before scoring
    # Floor on the log-amount scale, so near-constant series (rent) aren't hair-trigger.
    min_scale: float = 0.1


def _prior_robust_z(log_amounts: np.ndarray, min_history: int, min_scale: float) -> np.ndarray:
    """z of each value against the values strictly before it (NaN until min_history)."""
    z = np.full(len(log_amounts), np.nan)
    for i in range(min_history, len(log_amounts)):
        prior = log_amounts[:i]
        median = np.median(prior)
        scale = max(MAD_TO_STD * np.median(np.abs(prior - median)), min_scale)
        z[i] = (log_amounts[i] - median) / scale
    return z


def _grouped_z(df: pd.DataFrame, keys: list, config: AnomalyConfig) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for _, group in df.groupby(keys, sort=False):
        group = group.sort_values("date", kind="stable")
        z = _prior_robust_z(group["log_amount"].to_numpy(), config.min_history, config.min_scale)
        out[group.index] = z
    return out


def score_anomalies(df: pd.DataFrame, config: AnomalyConfig | None = None) -> pd.DataFrame:
    """Adds anomaly_z, anomaly_basis ('series' | 'category' | None) and is_anomaly."""
    config = config or AnomalyConfig()
    work = df.assign(
        series=series_key(df),
        direction=np.where(df["amount"] > 0, "in", "out"),
        log_amount=np.log1p(df["amount"].abs()),
    )
    z_series = _grouped_z(work, ["account_id", "series"], config)
    z_category = _grouped_z(work, ["account_id", "category", "direction"], config)
    z = z_series.fillna(z_category)
    basis = np.where(z_series.notna(), "series", np.where(z_category.notna(), "category", None))
    return df.assign(
        anomaly_z=z.round(3),
        anomaly_basis=basis,
        is_anomaly=(z >= config.threshold).fillna(False).astype(bool),
    )


def _forest_features(scored: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [np.log1p(scored["amount"].abs()), scored["anomaly_z"].fillna(0).clip(-10, 50)]
    )


def isolation_forest_scores(
    fit_on: pd.DataFrame, score: pd.DataFrame, seed: int = 42
) -> np.ndarray:
    """Comparison model: IsolationForest on [log |amount|, robust z]. Higher = more anomalous.

    Both frames must come from `score_anomalies`. Fit on one set of accounts and score
    another, so the comparison with the z-score threshold is out-of-sample.
    """
    from sklearn.ensemble import IsolationForest

    forest = IsolationForest(n_estimators=200, random_state=seed)
    forest.fit(_forest_features(fit_on))
    return -forest.score_samples(_forest_features(score))
