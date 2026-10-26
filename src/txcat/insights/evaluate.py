"""Score the detectors against the synthetic ground truth.

The generator records which rows come from an account's recurring profile
(`is_recurring_true`) and which amounts were spiked (`is_injected_anomaly`). Detectors are
unsupervised, so instead of a time split the synthetic accounts are split in two: settings
are chosen on the *dev* accounts (odd ids) and reported on the *held-out* accounts (even
ids). Kaggle rows have no ground truth and are excluded.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from txcat.insights.anomaly import AnomalyConfig, isolation_forest_scores, score_anomalies
from txcat.insights.recurring import RecurringConfig, detect_recurring, mark_recurring

Z_GRID = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0)


@dataclass(frozen=True)
class DetectorScore:
    detector: str
    accounts: str
    slice: str
    rows: int
    positives: int
    precision: float
    recall: float
    f1: float
    setting: str
    average_precision: float | None = None


def account_halves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    synthetic = df[df["source"] == "synthetic"]
    number = synthetic["account_id"].str.extract(r"(\d+)$", expand=False).astype(int)
    return synthetic[number % 2 == 1], synthetic[number % 2 == 0]


def _prf(y_true, y_pred) -> tuple[float, float, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return float(p), float(r), float(f)


def _score(name, accounts, y_true, y_pred, setting, ap=None, slice_="all") -> DetectorScore:
    p, r, f = _prf(y_true, y_pred)
    n_pos = int(np.sum(y_true))
    return DetectorScore(name, accounts, slice_, len(y_true), n_pos, p, r, f, setting, ap)


def _held_out_slices(name, held, y_pred, scores, setting) -> list[DetectorScore]:
    """Held-out scores overall and split into recurring bills vs discretionary spend."""
    y_pred = pd.Series(np.asarray(y_pred), index=held.index)
    scores = pd.Series(np.asarray(scores), index=held.index)
    recurring = held["is_recurring_true"].astype(bool)
    out = []
    everything = pd.Series(True, index=held.index)
    for slice_, mask in (
        ("all", everything),
        ("recurring", recurring),
        ("discretionary", ~recurring),
    ):
        y = held.loc[mask, "is_injected_anomaly"]
        ap = average_precision_score(y, scores[mask]) if y.any() else None
        out.append(_score(name, "held-out", y, y_pred[mask], setting, ap, slice_))
    return out


def evaluate_recurring(df: pd.DataFrame, config: RecurringConfig | None = None):
    """Row-level: is each transaction correctly marked as part of a recurring series?"""
    scores = []
    for accounts, part in zip(("dev", "held-out"), account_halves(df), strict=True):
        predicted = mark_recurring(part, detect_recurring(part, config))
        truth = part["is_recurring_true"].astype(bool)
        scores.append(_score("recurring", accounts, truth, predicted, "default config"))
    return scores


def evaluate_anomalies(df: pd.DataFrame):
    """Robust z (threshold tuned on dev) vs IsolationForest (cutoff tuned on dev)."""
    dev, held = (score_anomalies(part, AnomalyConfig()) for part in account_halves(df))
    y_dev = dev["is_injected_anomaly"]
    z_dev, z_held = dev["anomaly_z"].fillna(-np.inf), held["anomaly_z"].fillna(-np.inf)

    best_z = max(Z_GRID, key=lambda t: _prf(y_dev, z_dev >= t)[2])
    z_setting = f"z >= {best_z}"
    scores = [_score("anomaly: robust z", "dev", y_dev, z_dev >= best_z, z_setting)]
    scores += _held_out_slices(
        "anomaly: robust z", held, z_held >= best_z, held["anomaly_z"].fillna(-100), z_setting
    )

    forest_dev = isolation_forest_scores(dev, dev)
    forest_held = isolation_forest_scores(dev, held)
    cutoffs = np.quantile(forest_dev, [0.95, 0.98, 0.99, 0.995, 0.998])
    best_cut = max(cutoffs, key=lambda c: _prf(y_dev, forest_dev >= c)[2])
    setting = f"top {np.mean(forest_dev >= best_cut):.1%} of dev scores"
    scores.append(_score("anomaly: IsolationForest", "dev", y_dev, forest_dev >= best_cut, setting))
    scores += _held_out_slices(
        "anomaly: IsolationForest", held, forest_held >= best_cut, forest_held, setting
    )
    return scores, best_z


def evaluation_table(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    anomaly_scores, best_z = evaluate_anomalies(df)
    rows = [vars(s) for s in evaluate_recurring(df) + anomaly_scores]
    return pd.DataFrame(rows).round(3), best_z
