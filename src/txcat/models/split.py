"""Time-based train/validation/test split, applied per source.

The sources cover disjoint periods (Kaggle 2018-19, synthetic 2023-25), so a single global
cutoff would put one source entirely in train and the other entirely in test. Instead each
source is cut at its own date quantiles: the oldest rows train, the newest rows test.
"""

import pandas as pd

SPLITS = ("train", "val", "test")


def time_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    by: str = "source",
    date_col: str = "date",
) -> pd.Series:
    """Label every row 'train', 'val' or 'test' by date within its `by` group."""
    if not (val_frac > 0 and test_frac > 0 and val_frac + test_frac < 1):
        raise ValueError("fractions must be positive and sum to less than 1")
    split = pd.Series("train", index=df.index, dtype=object)
    for _, group in df.groupby(by):
        dates = group[date_col]
        val_start = dates.quantile(1 - val_frac - test_frac)
        test_start = dates.quantile(1 - test_frac)
        split[group.index[dates >= val_start]] = "val"
        split[group.index[dates >= test_start]] = "test"
    return split


def split_bounds(df: pd.DataFrame, split: pd.Series, by: str = "source") -> pd.DataFrame:
    """Date range and row count of each (group, split), for logging and the README."""
    return (
        df.assign(split=split)
        .groupby([by, "split"])["date"]
        .agg(start="min", end="max", rows="count")
        .reset_index()
        .assign(start=lambda d: d["start"].dt.date, end=lambda d: d["end"].dt.date)
    )
