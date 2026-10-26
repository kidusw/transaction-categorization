"""Pure table helpers for the dashboard (kept out of app.py so they are testable)."""

import pandas as pd

COMPARISON_COLUMNS = [
    "model", "test macro-F1 (synthetic)", "test macro-F1 (kaggle)",
    "test macro-F1 (unseen text)", "threshold", "auto-accept coverage",
    "auto-accept accuracy",
]  # fmt: skip


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Phase 3 comparison as numbers, without columns that are empty for this model.

    A model trained without the Kaggle data (fresh clone, no token) has no Kaggle scores:
    those arrive as nulls and are dropped rather than shown as blanks.
    """
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table[[c for c in COMPARISON_COLUMNS if c in table]]
    numeric = table.columns[1:]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce")
    return table.dropna(axis=1, how="all")
