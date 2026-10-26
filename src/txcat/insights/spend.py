"""Monthly spend and income by category."""

import pandas as pd


def monthly_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy frame: account_id, month, category, spend (money out, positive), income, count."""
    work = df.assign(
        month=df["date"].dt.to_period("M"),
        spend=(-df["amount"]).clip(lower=0),
        income=df["amount"].clip(lower=0),
    )
    out = (
        work.groupby(["account_id", "month", "category"], observed=True)
        .agg(spend=("spend", "sum"), income=("income", "sum"), transactions=("amount", "size"))
        .reset_index()
    )
    return out.round({"spend": 2, "income": 2})


def spend_table(df: pd.DataFrame, account_id: str, months: int | None = None) -> pd.DataFrame:
    """Category x month table of spend for one account (most recent `months` columns)."""
    monthly = monthly_by_category(df[df["account_id"] == account_id])
    table = monthly.pivot_table(
        index="category", columns="month", values="spend", aggfunc="sum", fill_value=0
    )
    if months:
        table = table.iloc[:, -months:]
    table = table.loc[table.sum(axis=1) > 0]
    return table.sort_values(table.columns[-1], ascending=False) if table.shape[1] else table
