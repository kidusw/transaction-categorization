"""Pandera contract for raw transaction files, plus row-level rejection reporting."""

from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from txcat.db.models import DATA_SOURCES

REQUIRED_COLUMNS = ("account_id", "date", "description", "amount", "source")
MAX_ABS_AMOUNT = 1e10


class IngestionError(ValueError):
    """The file as a whole is unusable (e.g. a required column is missing)."""


RAW_TRANSACTION_SCHEMA = pa.DataFrameSchema(
    {
        "account_id": pa.Column(str, pa.Check.str_length(1, 64)),
        "date": pa.Column(
            "datetime64[ns]",
            pa.Check(lambda s: s <= pd.Timestamp.now(), error="date is in the future"),
        ),
        "description": pa.Column(str, pa.Check.str_length(1, 500)),
        "amount": pa.Column(
            float,
            [
                pa.Check(lambda s: s != 0, error="amount is zero"),
                pa.Check(lambda s: s.abs() < MAX_ABS_AMOUNT, error="amount out of range"),
            ],
        ),
        "currency": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{3}$")),
        "category": pa.Column(str, pa.Check.str_length(1, 64), nullable=True),
        "source": pa.Column(str, pa.Check.isin(DATA_SOURCES)),
    },
    coerce=True,
    strict="filter",
)


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame  # original columns + `reason`


def _tidy(df: pd.DataFrame, default_currency: str) -> pd.DataFrame:
    """Normalise whitespace/case and fill optional columns before validation."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IngestionError(f"missing required columns: {', '.join(missing)}")

    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip().replace("", None)
    if "currency" not in df.columns:
        df["currency"] = default_currency
    df["currency"] = df["currency"].fillna(default_currency).str.upper()
    if "category" not in df.columns:
        df["category"] = None
    df["source"] = df["source"].str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def validate_transactions(df: pd.DataFrame, default_currency: str = "ETB") -> ValidationResult:
    """Validate a raw frame; bad rows are split out with a human-readable reason."""
    df = _tidy(df, default_currency).reset_index(drop=True)
    try:
        return ValidationResult(RAW_TRANSACTION_SCHEMA.validate(df, lazy=True), df.iloc[0:0])
    except SchemaErrors as exc:
        cases = exc.failure_cases
        if cases["index"].isna().any():
            raise IngestionError(f"schema-level failure:\n{cases[cases['index'].isna()]}") from exc
        reasons = (
            cases.assign(reason=cases["column"].astype(str) + ": " + cases["check"].astype(str))
            .groupby("index")["reason"]
            .agg(lambda r: "; ".join(sorted(set(r))))
        )
        bad = df.index.isin(reasons.index)
        rejected = df[bad].assign(reason=reasons)
        valid = RAW_TRANSACTION_SCHEMA.validate(df[~bad])
        return ValidationResult(valid, rejected)
