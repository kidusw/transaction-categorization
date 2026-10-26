"""Build the processed modelling dataset from raw Kaggle files + the synthetic generator."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from txcat.data.clean import clean_descriptions, has_ethiopic
from txcat.data.kaggle import (
    PERSONAL_FINANCE,
    SEMANTIC_LEXICON,
    load_personal_finance,
    load_semantic_lexicon,
)
from txcat.data.merchants import MerchantNormalizer
from txcat.data.synthetic import SyntheticGenerator
from txcat.data.taxonomy import CATEGORIES
from txcat.db.models import DATA_SOURCES
from txcat.ingest.loader import transaction_hashes

PROCESSED_SCHEMA = pa.DataFrameSchema(
    {
        "txn_id": pa.Column(str, unique=True),
        "account_id": pa.Column(str),
        "date": pa.Column("datetime64[ns]"),
        "description": pa.Column(str, pa.Check.str_length(min_value=1)),
        "description_clean": pa.Column(str, pa.Check.str_length(min_value=1)),
        "merchant": pa.Column(str, nullable=True),
        "amount": pa.Column(float, pa.Check.ne(0)),
        "currency": pa.Column(str, pa.Check.isin(["ETB", "USD"])),
        "category": pa.Column(str, pa.Check.isin(list(CATEGORIES))),
        "source": pa.Column(str, pa.Check.isin(DATA_SOURCES)),
        "dataset": pa.Column(str),
        "is_injected_anomaly": pa.Column(bool),
        "is_recurring_true": pa.Column("boolean", nullable=True),
        "has_ethiopic": pa.Column(bool),
    },
    strict=False,
)

INGEST_COLUMNS = ["account_id", "date", "description", "amount", "currency", "category", "source"]


@dataclass
class BuildReport:
    rows_by_dataset: dict[str, int] = field(default_factory=dict)
    lexicon_rows: int = 0
    merchant_coverage: float = 0.0  # share of rows with a matched merchant
    merchant_accuracy: float | None = None  # vs synthetic ground truth
    merchant_precision: float | None = None  # of the rows that matched, share correct
    missing_sources: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)


def _merchant_scores(
    df: pd.DataFrame, normalizer: MerchantNormalizer
) -> tuple[float | None, float | None]:
    """(accuracy, precision) against synthetic ground truth for catalog merchants."""
    known = {m.name for m in normalizer.owners}
    evaluable = df[df["merchant_true"].isin(known)]
    if evaluable.empty:
        return None, None
    correct = evaluable["merchant"] == evaluable["merchant_true"]
    matched = evaluable["merchant"].notna()
    precision = float(correct[matched].mean()) if matched.any() else None
    return float(correct.mean()), precision


def build_dataset(
    raw_dir: Path, processed_dir: Path, n_accounts: int = 30, seed: int = 42
) -> tuple[pd.DataFrame, BuildReport]:
    report = BuildReport()
    frames = [SyntheticGenerator(seed).generate(n_accounts)]

    synthetic_csv = raw_dir / "synthetic" / "transactions.csv"
    synthetic_csv.parent.mkdir(parents=True, exist_ok=True)
    frames[0].to_csv(synthetic_csv, index=False)
    report.outputs.append(synthetic_csv)

    if PERSONAL_FINANCE.local_path(raw_dir).exists():
        frames.append(load_personal_finance(PERSONAL_FINANCE.local_path(raw_dir)))
    else:
        report.missing_sources.append(PERSONAL_FINANCE.slug)

    df = pd.concat(frames, ignore_index=True)
    df["amount"] = df["amount"].astype(float)
    df["is_recurring_true"] = df["is_recurring_true"].astype("boolean")
    df["description_clean"] = clean_descriptions(df["description"])
    normalizer = MerchantNormalizer()
    df["merchant"] = normalizer.normalize(df["description_clean"])
    df["has_ethiopic"] = df["description"].map(has_ethiopic)
    df["txn_id"] = transaction_hashes(df).str[:16]
    df = PROCESSED_SCHEMA.validate(df)

    report.rows_by_dataset = df["dataset"].value_counts().to_dict()
    report.merchant_coverage = float(df["merchant"].notna().mean())
    report.merchant_accuracy, report.merchant_precision = _merchant_scores(df, normalizer)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "transactions.parquet"
    df.to_parquet(out, index=False)
    report.outputs.append(out)

    if SEMANTIC_LEXICON.local_path(raw_dir).exists():
        lexicon = load_semantic_lexicon(SEMANTIC_LEXICON.local_path(raw_dir))
        lexicon["text_clean"] = clean_descriptions(lexicon["text"])
        lexicon = lexicon[lexicon["text_clean"].str.len() > 0]
        lex_out = processed_dir / "lexicon.parquet"
        lexicon.to_parquet(lex_out, index=False)
        report.lexicon_rows = len(lexicon)
        report.outputs.append(lex_out)
    else:
        report.missing_sources.append(SEMANTIC_LEXICON.slug)

    return df, report
