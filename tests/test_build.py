import pandas as pd
import pytest

from txcat.data.build import build_dataset
from txcat.data.kaggle import (
    PERSONAL_FINANCE,
    SEMANTIC_LEXICON,
    load_personal_finance,
    load_semantic_lexicon,
)

PERSONAL_ROWS = """Date,Description,Amount,Transaction Type,Category,Account Name
01/01/2018,Amazon,11.11,debit,Shopping,Platinum Card
01/02/2018,Mortgage Payment,1247.44,debit,Mortgage & Rent,Checking
01/12/2018,Biweekly Paycheck,2000.00,credit,Paycheck,Checking
"""
LEXICON_ROWS = """entity,category,type
zus coffee,Food & Beverage,merchant
broadband bill,Bills,keyword
zus coffee,Food & Beverage,merchant
"""


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    for ds, content in [(PERSONAL_FINANCE, PERSONAL_ROWS), (SEMANTIC_LEXICON, LEXICON_ROWS)]:
        ds.local_dir(raw).mkdir(parents=True)
        ds.local_path(raw).write_text(content, encoding="utf-8")
    return raw


def test_personal_finance_mapping(raw_dir):
    df = load_personal_finance(PERSONAL_FINANCE.local_path(raw_dir))

    assert df["amount"].tolist() == [-11.11, -1247.44, 2000.0]
    assert df["category"].tolist() == ["Shopping", "Housing", "Income"]
    assert df["account_id"].tolist() == ["KPF-platinum-card", "KPF-checking", "KPF-checking"]
    assert (df["source"] == "kaggle").all()


def test_unmapped_category_fails_loudly(tmp_path):
    path = tmp_path / "pf.csv"
    path.write_text(PERSONAL_ROWS + "01/13/2018,Zoo,5,debit,Pets,Checking\n")

    with pytest.raises(ValueError, match="Pets"):
        load_personal_finance(path)


def test_lexicon_mapping_dedupes(raw_dir):
    lex = load_semantic_lexicon(SEMANTIC_LEXICON.local_path(raw_dir))

    assert lex["category"].tolist() == ["Dining & Coffee", "Utilities"]


def test_build_writes_processed_outputs(raw_dir, tmp_path):
    processed = tmp_path / "processed"

    df, report = build_dataset(raw_dir, processed, n_accounts=2, seed=1)

    stored = pd.read_parquet(processed / "transactions.parquet")
    assert len(stored) == len(df) > 0
    assert set(stored["source"]) == {"synthetic", "kaggle"}
    assert report.rows_by_dataset["kaggle_personal_finance"] == 3
    assert report.lexicon_rows == 2
    assert report.merchant_precision > 0.95
    assert not report.missing_sources
    assert (raw_dir / "synthetic" / "transactions.csv").exists()


def test_build_without_kaggle_files_still_works(tmp_path):
    df, report = build_dataset(tmp_path / "raw", tmp_path / "processed", n_accounts=1)

    assert set(df["source"]) == {"synthetic"}
    assert len(report.missing_sources) == 2
