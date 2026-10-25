from datetime import date

import pandas as pd
import pytest

from txcat.data.build import INGEST_COLUMNS
from txcat.data.clean import has_ethiopic
from txcat.data.synthetic import SyntheticGenerator
from txcat.data.taxonomy import CATEGORIES
from txcat.ingest import validate_transactions


@pytest.fixture(scope="module")
def synthetic():
    return SyntheticGenerator(seed=7).generate(n_accounts=5, end=date(2023, 12, 31))


def test_same_seed_same_data(synthetic):
    again = SyntheticGenerator(seed=7).generate(n_accounts=5, end=date(2023, 12, 31))

    pd.testing.assert_frame_equal(synthetic, again)


def test_different_seed_different_data(synthetic):
    other = SyntheticGenerator(seed=8).generate(n_accounts=5, end=date(2023, 12, 31))

    assert not synthetic["description"].equals(other["description"])


def test_every_row_is_labelled_synthetic(synthetic):
    assert (synthetic["source"] == "synthetic").all()
    assert (synthetic["dataset"] == "synthetic_et").all()


def test_categories_and_currency(synthetic):
    assert set(synthetic["category"]) <= set(CATEGORIES)
    assert (synthetic["currency"] == "ETB").all()
    assert synthetic["date"].between("2023-01-01", "2023-12-31").all()


def test_mixed_language_descriptions(synthetic):
    ethiopic = synthetic["description"].map(has_ethiopic)
    assert 0.2 < ethiopic.mean() < 0.8


def test_salary_is_monthly_income(synthetic):
    income = synthetic[synthetic["category"] == "Income"]

    assert (income["amount"] > 0).all()
    assert (income.groupby("account_id").size() == 12).all()


def test_passes_ingestion_contract(synthetic):
    raw = synthetic[INGEST_COLUMNS].astype({"date": str, "amount": str})

    assert validate_transactions(raw).rejected.empty
