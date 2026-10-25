import pandas as pd
import pytest

from txcat.ingest import IngestionError, validate_transactions


def test_valid_rows_pass_and_are_tidied(sample_df):
    result = validate_transactions(sample_df)

    assert len(result.valid) == 8
    valid = result.valid
    assert valid["description"].str.strip().eq(valid["description"]).all()
    assert set(valid["currency"]) == {"ETB", "USD"}  # blank -> default, "usd" -> "USD"
    assert pd.api.types.is_datetime64_any_dtype(valid["date"])
    assert valid["amount"].dtype == float


def test_invalid_rows_are_rejected_with_reasons(sample_df):
    rejected = validate_transactions(sample_df).rejected

    reasons = dict(zip(rejected["description"].fillna(""), rejected["reason"], strict=True))
    assert len(rejected) == 6
    assert "date" in reasons["AMAZON MKTPLACE"]
    assert "description" in reasons[""]
    assert "amount is zero" in reasons["Refund zero"]
    assert "source" in reasons["Shell oil"]
    assert "future" in reasons["Future rent"]
    assert "currency" in reasons["Uber trip"]


def test_missing_required_column_rejects_whole_file(sample_df):
    with pytest.raises(IngestionError, match="amount"):
        validate_transactions(sample_df.drop(columns="amount"))


def test_optional_columns_default(sample_df):
    result = validate_transactions(sample_df.drop(columns=["currency", "category"]), "USD")

    assert (result.valid["currency"] == "USD").all()
    assert result.valid["category"].isna().all()
