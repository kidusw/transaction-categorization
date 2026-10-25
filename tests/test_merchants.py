import pandas as pd
import pytest

from txcat.data.catalog import MERCHANTS
from txcat.data.merchants import MerchantNormalizer
from txcat.data.taxonomy import CATEGORIES


@pytest.fixture(scope="module")
def normalizer():
    return MerchantNormalizer()


@pytest.mark.parametrize(
    ("text", "merchant"),
    [
        ("kaldis coffee bole", "Kaldis Coffee"),
        ("kaldi coffee", "Kaldis Coffee"),  # typo
        ("pos purchase shoa supermarket", "Shoa Supermarket"),
        ("ቴሌብር የአየር ሰዓት ግዢ", "telebirr Airtime"),
        ("ቤት ኪራይ ጥቅምት", "House Rent"),
        ("cbe transfer to abebe k", "Commercial Bank of Ethiopia"),
        ("cbe loan repayment", "CBE Loan"),  # longer alias wins the tie with "cbe"
        ("starbucks seattle", "Starbucks"),
    ],
)
def test_matches_known_merchants(normalizer, text, merchant):
    assert normalizer.match(text).name == merchant


@pytest.mark.parametrize("text", ["grocery store", "biweekly paycheck", "random words here"])
def test_returns_none_when_unsure(normalizer, text):
    assert normalizer.match(text) is None


def test_normalize_series_keeps_index(normalizer):
    s = pd.Series(["tomoca coffee piassa", "nothing"], index=[10, 20])

    out = normalizer.normalize(s)

    assert out[10] == "Tomoca Coffee"
    assert pd.isna(out[20])


def test_catalog_is_consistent():
    names = [m.name for m in MERCHANTS]
    assert len(names) == len(set(names))
    assert all(m.category in CATEGORIES for m in MERCHANTS)
    assert all(alias == alias.lower() for m in MERCHANTS for alias in m.aliases)
