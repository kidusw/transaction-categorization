from sqlalchemy import func, select

from txcat.db import Account, Category, Transaction
from txcat.ingest import load_transactions, validate_transactions


def _count(session, model) -> int:
    return session.scalar(select(func.count()).select_from(model))


def test_load_creates_accounts_categories_and_rows(sqlite_session, sample_df):
    valid = validate_transactions(sample_df).valid

    report = load_transactions(sqlite_session, valid)

    assert report.inserted == len(valid) == _count(sqlite_session, Transaction)
    assert report.accounts_created == 2
    assert report.categories_created == valid["category"].nunique()
    assert _count(sqlite_session, Category) == report.categories_created
    usd = sqlite_session.scalar(select(Account).where(Account.external_id == "ACC-002"))
    assert usd.currency == "USD"


def test_reload_is_idempotent_but_genuine_repeats_survive(sqlite_session, sample_df):
    valid = validate_transactions(sample_df).valid
    load_transactions(sqlite_session, valid)

    again = load_transactions(sqlite_session, valid)

    assert again.inserted == 0
    assert again.duplicates == len(valid)
    assert again.accounts_created == again.categories_created == 0
    coffees = sqlite_session.scalars(
        select(Transaction).where(Transaction.description == "Kaldis coffee")
    ).all()
    assert len(coffees) == 2


def test_labelled_rows_record_label_source(sqlite_session, sample_df):
    load_transactions(sqlite_session, validate_transactions(sample_df).valid)

    rows = sqlite_session.scalars(select(Transaction)).all()
    for row in rows:
        assert (row.category_id is None) == (row.category_source is None)
    assert {r.source for r in rows} == {"kaggle", "synthetic"}


def test_empty_frame_is_noop(sqlite_session, sample_df):
    report = load_transactions(sqlite_session, validate_transactions(sample_df).valid.iloc[0:0])

    assert report.inserted == 0
