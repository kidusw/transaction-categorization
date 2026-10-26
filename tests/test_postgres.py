"""Integration tests against a real PostgreSQL (skipped when TEST_DATABASE_URL is unset)."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from txcat.config import get_settings
from txcat.ingest import load_transactions, validate_transactions

ROOT = Path(__file__).parents[1]

pytestmark = pytest.mark.postgres


@pytest.fixture
def pg_url():
    url = get_settings().test_database_url
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


@pytest.fixture
def migrated(pg_url):
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["url"] = pg_url
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield create_engine(pg_url)
    command.downgrade(cfg, "base")


def test_migration_creates_tables(migrated):
    tables = set(inspect(migrated).get_table_names())

    assert {"accounts", "categories", "transactions", "review_queue", "alembic_version"} <= tables


def test_ingest_into_postgres_is_idempotent(migrated, sample_df):
    valid = validate_transactions(sample_df).valid
    with Session(migrated) as session:
        first = load_transactions(session, valid)
        second = load_transactions(session, valid)
        session.commit()

    assert first.inserted == len(valid)
    assert second.inserted == 0
