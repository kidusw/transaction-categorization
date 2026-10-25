from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from txcat.db import Base

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sample_transactions.csv", dtype=str)


@pytest.fixture
def sqlite_session():
    """Fast in-memory database for unit tests; Postgres is covered by `postgres` tests."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()
