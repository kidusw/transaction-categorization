"""Load validated transactions into the database idempotently."""

import hashlib
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from txcat.db.models import Account, Category, Transaction

CHUNK_SIZE = 1000


@dataclass
class LoadReport:
    inserted: int = 0
    duplicates: int = 0
    accounts_created: int = 0
    categories_created: int = 0


def transaction_hashes(df: pd.DataFrame) -> pd.Series:
    """Stable row identity. Genuine repeats (two identical coffees on one day) get an
    occurrence counter so they survive, while re-loading the same file is a no-op."""
    key = (
        df["account_id"].astype(str)
        + "|"
        + df["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + df["amount"].map("{:.2f}".format)
        + "|"
        + df["description"].astype(str)
    )
    occurrence = key.groupby(key).cumcount().astype(str)
    return (key + "#" + occurrence).map(lambda k: hashlib.sha256(k.encode()).hexdigest())


def _get_or_create(session: Session, model, field: str, values, defaults=None) -> tuple[dict, int]:
    column = getattr(model, field)
    existing = dict(session.execute(select(column, model.id).where(column.in_(values))).all())
    created = 0
    for value in values:
        if value not in existing:
            obj = model(**{field: value}, **(defaults or {}).get(value, {}))
            session.add(obj)
            session.flush()
            existing[value] = obj.id
            created += 1
    return existing, created


def _insert_ignore(session: Session):
    dialect = session.get_bind().dialect.name
    insert = {"postgresql": postgresql.insert, "sqlite": sqlite.insert}[dialect]
    # RETURNING gives an exact inserted count; rowcount is -1 for batched inserts on psycopg.
    return (
        insert(Transaction)
        .on_conflict_do_nothing(index_elements=["txn_hash"])
        .returning(Transaction.id)
    )


def load_transactions(session: Session, df: pd.DataFrame) -> LoadReport:
    """Insert validated rows (see `validate_transactions`), creating accounts/categories."""
    report = LoadReport()
    if df.empty:
        return report

    currencies = df.groupby("account_id")["currency"].first().to_dict()
    account_ids, report.accounts_created = _get_or_create(
        session,
        Account,
        "external_id",
        sorted(df["account_id"].unique()),
        defaults={a: {"currency": c} for a, c in currencies.items()},
    )
    category_names = sorted(df["category"].dropna().unique())
    category_ids, report.categories_created = _get_or_create(
        session, Category, "name", category_names
    )

    df = df.assign(txn_hash=transaction_hashes(df))
    rows = [
        {
            "account_id": account_ids[r.account_id],
            "txn_date": r.date.date(),
            "description": r.description,
            "amount": round(float(r.amount), 2),
            "currency": r.currency,
            "source": r.source,
            "category_id": category_ids.get(r.category) if pd.notna(r.category) else None,
            "category_source": "label" if pd.notna(r.category) else None,
            "txn_hash": r.txn_hash,
        }
        for r in df.itertuples(index=False)
    ]
    for start in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[start : start + CHUNK_SIZE]
        report.inserted += len(session.execute(_insert_ignore(session).values(chunk)).all())
    report.duplicates = len(rows) - report.inserted
    return report
