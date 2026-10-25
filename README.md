# Transaction Categorization & Financial Insights

End-to-end ML project: categorize bank transactions from their descriptions (English and
Ethiopian-style mixed Amharic/English), then derive insights such as recurring payments,
monthly spend by category and anomalies. See [PLAN.md](PLAN.md) for the phased roadmap.

> Work in progress — currently at Phase 1 (skeleton + ingestion).

## Quickstart

```bash
cp .env.example .env            # fill in values; .env is git-ignored
docker compose up -d db         # PostgreSQL 16 on localhost:5433
uv sync
uv run alembic upgrade head     # create tables
uv run txcat ingest tests/fixtures/sample_transactions.csv --rejects rejects.csv
uv run pytest
```

## Data contract

Input CSV columns: `account_id`, `date`, `description`, `amount` (signed: negative = money
out), `source` (`kaggle` | `synthetic` | `user`), optional `currency` (ISO-4217, default
`ETB`) and `category`. Rows failing the Pandera schema are rejected with a per-row reason;
the valid rest is loaded idempotently (re-running the same file inserts nothing).

Synthetic rows are always tagged `source = 'synthetic'` and are never presented as real data.
