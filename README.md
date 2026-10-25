# Transaction Categorization & Financial Insights

End-to-end ML project: categorize bank transactions from their descriptions (English and
Ethiopian-style mixed Amharic/English), then derive insights such as recurring payments,
monthly spend by category and anomalies. See [PLAN.md](PLAN.md) for the phased roadmap.

> Work in progress: Phases 1–2 done (ingestion, data). Next: models.

## Quickstart

```bash
cp .env.example .env            # fill in values; .env is git-ignored
docker compose up -d db         # PostgreSQL 16 on localhost:5433
uv sync
uv run alembic upgrade head     # create tables
uv run txcat data download      # Kaggle sources -> data/raw/kaggle/ (needs KAGGLE_API_TOKEN)
uv run txcat data build --load  # synthetic + clean + merchants -> data/processed/, and Postgres
uv run pytest
```

## Data

| Dataset | Rows | What it is | Used for |
|---|---|---|---|
| `synthetic_et` (**synthetic**) | ~33k | Generated Ethiopian-style statements, 30 accounts, Jan 2023 – Jun 2025 | Training/eval, insights |
| [`bukolafatunde/personal-finance`](https://www.kaggle.com/datasets/bukolafatunde/personal-finance) | 806 | Real-format US personal transactions (2018), labelled | Training/eval, insights |
| [`lowlevelexe/malaysian-semantic-transaction-dataset`](https://www.kaggle.com/datasets/lowlevelexe/malaysian-semantic-transaction-dataset) | 32k | Merchant/keyword phrases with categories (no dates/amounts) | Extra training text |

There is no large, public, labelled bank-description dataset, so the synthetic generator
supplies most of the volume. **Synthetic rows are always tagged `source = 'synthetic'`**,
are never presented as real customer data, and metrics will be reported separately per source.

**Synthetic generator** (`txcat.data.synthetic`, seeded and deterministic): each account has
a profile (salary on payday, optional rent, iddir, equb, DStv, loan, utilities) producing
recurring payments, plus Poisson-distributed spending across categories. Descriptions mix
English and Amharic (~40% contain Ge'ez script) with statement-style noise: `FT…` references,
phone numbers, upper-casing, typos. 0.5% of rows are amount spikes flagged
`is_injected_anomaly` for evaluating the anomaly detector.

**Cleaning** (`txcat.data.clean`): NFKC, lower-case, strip references/dates/phone numbers,
keep Ge'ez script. **Merchant normalisation** (`txcat.data.merchants`): rapidfuzz
`token_set_ratio` against a catalog of ~65 merchants and their aliases (English + Amharic).
On synthetic data it gets **97% accuracy at 100% precision**: it would rather return no
merchant than a wrong one.

**Taxonomy** (`txcat.data.taxonomy`): 19 unified categories; every source label maps
explicitly, and an unmapped label fails the build.

## Data contract

Input CSV columns: `account_id`, `date`, `description`, `amount` (signed: negative = money
out), `source` (`kaggle` | `synthetic` | `user`), optional `currency` (ISO-4217, default
`ETB`) and `category`. Rows failing the Pandera schema are rejected with a per-row reason;
the valid rest is loaded idempotently (re-running the same file inserts nothing).
