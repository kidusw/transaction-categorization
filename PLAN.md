# Project 1 — Transaction Categorization & Financial Insights

Categorize bank transactions from their free-text descriptions (English + Ethiopian-style
mixed Amharic/English), then turn categorized history into insights: recurring payments,
monthly spend by category, and anomaly flags. Served via FastAPI with a review queue for
low-confidence predictions, and a Streamlit dashboard.

Package name: `txcat`. Rules from the portfolio `PLAN.md` apply (time-based splits, honest
metrics, synthetic data labelled synthetic, secrets/data git-ignored).

---

## Phase 1 — Template + ingestion ✅

- [x] Repo skeleton: `uv` project (Python 3.12), src layout, ruff + pytest config
- [x] `.gitignore` (covers `.env`, `data/`, `mlruns/`), `.env.example` with blank values
- [x] Settings via pydantic-settings (`DATABASE_URL`, `KAGGLE_API_TOKEN`)
- [x] PostgreSQL 16 via `docker-compose.yml`
- [x] SQLAlchemy models: `accounts`, `categories`, `transactions`; Alembic initial migration
- [x] Pandera schema for raw transactions; CSV → validate → load ingestion CLI
- [x] GitHub Actions CI: ruff + pytest
- **Exit criteria:** `uv run pytest` and `uv run ruff check` green; `alembic upgrade head`
  creates the 3 tables in Postgres; a sample CSV ingests and invalid rows are rejected with
  a clear report.

## Phase 2 — Data

- [ ] Kaggle download script for a bank-transaction categorization dataset (token from `.env`)
- [ ] Faker-based synthetic generator for Ethiopian-style descriptions (telebirr airtime,
      CBE transfer, Ethio Telecom, mixed Amharic/English), every row `source='synthetic'`
- [ ] Cleaning: lower-casing, stripping reference numbers/dates, unicode normalisation
- [ ] Merchant normalisation with rapidfuzz against a canonical merchant list
- [ ] Unified category taxonomy + mapping from the Kaggle labels
- **Exit criteria:** `make data` (or equivalent CLI) produces a processed parquet with a
  `source` column; tests cover the cleaner, normaliser and generator determinism (seeded).

## Phase 3 — Models

- [ ] Time-based train/validation/test split
- [ ] Rule/keyword baseline
- [ ] TF-IDF (word + char n-grams) + Logistic Regression
- [ ] Multilingual sentence-transformer embeddings + LogReg / LightGBM
- [ ] All runs logged to MLflow (params, macro-F1, per-class report, confusion matrix);
      best model registered
- **Exit criteria:** comparison table of macro-F1 (overall, real vs synthetic slices) in
  the README; best model beats the baseline on the future test slice.

## Phase 4 — Insights

- [ ] Recurring-payment detector (same merchant, regular interval, stable amount)
- [ ] Monthly spend by category
- [ ] Anomaly flag (amount vs. merchant/category history, robust z-score / IsolationForest)
- **Exit criteria:** unit tests with hand-built histories for each detector.

## Phase 5 — API (FastAPI)

- [ ] `POST /categorize` (batch) returns category + confidence
- [ ] `GET /accounts/{id}/insights`
- [ ] Predictions below a confidence threshold go to a `review_queue` table;
      `GET /review-queue`, `POST /review-queue/{id}` to accept/correct labels
- **Exit criteria:** API tests with `TestClient`; OpenAPI docs render.

## Phase 6 — Ship

- [ ] Streamlit dashboard (upload/categorize, insights, review queue)
- [ ] README with architecture diagram, results table, how to run
- [ ] MODEL_CARD.md (data, metrics, limitations, synthetic-data disclosure)
- [ ] Dockerfile + `docker compose up` brings up db, api, dashboard, mlflow
- **Exit criteria:** fresh clone → `docker compose up` → dashboard works end-to-end;
  CI green.
