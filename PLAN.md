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

## Phase 2 — Data ✅

- [x] Kaggle download script for a bank-transaction categorization dataset (token from `.env`)
- [x] Faker-based synthetic generator for Ethiopian-style descriptions (telebirr airtime,
      CBE transfer, Ethio Telecom, mixed Amharic/English), every row `source='synthetic'`
- [x] Cleaning: lower-casing, stripping reference numbers/dates, unicode normalisation
- [x] Merchant normalisation with rapidfuzz against a canonical merchant list
- [x] Unified category taxonomy + mapping from the Kaggle labels
- Sources chosen: `bukolafatunde/personal-finance` (806 real-format labelled transactions)
  and `lowlevelexe/malaysian-semantic-transaction-dataset` (32k labelled merchant/keyword
  phrases, CDLA-Permissive). No large public labelled bank-description dataset exists, so
  the synthetic Ethiopian set (~33k rows, 30 accounts, 2023-01 → 2025-06) carries volume.
- **Exit criteria:** `txcat data download && txcat data build` produces a processed parquet with a
  `source` column; tests cover the cleaner, normaliser and generator determinism (seeded).

## Phase 3 — Models ✅

- [x] Time-based train/validation/test split (per source: 70/15/15 by date)
- [x] Rule/keyword baseline (merchant catalog + English/Amharic keyword rules)
- [x] TF-IDF (word + char_wb 2-5 n-grams) + Logistic Regression, with/without lexicon
- [x] Multilingual sentence embeddings + LogReg / LightGBM. Uses fastembed (ONNX Runtime,
      quantized paraphrase-multilingual-MiniLM-L12-v2, ~240 MB) instead of PyTorch +
      sentence-transformers to avoid a 1-2 GB download; optional `embeddings` extra
- [x] All runs logged to MLflow (params, macro-F1 + per-class F1 per slice, confusion
      matrix, threshold); best model registered as `txcat-categorizer@champion`
- [x] Confidence threshold for the review queue, chosen on validation per source
- Result: `tfidf-logreg-lexicon` wins, with Kaggle test macro-F1 0.991 vs 0.915 for the
  baseline and 0.996 vs 0.920 on unseen text. Embeddings underperform TF-IDF on unseen
  text (0.92-0.96). See README.
- **Exit criteria:** comparison table of macro-F1 (overall, real vs synthetic slices) in
  the README; best model beats the baseline on the future test slice.

## Phase 4 — Insights ✅

- [x] Recurring-payment detector (same merchant, regular interval, stable amount)
- [x] Monthly spend by category
- [x] Anomaly flag (amount vs. merchant/category history, robust z-score / IsolationForest)
- [x] Scored against synthetic ground truth (`is_recurring_true`, `is_injected_anomaly`):
      settings chosen on dev accounts, reported on held-out accounts
- Result: recurring precision 0.999 / recall 0.976. Anomalies: F1 1.0 on recurring bills,
  0.17 (z) / 0.21 (IsolationForest) on discretionary spend, where the generator's own
  amount spread hides 6-12x spikes. See README.
- **Exit criteria:** unit tests with hand-built histories for each detector.

## Phase 5 — API (FastAPI) ✅

- [x] `POST /categorize` (batch, up to 1000) returns category + confidence + merchant
- [x] `GET /accounts/{id}/insights` (recurring, monthly spend, anomalies from Postgres)
- [x] Predictions below the Phase 3 threshold go to a `review_queue` table (migration
      0002); `GET /review`, `POST /review/{id}` saves the corrected label for retraining
- [x] `GET /health` reports the served model version (`txcat-categorizer@champion`)
- [x] `api` service in docker-compose, serving a model exported with `txcat export-model`
- No auth yet: an API-key header is planned before any public deployment
- **Exit criteria:** API tests with `TestClient`; OpenAPI docs render.

## Phase 6 — Ship ✅

- [x] Streamlit dashboard (insights, try-it/upload, review queue, model page), API-only client
- [x] README with architecture diagram, results table, how to run, screenshots
- [x] MODEL_CARD.md (data, metrics, limitations, synthetic-data disclosure)
- [x] Dockerfile + `docker compose up` brings up db, bootstrap (data + train + export), api
      and dashboard. MLflow runs inside bootstrap (SQLite store) rather than as a server;
      the API serves the exported skops model, so its image needs no MLflow
- [x] Verified from a fresh `git clone`: `docker compose up` → dashboard works end-to-end
      (synthetic data only without a Kaggle token). Images: api 871 MB (was 2.45 GB),
      dashboard 861 MB, bootstrap 1.48 GB
- **Exit criteria:** fresh clone → `docker compose up` → dashboard works end-to-end;
  CI green.
