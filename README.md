# Transaction Categorization & Financial Insights

End-to-end ML project: categorize bank transactions from their descriptions (English and
Ethiopian-style mixed Amharic/English), then derive insights such as recurring payments,
monthly spend by category and anomalies. See [PLAN.md](PLAN.md) for the phased roadmap.

> Work in progress: Phases 1–3 done (ingestion, data, models). Next: insights.

## Quickstart

```bash
cp .env.example .env            # fill in values; .env is git-ignored
docker compose up -d db         # PostgreSQL 16 on localhost:5433
uv sync
uv run alembic upgrade head     # create tables
uv run txcat data download      # Kaggle sources -> data/raw/kaggle/ (needs KAGGLE_API_TOKEN)
uv run txcat data build --load  # synthetic + clean + merchants -> data/processed/, and Postgres
uv run txcat train              # baseline + TF-IDF models, logged to MLflow (mlflow.db)
uv run pytest
```

Optional embedding models (fastembed on ONNX Runtime, no PyTorch; ~240 MB model download
on first use, cached in `data/models/fastembed/`):

```bash
uv sync --extra embeddings
uv run txcat train              # now also trains embed-logreg and embed-lgbm
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs at :5000
```

Try the cleaning + merchant matching on any description:

```bash
uv run txcat match "CBE transfer to Abebe K. FT24123ABCD" "ቤት ኪራይ ጥቅምት" "Grocery Store"
```

(On Windows PowerShell, run `[Console]::OutputEncoding = [Text.Encoding]::UTF8;
$env:PYTHONIOENCODING = "utf-8"` first so Amharic prints correctly.)

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

## Models (Phase 3)

All models take the same input (cleaned description + signed amount, used only for
money-in/out direction) and are scikit-learn estimators, so they're evaluated, logged to
MLflow and served the same way.

| Model | What it is |
|---|---|
| `rule-baseline` | Merchant catalog match → category; else English/Amharic keyword rules; else most common class for money in/out |
| `tfidf-logreg` | Word (1–2) + `char_wb` (2–5) n-gram TF-IDF, logistic regression |
| `tfidf-logreg-lexicon` | Same, plus the 32k Kaggle lexicon phrases as extra training text |
| `embed-logreg` / `embed-lgbm` | `paraphrase-multilingual-MiniLM-L12-v2` (quantized ONNX via fastembed, 384-d) + LogReg / LightGBM |

**Evaluation protocol.** Time-based split *per source* (the sources cover different years):
oldest 70% train, next 15% validation, newest 15% test. Hyperparameters and the
review-queue threshold are chosen on validation; test is scored once. Model selection
uses the mean of the synthetic and Kaggle macro-F1, so the 33k synthetic rows can't drown
out the real-format ones. The **unseen text** slice is test rows whose cleaned
description never appears in training.

**Results on the future test slice** (from `reports/model_comparison.md`):

| Model | Macro-F1 all | Synthetic | Kaggle | Unseen text (812 rows) | Review threshold | Auto-accepted | Accuracy of auto-accepted |
|---|---|---|---|---|---|---|---|
| rule-baseline | 0.988 | 0.988 | 0.915 | 0.920 | 0.70 | 98.6% | 99.7% |
| tfidf-logreg | 1.000 | 1.000 | 0.983 | 0.997 | 0.22 | 100% | 100% |
| **tfidf-logreg-lexicon** (registered) | 1.000 | 1.000 | **0.991** | **0.996** | 0.42 | 99.9% | 100% |
| embed-logreg | 0.994 | 0.994 | 0.992 | 0.957 | 0.92 | 97.5% | 99.9% |
| embed-lgbm | 0.988 | 0.987 | 1.000 | 0.923 | 0.53 | 99.8% | 99.5% |

**How to read this honestly:**

- **Synthetic scores are near-perfect because the data is template-generated.** The
  generator draws from a fixed set of merchants and templates, so the future slice
  looks a lot like training. These numbers show the pipeline works; they are not an
  estimate of accuracy on real statements.
- **The Kaggle slice is small (123 test rows) and repetitive.** 121 of its 123 test
  descriptions also appear in training (same merchants every month), so one error moves
  macro-F1 by several points. The 0.91 → 0.99 gain over the baseline is real but noisy.
- **Unseen text is the most informative column.** On descriptions never seen in training
  (mostly new noise variants: typos, references, mixed scripts), character n-gram TF-IDF
  holds at 0.996, while the general-purpose sentence embeddings drop to 0.92–0.96.
  Merchant names and statement noise aren't what those embeddings were trained on.
- **Winner: TF-IDF + LogReg with the lexicon.** It beats the rule baseline on every
  slice, needs no model download, and trains in seconds. It is registered in MLflow as
  `txcat-categorizer@champion` with its review threshold as a version tag.
- **Review queue:** the threshold is the lowest confidence at which auto-accepted
  validation predictions reach 97% accuracy *on each source*. Predictions below it go to
  the review queue (Phase 5).
