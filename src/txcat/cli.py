from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import pandas as pd
import typer

if TYPE_CHECKING:
    from txcat.ingest import ValidationResult

app = typer.Typer(help="Transaction categorization toolkit.", no_args_is_help=True)
data_app = typer.Typer(help="Download and build datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")
insights_app = typer.Typer(
    help="Recurring payments, monthly spend, anomalies.", no_args_is_help=True
)
app.add_typer(insights_app, name="insights")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


@app.callback()
def main() -> None:
    """txcat command-line interface."""


def _load(result: "ValidationResult") -> None:
    from txcat.db import get_engine, session_scope
    from txcat.ingest import load_transactions

    with session_scope(get_engine()) as session:
        report = load_transactions(session, result.valid)
    typer.echo(
        f"valid={len(result.valid)} rejected={len(result.rejected)} "
        f"inserted={report.inserted} duplicates={report.duplicates} "
        f"new_accounts={report.accounts_created} new_categories={report.categories_created}"
    )


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="CSV of transactions.")],
    rejects: Annotated[
        Path | None, typer.Option(help="Write rejected rows (with reasons) here.")
    ] = None,
    default_currency: Annotated[str, typer.Option(help="Currency for rows without one.")] = "ETB",
) -> None:
    """Validate a CSV and load the valid rows into the database."""
    from txcat.ingest import IngestionError, validate_transactions

    try:
        result = validate_transactions(pd.read_csv(path, dtype=str), default_currency)
    except IngestionError as exc:
        typer.secho(f"File rejected: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _load(result)
    if len(result.rejected):
        for idx, reason in result.rejected["reason"].head(10).items():
            typer.secho(f"  row {idx}: {reason}", fg=typer.colors.YELLOW)
        if rejects:
            result.rejected.to_csv(rejects, index_label="row")
            typer.echo(f"rejected rows written to {rejects}")


@app.command()
def match(
    descriptions: Annotated[list[str], typer.Argument(help="One or more raw descriptions.")],
) -> None:
    """Show how descriptions are cleaned and which merchant (if any) they map to."""
    from txcat.data.clean import clean_description
    from txcat.data.merchants import MerchantNormalizer

    normalizer = MerchantNormalizer()
    for raw in descriptions:
        cleaned = clean_description(raw)
        merchant = normalizer.match(cleaned)
        found = f"{merchant.name} [{merchant.category}]" if merchant else "no confident match"
        typer.echo(f"{raw!r}\n  cleaned:  {cleaned!r}\n  merchant: {found}")


def _markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (avoids a tabulate dependency)."""
    rows = [list(map(str, df.columns)), ["---"] * len(df.columns)]
    rows += [[str(v) for v in row] for row in df.itertuples(index=False)]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


@app.command()
def train(
    models: Annotated[
        list[str] | None,
        typer.Option("--model", "-m", help="Model(s) to train (repeatable). Default: all."),
    ] = None,
    register: Annotated[bool, typer.Option(help="Register the best model in MLflow.")] = True,
    report: Annotated[
        Path | None, typer.Option(help="Write the comparison table (markdown) here.")
    ] = Path("reports/model_comparison.md"),
) -> None:
    """Train and compare categorization models on a time-based split; log to MLflow."""
    from txcat.config import get_settings
    from txcat.models.embeddings import fastembed_available
    from txcat.models.train import SPECS, prepare_dataset, results_table, train_all

    known = {s.name for s in SPECS}
    if unknown := set(models or []) - known:
        raise typer.BadParameter(f"unknown model(s) {sorted(unknown)}; choose from {sorted(known)}")
    if not models and not fastembed_available():
        typer.secho(
            "fastembed not installed: skipping embedding models (uv sync --extra embeddings)",
            fg=typer.colors.YELLOW,
        )
    transactions = pd.read_parquet(PROCESSED_DIR / "transactions.parquet")
    lexicon_path = PROCESSED_DIR / "lexicon.parquet"
    lexicon = pd.read_parquet(lexicon_path) if lexicon_path.exists() else None
    data = prepare_dataset(transactions, lexicon)
    typer.echo(data.bounds.to_string(index=False))

    results = train_all(
        data, models or None, tracking_uri=get_settings().mlflow_tracking_uri, register=register
    )
    table = results_table(results).round(3)
    typer.echo(table.to_string(index=False))
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Model comparison (generated by `txcat train`)\n\n"
            + _markdown(table)
            + "\n\nSplit (per source, by date):\n\n"
            + _markdown(data.bounds)
            + "\n",
            encoding="utf-8",
        )
        # Machine-readable copy: shipped with the exported model for the dashboard.
        report.with_suffix(".json").write_text(
            table.to_json(orient="records", indent=2), encoding="utf-8"
        )
        typer.echo(f"wrote {report} (+ .json)")


@app.command()
def bootstrap(
    out: Annotated[Path, typer.Option(help="Where to export the served model.")] = Path(
        "models/champion"
    ),
) -> None:
    """One-shot setup for a fresh environment: migrate, build + load data, train, export.

    Idempotent: each step is skipped when its result already exists. `docker compose up`
    runs this before starting the API, so a fresh clone works with no manual steps.
    Without KAGGLE_API_TOKEN only the synthetic data is used.
    """
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import func, select

    from txcat.config import get_settings
    from txcat.db import Transaction, get_engine
    from txcat.models.train import REGISTERED_MODEL

    settings = get_settings()
    typer.secho("1/4 migrate database", bold=True)
    command.upgrade(Config("alembic.ini"), "head")

    processed = PROCESSED_DIR / "transactions.parquet"
    if processed.exists():
        typer.secho("2/4 data: already built", bold=True)
    else:
        typer.secho("2/4 data: download (if a Kaggle token is set) + build", bold=True)
        token = settings.kaggle_api_token
        if token and token.get_secret_value():
            try:
                data_download(force=False)
            except (typer.Exit, Exception):
                typer.secho("Kaggle download failed: continuing with synthetic data only")
        data_build(n_accounts=30, seed=42, load=False)

    with get_engine().connect() as conn:
        stored = conn.execute(select(func.count(Transaction.id))).scalar_one()
    if stored:
        typer.secho(f"3/4 load: {stored} transactions already in the database", bold=True)
    else:
        typer.secho("3/4 load transactions into the database", bold=True)
        from txcat.data.build import INGEST_COLUMNS
        from txcat.ingest import validate_transactions

        df = pd.read_parquet(processed)
        _load(validate_transactions(df[INGEST_COLUMNS].astype({"date": str, "amount": str})))

    if out.exists():
        typer.secho(f"4/4 model: {out} already exported", bold=True)
        return
    typer.secho("4/4 train baseline + TF-IDF, register, export", bold=True)
    names = ["rule-baseline", "tfidf-logreg"]
    if (PROCESSED_DIR / "lexicon.parquet").exists():
        names.append("tfidf-logreg-lexicon")
    train(models=names, register=True, report=Path("reports/model_comparison.md"))
    export_model(out=out, model_uri=f"models:/{REGISTERED_MODEL}@champion")


@app.command("export-model")
def export_model(
    out: Annotated[Path, typer.Option(help="Directory to write the model to.")] = Path(
        "models/champion"
    ),
    model_uri: Annotated[
        str | None, typer.Option(help="Registry URI. Default: MODEL_URI setting.")
    ] = None,
) -> None:
    """Copy the served model out of the MLflow registry into a portable directory."""
    from txcat.api.predictor import export_predictor
    from txcat.config import get_settings

    settings = get_settings()
    uri = model_uri or settings.model_uri
    if out.exists():
        raise typer.BadParameter(f"{out} already exists; remove it first")
    export_predictor(uri, out, settings.mlflow_tracking_uri)
    typer.echo(f"exported {uri} -> {out}")


@insights_app.command("show")
def insights_show(
    account_id: Annotated[str, typer.Argument(help="Account id, e.g. SYN-0001.")],
    months: Annotated[int, typer.Option(help="Months of spend to show.")] = 3,
    anomalies: Annotated[int, typer.Option(help="Most recent anomalies to show.")] = 5,
) -> None:
    """Recurring payments, recent monthly spend and anomalies for one account."""
    from txcat.insights.anomaly import score_anomalies
    from txcat.insights.recurring import detect_recurring
    from txcat.insights.spend import spend_table

    df = pd.read_parquet(PROCESSED_DIR / "transactions.parquet")
    df = df[df["account_id"] == account_id]
    if df.empty:
        raise typer.BadParameter(f"no transactions for account {account_id!r}")
    currency = df["currency"].mode().iat[0]

    recurring = detect_recurring(df)
    typer.secho(f"Recurring payments ({len(recurring)})", bold=True)
    for r in recurring.sort_values("typical_amount", ascending=False).itertuples():
        typer.echo(
            f"  {r.counterparty:<30} {r.direction:<3} {r.period:<9} "
            f"~{r.typical_amount:>10,.0f} {currency}  next ~{r.next_expected:%Y-%m-%d}"
        )
    typer.secho(f"\nSpend by category, last {months} months ({currency})", bold=True)
    typer.echo(spend_table(df, account_id, months).round(0).to_string())

    scored = score_anomalies(df)
    flagged = scored[scored["is_anomaly"]].sort_values("date").tail(anomalies)
    typer.secho(f"\nRecent anomalies ({int(scored['is_anomaly'].sum())} total)", bold=True)
    for r in flagged.itertuples():
        typer.echo(
            f"  {r.date:%Y-%m-%d}  {r.amount:>12,.2f}  z={r.anomaly_z:>5.1f}  "
            f"{r.category:<20} {r.description}"
        )


@insights_app.command("evaluate")
def insights_evaluate(
    report: Annotated[
        Path | None, typer.Option(help="Write the evaluation table (markdown) here.")
    ] = Path("reports/insights_evaluation.md"),
) -> None:
    """Score the detectors against the synthetic ground truth (dev vs held-out accounts)."""
    from txcat.insights.evaluate import evaluation_table

    table, best_z = evaluation_table(pd.read_parquet(PROCESSED_DIR / "transactions.parquet"))
    typer.echo(table.to_string(index=False))
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Insight detectors vs synthetic ground truth (generated by "
            "`txcat insights evaluate`)\n\n"
            f"Anomaly z threshold chosen on dev accounts: {best_z}\n\n" + _markdown(table) + "\n",
            encoding="utf-8",
        )
        typer.echo(f"wrote {report}")


@data_app.command("download")
def data_download(
    force: Annotated[bool, typer.Option(help="Re-download even if present.")] = False,
) -> None:
    """Download the Kaggle source datasets into data/raw/kaggle/."""
    from txcat.data.kaggle import KaggleDownloadError, download

    try:
        paths = download(RAW_DIR, force=force)
    except KaggleDownloadError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    for slug, path in paths.items():
        typer.echo(f"{slug}: {path}")


@data_app.command("build")
def data_build(
    n_accounts: Annotated[int, typer.Option(help="Synthetic accounts to generate.")] = 30,
    seed: Annotated[int, typer.Option(help="Random seed for the generator.")] = 42,
    load: Annotated[bool, typer.Option(help="Also ingest the result into Postgres.")] = False,
) -> None:
    """Generate synthetic data, clean + normalise everything, write data/processed/."""
    from txcat.data.build import INGEST_COLUMNS, build_dataset

    df, report = build_dataset(RAW_DIR, PROCESSED_DIR, n_accounts=n_accounts, seed=seed)
    for dataset, n in report.rows_by_dataset.items():
        typer.echo(f"{dataset}: {n} rows")
    if report.lexicon_rows:
        typer.echo(f"kaggle_semantic_lexicon: {report.lexicon_rows} phrases")
    typer.echo(f"merchant coverage: {report.merchant_coverage:.1%}")
    if report.merchant_accuracy is not None:
        typer.echo(
            f"merchant accuracy vs synthetic ground truth: {report.merchant_accuracy:.1%} "
            f"(precision {report.merchant_precision:.1%})"
        )
    for slug in report.missing_sources:
        typer.secho(f"missing {slug}: run `txcat data download`", fg=typer.colors.YELLOW)
    for path in report.outputs:
        typer.echo(f"wrote {path}")
    if load:
        from txcat.ingest import validate_transactions

        _load(validate_transactions(df[INGEST_COLUMNS].astype({"date": str, "amount": str})))
