from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from txcat.db import get_engine, session_scope
from txcat.ingest import IngestionError, ValidationResult, load_transactions, validate_transactions

app = typer.Typer(help="Transaction categorization toolkit.", no_args_is_help=True)
data_app = typer.Typer(help="Download and build datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


@app.callback()
def main() -> None:
    """txcat command-line interface."""


def _load(result: ValidationResult) -> None:
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
        _load(validate_transactions(df[INGEST_COLUMNS].astype({"date": str, "amount": str})))
