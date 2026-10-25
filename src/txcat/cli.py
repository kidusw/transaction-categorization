from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from txcat.db import get_engine, session_scope
from txcat.ingest import IngestionError, load_transactions, validate_transactions

app = typer.Typer(help="Transaction categorization toolkit.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """txcat command-line interface."""


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

    with session_scope(get_engine()) as session:
        report = load_transactions(session, result.valid)

    typer.echo(
        f"valid={len(result.valid)} rejected={len(result.rejected)} "
        f"inserted={report.inserted} duplicates={report.duplicates} "
        f"new_accounts={report.accounts_created} new_categories={report.categories_created}"
    )
    if len(result.rejected):
        for idx, reason in result.rejected["reason"].head(10).items():
            typer.secho(f"  row {idx}: {reason}", fg=typer.colors.YELLOW)
        if rejects:
            result.rejected.to_csv(rejects, index_label="row")
            typer.echo(f"rejected rows written to {rejects}")
