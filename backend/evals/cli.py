"""``rag-evals`` command-line interface (Epic 14 Phase 14.1).

Scaffold stubs: every subcommand exists with its final argument shape but
prints ``not implemented yet`` and exits 0. Extraction/judge subcommands gain
real behaviour in Epic 15; retrieval metrics in Epic 16. The commands import no
``Settings``, no provider, and no DB code — nothing here can issue a live call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="rag-recipes evaluation harness")


def _not_implemented(name: str) -> None:
    typer.echo(f"{name}: not implemented yet")


@app.command("extraction")
def extraction(
    fixtures: Annotated[
        str,
        typer.Option(help="Recipe fixture set under data/fixtures/synthetic_recipes/."),
    ],
    judge: Annotated[
        str | None,
        typer.Option(help="Judge prompt name under data/fixtures/judge_prompts/ (Epic 15)."),
    ] = None,
) -> None:
    """Evaluate extraction quality against golden fixtures (Epic 15)."""
    _not_implemented("extraction")


@app.command("retrieval")
def retrieval(
    queries: Annotated[
        str,
        typer.Option(help="Query fixture set under data/fixtures/queries/."),
    ],
    k: Annotated[int, typer.Option(help="Rank cutoff for retrieval metrics.")] = 10,
) -> None:
    """Evaluate retrieval quality against golden queries/qrels (Epic 16)."""
    _not_implemented("retrieval")


@app.command("judge-alignment")
def judge_alignment(
    judge: Annotated[
        str,
        typer.Option(help="Judge prompt name under data/fixtures/judge_prompts/."),
    ],
) -> None:
    """Measure LLM-judge vs human-rating agreement (Epic 15)."""
    _not_implemented("judge-alignment")


@app.command("confidence-review")
def confidence_review() -> None:
    """Review low-confidence extractions for judge calibration (Epic 15)."""
    _not_implemented("confidence-review")


@app.command("diff")
def diff(
    baseline_path: Annotated[Path, typer.Argument(help="Committed baseline JSON file.")],
    new_report_path: Annotated[Path, typer.Argument(help="New report run directory.")],
) -> None:
    """Diff a new report run against a committed baseline (skeleton — Epics 15/16)."""
    from evals.reports import diff_against_baseline

    try:
        result = diff_against_baseline(baseline_path, new_report_path)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(result.summary)


if __name__ == "__main__":
    app()
