"""``rag-evals`` command-line interface (Epic 14 Phase 14.1, Epic 16).

``retrieval`` is real (Epic 16); extraction/judge subcommands remain scaffold
stubs that print ``not implemented yet`` and exit 0 until Epic 15. Importing
this module issues no live call — but **running** ``retrieval`` builds the
default in-process search caller, which reaches real providers and therefore
needs operator credentials, a reachable database and Redis (see
``evals.retrieval._default_search``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from evals.retrieval import VALID_MODES, run_retrieval_eval

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
    mode: Annotated[
        str,
        typer.Option(help="Retrieval mode: hybrid, keyword, or vector."),
    ] = "hybrid",
    label: Annotated[
        str,
        typer.Option(help="Run label used in the report directory name."),
    ] = "retrieval",
) -> None:
    """Evaluate retrieval quality against golden queries/qrels (Epic 16).

    Exits 2 for a bad input — an unknown mode, or a fixture set that is
    missing, half-present, or whose queries.tsv and qrels.tsv disagree on
    query ids.
    """
    # Validated here as well as in the runner so a typo dies with a clean CLI
    # error before any Settings/search construction.
    if mode not in VALID_MODES:
        typer.echo(
            f"error: invalid mode {mode!r}: expected one of {sorted(VALID_MODES)}", err=True
        )
        raise typer.Exit(2)
    try:
        report = asyncio.run(
            run_retrieval_eval(query_set=queries, k=k, label=label, mode=mode)
        )
    except (FileNotFoundError, ValueError) as exc:
        # Fixture problems are caller errors, not eval results: surface them as
        # exit 2 instead of an unhandled traceback exiting 1.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(str(report.path))


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
    """Diff a new report run against a committed baseline.

    Retrieval reports print the per-metric headline, per-query regressions,
    and the biggest NDCG@10 drops, and exit 1 when a regression past the
    threshold is found (extraction diffs land with Epic 15). Exit 2 covers
    every input error: missing paths, malformed JSON, a run finalized
    ``failed``, or a ``report_type`` mismatch between the two sides.
    """
    from evals.reports import diff_against_baseline

    try:
        result = diff_against_baseline(baseline_path, new_report_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(result.summary)
    if result.status == "regression":
        raise typer.Exit(1)


@app.command("save-baseline")
def save_baseline(
    report_path: Annotated[
        Path, typer.Argument(help="Report run directory containing results.json.")
    ],
    name: Annotated[
        str,
        typer.Option(help="Baseline name; written to evals/baselines/<name>.json."),
    ],
) -> None:
    """Promote a report run's results.json to a committed baseline."""
    from evals.reports import save_as_baseline

    try:
        path = save_as_baseline(report_path, name)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(str(path))


if __name__ == "__main__":
    app()
