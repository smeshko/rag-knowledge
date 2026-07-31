"""``rag-evals`` command-line interface (Epic 14 Phase 14.1, Epic 15).

``extraction`` runs the real extraction eval (Epic 15 Phase 15.1); the other
subcommands are scaffold stubs that print ``not implemented yet`` and exit 0
until Epics 15/16 fill them in. Provider/``Settings`` imports stay lazy inside
the command bodies, so importing this module (and rendering ``--help``) can
never issue a live call. ``_build_llm_provider`` below is the **only**
live-provider construction site in the eval harness — the driver takes the
provider injected, and tests monkeypatch this seam with ``FakeLLMProvider``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from rag_recipes.config import Settings
    from rag_recipes.providers.llm.base import LLMProvider

app = typer.Typer(help="rag-recipes evaluation harness")


def _build_llm_provider(settings: Settings) -> LLMProvider:
    """Construct the live extraction LLM provider from settings.

    Mirrors ``rag_recipes.ingestion.jobs._build_llm_provider`` (module-private,
    so not imported): dispatch on ``settings.llm_provider`` — ``"anthropic"``
    builds Claude on the Anthropic settings, anything else stays OpenAI. Not
    ``api.dependencies.get_llm_provider``, which resolves the *answer* model.
    Never exercised by tests (they monkeypatch this seam).
    """
    from rag_recipes.providers.llm.anthropic import AnthropicLLMProvider
    from rag_recipes.providers.llm.openai import OpenAILLMProvider

    if settings.llm_provider == "anthropic":
        api_key = settings.anthropic_api_key
        if api_key is None:
            raise ValueError("anthropic_api_key is required when llm_provider == 'anthropic'")
        return AnthropicLLMProvider(
            api_key,
            default_model=settings.anthropic_llm_model,
            max_rate_limit_retries=settings.llm_max_rate_limit_retries,
            request_timeout=settings.llm_request_timeout_seconds,
            max_tokens=settings.anthropic_max_tokens,
        )
    return OpenAILLMProvider(
        settings.openai_api_key,
        default_model=settings.llm_model,
        max_rate_limit_retries=settings.llm_max_rate_limit_retries,
        request_timeout=settings.llm_request_timeout_seconds,
    )


def _not_implemented(name: str) -> None:
    typer.echo(f"{name}: not implemented yet")


@app.command("extraction")
def extraction(
    fixtures: Annotated[
        str,
        typer.Option(help="Recipe fixture set under data/fixtures/synthetic_recipes/."),
    ],
    label: Annotated[
        str,
        typer.Option(help="Run label stamped into the report directory name and metadata."),
    ],
    judge: Annotated[
        str | None,
        typer.Option(help="Judge prompt name under data/fixtures/judge_prompts/ (Epic 15)."),
    ] = None,
) -> None:
    """Evaluate extraction quality against golden fixtures (Epic 15)."""
    import asyncio

    from evals.extraction import run_extraction_eval
    from rag_recipes.config import get_settings

    settings = get_settings()
    try:
        run = asyncio.run(
            run_extraction_eval(
                fixtures,
                label,
                llm_provider=_build_llm_provider(settings),
                judge=judge,
            )
        )
    except ValueError as exc:  # empty/unknown fixture set — a caller error
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"report: {run.path}")


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
    fixtures: Annotated[
        str,
        typer.Option(help="Recipe fixture set under data/fixtures/synthetic_recipes/."),
    ],
    report: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Extraction run directory to record the agreement metric into "
                "(default: the latest run under evals/reports/)."
            )
        ),
    ] = None,
) -> None:
    """Measure LLM-judge vs human-rating agreement (Epic 15)."""
    import asyncio

    from evals.alignment import run_judge_alignment
    from evals.reports import latest_run_dir
    from rag_recipes.config import get_settings

    run_dir = report if report is not None else latest_run_dir()
    try:
        result = asyncio.run(
            run_judge_alignment(
                judge,
                fixtures,
                llm_provider=_build_llm_provider(get_settings()),
                report_path=run_dir,
            )
        )
    except ValueError as exc:  # empty/unknown fixture set — a caller error
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    rate = "n/a" if result.agreement_rate is None else f"{result.agreement_rate:.2f}"
    typer.echo(f"Judge-human agreement ({result.judge_name}, {result.judge_version}): {rate}")
    for item in result.disagreements:
        typer.echo(f"DISAGREE {item.fixture_name}:")
        typer.echo(f"  human ({item.human_rating}): {item.human_critique}")
        typer.echo(f"  judge ({item.judge_rating}): {item.judge_critique}")
    if result.unrated:
        typer.echo(f"Unrated (judge error): {', '.join(result.unrated)}")
    if run_dir is not None:
        typer.echo(f"agreement recorded in: {run_dir}")


@app.command("confidence-review")
def confidence_review(
    report: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Extraction run directory to review "
                "(default: the latest run under evals/reports/)."
            )
        ),
    ] = None,
) -> None:
    """Review confidence calibration of the latest extraction eval (Epic 15)."""
    from evals.calibration import _render_block, run_confidence_review

    try:
        result = run_confidence_review(report)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(_render_block(result))
    typer.echo(f"calibration recorded in: {result.report_path}")


@app.command("diff")
def diff(
    baseline_path: Annotated[Path, typer.Argument(help="Committed baseline JSON file.")],
    new_report_path: Annotated[Path, typer.Argument(help="New report run directory.")],
) -> None:
    """Diff a new report run against a committed baseline (skeleton — Epics 15/16)."""
    from evals.reports import diff_against_baseline

    try:
        result = diff_against_baseline(baseline_path, new_report_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(result.summary)


if __name__ == "__main__":
    app()
