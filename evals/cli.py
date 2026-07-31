"""``rag-evals`` command-line interface (Epic 14 Phase 14.1, Epics 15 & 16).

All subcommands are real: ``extraction``/``judge-alignment``/``confidence-review``
run the extraction eval suite (Epic 15) and ``retrieval`` the retrieval eval
(Epic 16); ``diff``/``save-baseline`` cover both report types. Provider/
``Settings`` imports stay lazy inside the command bodies, so importing this
module (and rendering ``--help``) can never issue a live call.
``_build_llm_provider`` below is the **only** live-provider construction site
in the extraction harness — the driver takes the provider injected, and tests
monkeypatch this seam with ``FakeLLMProvider``. **Running** ``retrieval``
builds the default in-process search caller, which reaches real providers and
therefore needs operator credentials, a reachable database and Redis (see
``evals.retrieval._default_search``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from evals.retrieval import VALID_MODES, run_retrieval_eval

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
    k: Annotated[
        int,
        typer.Option(
            help=(
                "Top-k cutoff for the per-query breakdown and the baseline "
                "diff's dropped-from-top-k check, and the minimum search "
                "depth. The metric set itself is fixed: NDCG@10, Recall@5, "
                "Recall@10, MRR."
            )
        ),
    ] = 10,
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

    Exits 2 for a bad input — an unknown mode, a non-positive ``--k``, or a
    fixture set that is missing, half-present, or whose queries.tsv and
    qrels.tsv disagree on query ids.
    """
    # Validated here as well as in the runner so a typo dies with a clean CLI
    # error before any Settings/search construction.
    if mode not in VALID_MODES:
        typer.echo(
            f"error: invalid mode {mode!r}: expected one of {sorted(VALID_MODES)}", err=True
        )
        raise typer.Exit(2)
    if k < 1:
        typer.echo(f"error: invalid k {k!r}: expected a positive integer", err=True)
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
    """Diff a new report run against a committed baseline.

    Dispatches on the payload's ``report_type``: retrieval reports print the
    per-metric headline, per-query regressions, and the biggest NDCG@10 drops
    (Epic 16); extraction reports print per-metric deltas with
    ``[REGRESSION]`` flags (Epic 15). Either path exits 1 when a regression
    past its threshold is found. Exit 2 covers every input error: missing
    paths, malformed JSON, a run finalized ``failed``, a ``report_type``
    mismatch between the two sides, a payload failing structural validation,
    or two runs whose ``run`` blocks make them incomparable.
    """
    from evals.reports import diff_against_baseline

    try:
        result = diff_against_baseline(baseline_path, new_report_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(result.summary)
    if result.status == "incomparable":
        # Printed in full above (the warning plus the deltas), but a run-config
        # mismatch is an input error, not a quality regression: exit 2, not 1.
        raise typer.Exit(2)
    if result.status in ("regression", "regressions_detected"):
        # The two diff paths spell their regression status differently
        # (retrieval: "regression"; extraction: "regressions_detected") —
        # both mean the same thing to the exit-code contract.
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
