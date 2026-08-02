"""``rag-evals`` command-line interface (Epic 14 Phase 14.1, Epics 15 & 16).

All subcommands are real: ``extraction``/``judge-alignment``/``confidence-review``
run the extraction eval suite (Epic 15) and ``retrieval`` the retrieval eval
(Epic 16); ``diff``/``save-baseline`` cover both report types; the ``fixtures``
sub-app cuts real cookbook page ranges into fixture candidates (Epic 23.1).
Provider/``Settings`` imports stay lazy inside the command bodies, so importing
this module (and rendering ``--help``) can never issue a live call — that
includes ``evals.fixture_cutter``, which pulls in the PyMuPDF provider.
``_build_llm_provider`` and ``_build_judge_provider`` below are the **only**
live-provider construction sites in the extraction harness — the drivers take
providers injected, and tests monkeypatch these seams with ``FakeLLMProvider``.
The two are separate so a cross-provider comparison is not each candidate
grading itself (Epic 23.3). **Running** ``retrieval`` builds the default
in-process search caller, which reaches real providers and therefore needs
operator credentials, a reachable database and Redis (see
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

fixtures_app = typer.Typer(help="Author and maintain evaluation fixtures.")
app.add_typer(fixtures_app, name="fixtures")

#: Duplicated from ``evals.fixture_cutter.DEFAULT_MIN_TEXT_CHARS`` (itself
#: mirroring ``Settings.pdf_min_text_chars_for_page``) because a Typer default is
#: evaluated at *decoration* time — importing the cutter to read it would drag
#: the PyMuPDF provider into ``--help``. A unit test pins the two together.
_CUT_MIN_TEXT_CHARS = 20


def _build_llm_provider(settings: Settings) -> LLMProvider:
    """Construct the live extraction LLM provider from settings.

    Delegates to the provider registry (Epic 23.4), which resolves the
    *extraction* model — not ``api.dependencies.get_llm_provider``, which resolves
    the *answer* model. This wrapper survives as the monkeypatch seam the tests
    replace with a ``FakeLLMProvider``; it is never itself exercised by a test.

    The import stays inside the body so importing this module — and rendering
    ``--help`` — cannot reach a provider, per the module docstring.
    """
    from rag_recipes.providers.llm.registry import build_llm_provider

    return build_llm_provider(settings)


def _build_judge_provider(settings: Settings) -> LLMProvider | None:
    """Construct a separate provider for the LLM judge, or ``None`` (Epic 23.3).

    ``None`` when neither ``judge_llm_provider`` nor ``judge_llm_model`` is set —
    and that is deliberately ``None`` rather than "a second provider configured
    identically". The driver then reuses the *same object* for both roles, which
    is exactly today's behaviour, rather than an equivalent-but-distinct one that
    would double provider construction for no benefit.

    When set, the provider name and model are resolved independently. The model
    falls back to the **judge provider's own** extraction model, never to
    ``llm_model`` — sending the extraction provider's model id to a different
    vendor is the failure Epic 19.1 DECISIONS #5 exists to prevent, and it would
    reappear here for anyone who set ``JUDGE_LLM_PROVIDER`` alone.

    Why this exists at all: with one provider serving both roles, a cross-provider
    comparison grades every candidate with itself.
    """
    from rag_recipes.providers.llm.registry import build_llm_provider, resolve_extraction_model

    if settings.judge_llm_provider is None and settings.judge_llm_model is None:
        return None
    name = settings.judge_llm_provider or settings.llm_provider
    model = settings.judge_llm_model or resolve_extraction_model(settings, name)
    return build_llm_provider(settings, provider_name=name, model=model)


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
                "Extraction run directory whose persisted artifacts are rated; "
                "the agreement metric is recorded into it "
                "(default: the latest run under evals/reports/)."
            )
        ),
    ] = None,
) -> None:
    """Measure LLM-judge vs human-rating agreement (Epic 15, Epic 20.1).

    Rates the run's *persisted* extracted artifacts — never re-extracts. Exits
    2 for every unusable run (none at all, missing or malformed results.json, a
    failed run, a retrieval run, a fixture-set mismatch, a drifted fixture),
    checked *before* ``Settings`` or a provider are constructed so a missing
    API key can never mask the real error.
    """
    import asyncio

    from evals.alignment import load_alignment_run, run_judge_alignment
    from evals.fixtures import load_recipe_fixtures
    from evals.reports import latest_run_dir

    run_dir = report if report is not None else latest_run_dir()
    try:
        fixture_list = load_recipe_fixtures(fixtures)
        if not fixture_list:
            raise ValueError(
                f"recipe fixture set {fixtures!r} is empty or does not exist; "
                f"nothing to align"
            )
        # Pre-flight (DECISIONS #9): validate the run before get_settings() /
        # _build_llm_provider are evaluated — a config error must never mask
        # the "no run to align against" message.
        load_alignment_run(run_dir, fixtures, fixture_list)

        from rag_recipes.config import get_settings

        result = asyncio.run(
            run_judge_alignment(
                judge,
                fixtures,
                llm_provider=_build_llm_provider(get_settings()),
                report_path=run_dir,
            )
        )
    except (FileNotFoundError, ValueError) as exc:  # unusable run/fixtures — caller error
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    rate = "n/a" if result.agreement_rate is None else f"{result.agreement_rate:.2f}"
    typer.echo(f"Judge-human agreement ({result.judge_name}, {result.judge_version}): {rate}")
    for item in result.disagreements:
        typer.echo(f"DISAGREE {item.fixture_name}:")
        typer.echo(f"  human ({item.human_rating}): {item.human_critique}")
        typer.echo(f"  judge ({item.judge_rating}): {item.judge_critique}")
    if result.unrated:
        typer.echo(f"Unrated: {', '.join(result.unrated)}")
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


def _parse_page_range(spec: str) -> tuple[int, int]:
    """Parse ``"42-43"`` (or a bare ``"42"``) into an inclusive 1-based range.

    Raises ``ValueError`` with the offending spec named; the caller turns that
    into exit 2. Bounds themselves (inversion, out-of-document) are the cutter's
    job — this only rejects what is not a page range at all.
    """
    text = spec.strip()
    first_text, sep, last_text = text.partition("-")
    if not sep:
        first_text = last_text = text
    try:
        first_page = int(first_text)
        last_page = int(last_text)
    except ValueError as exc:
        raise ValueError(
            f"invalid --pages {spec!r}: expected N-M (inclusive, 1-based) or a single page N"
        ) from exc
    return first_page, last_page


@fixtures_app.command("cut")
def fixtures_cut(
    pdf: Annotated[
        Path,
        typer.Option(help="Source cookbook PDF to cut from."),
    ],
    fixture_set: Annotated[
        str,
        typer.Option("--set", help="Fixture set under data/fixtures/synthetic_recipes/."),
    ],
    pages: Annotated[
        list[str],
        typer.Option(
            "--pages",
            help=(
                "Inclusive 1-based page range 'N-M' (or a single page 'N'). "
                "Repeat for a batch; the book is extracted once for the whole batch."
            ),
        ),
    ],
    rationale: Annotated[
        list[str],
        typer.Option(
            "--rationale",
            help=(
                "Why this window was chosen, recorded in notes.md. Repeat once per "
                "--pages. Rationales are paired with ranges positionally — the Nth "
                "--rationale goes with the Nth --pages, regardless of where they sit "
                "on the command line — so always interleave them strictly: "
                "--pages A --rationale a --pages B --rationale b. Only the counts "
                "are checked, so a mispaired batch is accepted silently."
            ),
        ),
    ],
    min_text_chars: Annotated[
        int,
        typer.Option(help="Sparse-page threshold; matches PDF_MIN_TEXT_CHARS_FOR_PAGE."),
    ] = _CUT_MIN_TEXT_CHARS,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing fixture of the same derived name."),
    ] = False,
    invalidate_goldens: Annotated[
        bool,
        typer.Option(
            help=(
                "Also allow overwriting a fixture that carries expected.json. "
                "The golden then describes text that no longer exists — re-author it."
            )
        ),
    ] = False,
) -> None:
    """Cut real cookbook PDF page ranges into fixture candidates (Epic 23.1).

    Fixture names are derived (``<book-stem>-p<first>-<last>``), never supplied:
    the name becomes the span id Phase 23.2's goldens cite. The set this writes
    carries **no** ``expected.json`` — goldens are 23.2, so
    ``rag-evals extraction`` cannot score it yet.

    Every caller error exits 2 with the problem named: an unparseable, inverted
    or out-of-document ``--pages``, a missing ``--pdf``, a ``--rationale`` count
    that does not match ``--pages``, an unsafe ``--set``, or a collision without
    ``--overwrite``.
    """
    import asyncio

    from evals.fixture_cutter import FixtureCutterError, cut_fixtures

    try:
        ranges = [_parse_page_range(spec) for spec in pages]
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    try:
        results = asyncio.run(
            cut_fixtures(
                pdf,
                fixture_set=fixture_set,
                ranges=ranges,
                rationales=rationale,
                min_text_chars=min_text_chars,
                overwrite=overwrite,
                invalidate_goldens=invalidate_goldens,
            )
        )
    except FixtureCutterError as exc:
        # Every documented caller error is one of the cutter's named types
        # (all ValueError subclasses); a bare ValueError/FileNotFoundError
        # reaching here would be a bug in the cutter's error contract.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    for result in results:
        typer.echo(f"{result.name}\t{result.path}")
        if result.flagged_pages:
            flagged = ", ".join(str(page) for page in result.flagged_pages)
            typer.echo(
                f"warning: {result.name}: page(s) {flagged} are below "
                f"{min_text_chars} characters (likely image plates) — "
                f"check the cut before authoring a golden",
                err=True,
            )


if __name__ == "__main__":
    app()
