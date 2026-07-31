"""Extraction evaluation driver (Epic 15 Phase 15.1, doc 12 §§ 5/10).

Drives the *real* LLM extraction path (``run_extraction`` — real prompt, real
``recipe.v1`` schema, real parse/validate logic) over each recipe fixture of a
set, scores the first extracted recipe against the fixture's golden
``expected.json`` with the pure scorers in ``evals.scoring.objective``, and
writes ``results.json`` + ``summary.md`` through the Epic-14 ``ReportRun``.

Synthetic-window path (DECISIONS #1): each fixture's ``source.md`` becomes one
detached in-memory ``SourceSpan`` (id ``synthetic_span_id(fixture.name)``)
wrapped in a single ``Window`` — no PDF, no DB row. By design every synthetic
fixture holds a single recipe, so the driver scores the first extracted item.

Judge integration (Phase 15.2): when ``judge`` names a committed judge prompt,
the driver loads it via ``load_judge`` (reusing the *same* injected provider —
never a second one), replays cached ratings from the on-disk ``JudgeCache``
keyed by ``(fixture_id, judge_name, judge_version, model)``, calls the judge on
misses, and records per-fixture ratings plus an aggregate pass rate at
``results.judge``. A ``JudgeError`` (rejection, truncation, malformed verdict)
marks the fixture *unrated* — never a silent pass/fail — and the pass rate is
computed over rated fixtures only.

``results.json`` schema (the Epic-14 envelope wraps the payload)::

    {
      "metadata": {...},            # ReportRun provenance
      "status": "completed",        # or "failed" for a crashed run
      "results": {
        "fixture_set": str,
        "per_fixture": [            # one entry per fixture, keyed by name
          {"name", "status": "scored", "review_status", "warnings",
           "confidence_overall", "missing_fields", "scores": {...}},
          {"name", "status": "extraction_failed", "error"},
        ],
        "aggregate": {
          "fixtures", "recipes_extracted", "extraction_failures",
          "ready", "needs_review", "average_confidence",
          "field_accuracy": {...},  # per-field means (None when ineligible)
          "missing_field_counts": {...},
        },
        "judge": null | {           # null without --judge
          "name", "version", "model",
          "per_fixture": {<fixture name>: {"status": "rated", ...JudgeRating}
                          | {"status": "unrated", "error"}},
          "pass_rate",              # over rated fixtures; null when none rated
          "rated", "unrated", "passes", "fails"
        },
        "agreement": null,          # reserved — Phase 15.3 fills
        "calibration": null         # reserved — Phase 15.3 fills
      }
    }

The driver never constructs an LLM provider — ``llm_provider`` is a required
injected dependency (tests use ``FakeLLMProvider``); the only live-provider
construction site is the ``rag-evals`` CLI.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from evals.fixtures import load_recipe_fixtures
from evals.judge_cache import JudgeCache
from evals.judges import Judge, JudgeError, load_judge
from evals.models import RecipeFixture
from evals.reports import BASELINES_ROOT, ReportRun, SettingsLike, diff_against_baseline
from evals.scoring.objective import (
    score_ingredient_count,
    score_ingredients_detail,
    score_source_span_ids,
    score_step_count,
    score_times,
    score_title,
    score_yield,
)
from rag_recipes.ingestion.pipeline.extraction import (
    ExtractedRecipe,
    RecipeExtractionOutput,
    run_extraction,
)
from rag_recipes.ingestion.pipeline.windows import Window
from rag_recipes.ingestion.validation import SoftValidationThresholds, validate_soft
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.storage.enums import ExtractionRunStatus
from rag_recipes.storage.models.source_span import SourceSpan

__all__ = ["run_extraction_eval", "synthetic_span_id"]

_TIME_FIELDS = ("prep_time", "cook_time", "total_time")

_DEFAULT_BASELINE_PATH = BASELINES_ROOT / "extraction.json"


def synthetic_span_id(fixture_name: str) -> str:
    """Deterministic span id for a fixture's synthetic window.

    Fixture authors reference this id in ``expected.json`` (``source_span_ids``)
    so span-provenance scoring lines up with what a live model would echo back.
    """
    return f"span_eval_{fixture_name}"


class _NoOpSession:
    """Minimal ``AsyncSession`` stand-in for ``run_extraction``'s three calls.

    ``run_extraction`` only uses ``scalar()`` (cache lookup — ``None`` here, so
    the provider is always called), ``add()``, and ``flush()``. A real session
    would require a committed ``Document`` row (``ExtractionRun.document_id``
    FK), which the offline eval deliberately avoids (PLAN Risks).
    """

    def add(self, obj: object) -> None:
        return None

    async def flush(self) -> None:
        return None

    async def scalar(self, *args: object, **kwargs: object) -> None:
        return None


def _build_synthetic_window(fixture_name: str, source_md: str) -> Window:
    """Wrap a fixture's ``source.md`` in one detached single-page window.

    Only the load-bearing ``SourceSpan`` fields are set — ``id``/``text`` (read
    by ``format_window_for_llm``), ``text_hash`` (read by ``compute_input_hash``)
    and ``locator`` page bounds. The span is never ``session.add``-ed, so the
    model's non-nullable columns and ``@validates`` hooks never engage.
    """
    span = SourceSpan(
        id=synthetic_span_id(fixture_name),
        text=source_md,
        text_hash=hashlib.sha256(source_md.encode("utf-8")).hexdigest(),
        locator={"page_start": 1, "page_end": 1},
    )
    return Window(spans=(span,))


async def _extract_recipes(
    window: Window, fixture_name: str, provider: LLMProvider
) -> tuple[list[ExtractedRecipe], str | None]:
    """Run one extraction over ``window``; ``([], error)`` when nothing usable.

    A ``REJECTED`` run (``output_json is None`` — parse failure, refusal, or an
    Anthropic ``max_tokens`` truncation) is reported as an error string, never
    raised. ``LLMTechnicalError`` still propagates (transport failure aborts
    the eval run, and ``ReportRun.__exit__`` finalizes it as ``failed``).
    """
    session = cast(AsyncSession, _NoOpSession())
    run = await run_extraction(
        session,
        window,
        source_version=1,
        document_id=f"doc_eval_{fixture_name}",
        provider=provider,
    )
    if run.status is not ExtractionRunStatus.SUCCESS or run.output_json is None:
        return [], run.error_message or "extraction did not succeed"
    items = RecipeExtractionOutput.model_validate(run.output_json).items
    if not items:
        return [], "extraction returned no recipes"
    return items, None


def _missing_fields(recipe: ExtractedRecipe, expected: dict[str, Any]) -> list[str]:
    """Fields present in the golden recipe but absent from the extraction."""
    structured = recipe.structured_data
    expected_structured = expected.get("structured_data", {})
    missing: list[str] = []
    if expected.get("title") and not recipe.title.strip():
        missing.append("title")
    if expected_structured.get("yield") is not None and structured.yield_ is None:
        missing.append("yield")
    for field in _TIME_FIELDS:
        if expected_structured.get(field) is not None and getattr(structured, field) is None:
            missing.append(field)
    if expected_structured.get("ingredients") and not structured.ingredients:
        missing.append("ingredients")
    if expected_structured.get("steps") and not structured.steps:
        missing.append("steps")
    return missing


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


class _JudgeSection:
    """Accumulates per-fixture judge outcomes for the ``results.judge`` slot."""

    def __init__(self, judge: Judge, cache: JudgeCache) -> None:
        self._judge = judge
        self._cache = cache
        self.per_fixture: dict[str, dict[str, Any]] = {}
        self.passes = 0
        self.fails = 0
        self.unrated = 0

    async def rate(self, fixture: RecipeFixture, recipe: ExtractedRecipe) -> None:
        """Rate one scored fixture, replaying the cache; ``JudgeError`` → unrated."""
        rating = self._cache.get(
            fixture.name, self._judge.name, self._judge.version, self._judge.model
        )
        if rating is None:
            try:
                rating = await self._judge.judge(
                    json.dumps(recipe.model_dump(mode="json", by_alias=True), indent=2),
                    json.dumps(fixture.expected, indent=2),
                    fixture.source_md,
                )
            except JudgeError as exc:
                self.unrated += 1
                self.per_fixture[fixture.name] = {"status": "unrated", "error": str(exc)}
                return
            self._cache.put(rating, fixture_id=fixture.name)
        if rating.rating == "pass":
            self.passes += 1
        else:
            self.fails += 1
        self.per_fixture[fixture.name] = {"status": "rated", **rating.model_dump()}

    def skip(self, fixture_name: str, reason: str) -> None:
        """Record a fixture the judge never saw (e.g. its extraction failed)."""
        self.unrated += 1
        self.per_fixture[fixture_name] = {"status": "unrated", "error": reason}

    @property
    def pass_rate(self) -> float | None:
        """Pass rate over *rated* fixtures only — unrated never counts as fail."""
        rated = self.passes + self.fails
        return self.passes / rated if rated else None

    def payload(self) -> dict[str, Any]:
        return {
            "name": self._judge.name,
            "version": self._judge.version,
            "model": self._judge.model,
            "per_fixture": self.per_fixture,
            "pass_rate": self.pass_rate,
            "rated": self.passes + self.fails,
            "unrated": self.unrated,
            "passes": self.passes,
            "fails": self.fails,
        }


def _render_summary(
    label: str,
    fixture_set: str,
    aggregate: dict[str, Any],
    judge_payload: dict[str, Any] | None = None,
) -> str:
    """Render the doc-12 § 10 extraction-report block for a multi-fixture run.

    doc 12 § 10's ``Document:`` / ``Source version:`` header lines are
    single-document framing; a fixture-set run renders the set name + fixture
    count in their place. The judge pass-rate line appears only when a judge
    ran; the judge-human agreement line is Phase 15.3's output and absent here.
    """

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    accuracy = aggregate["field_accuracy"]
    missing = aggregate["missing_field_counts"]
    lines = [
        f"# Extraction eval — {label}",
        "",
        f"Fixture set: {fixture_set} ({aggregate['fixtures']} fixtures)",
        f"Recipes extracted: {aggregate['recipes_extracted']}",
        f"Extraction failures: {aggregate['extraction_failures']}",
        f"Ready: {aggregate['ready']}",
        f"Needs review: {aggregate['needs_review']}",
        f"Average confidence: {fmt(aggregate['average_confidence'])}",
        "Objective field accuracy:",
        f"  title: {fmt(accuracy['title_normalized'])}",
        f"  yield: {fmt(accuracy['yield'])}",
        f"  ingredients (count match): {fmt(accuracy['ingredient_count'])}",
        f"  ingredients (per-field): {fmt(accuracy['ingredients_detail_f1'])}",
        f"  steps (count match): {fmt(accuracy['step_count'])}",
    ]
    if judge_payload is not None:
        lines.append(
            f"Judge pass rate ({judge_payload['name']}, {judge_payload['version']}): "
            f"{fmt(judge_payload['pass_rate'])}"
        )
    lines += [
        f"Missing ingredients: {missing.get('ingredients', 0)}",
        f"Missing steps: {missing.get('steps', 0)}",
    ]
    return "\n".join(lines) + "\n"


def _score_fixture(
    recipe: ExtractedRecipe,
    expected: dict[str, Any],
    accuracy_values: defaultdict[str, list[float]],
) -> dict[str, Any]:
    """Score one extracted recipe against its golden values.

    Returns the serialisable per-field ``scores`` dict and feeds the aggregate
    accuracy accumulators (time/yield fields only count toward accuracy when the
    golden side carries a comparable value — their absence is a fixture-design
    fact, not an extraction miss).
    """
    structured = recipe.structured_data
    expected_structured = expected.get("structured_data", {})

    title = score_title(recipe.title, expected.get("title", ""))
    yield_match = score_yield(structured.yield_, expected_structured.get("yield"))
    times = {
        field: score_times(getattr(structured, field), expected_structured.get(field))
        for field in _TIME_FIELDS
    }
    expected_ingredients = list(expected_structured.get("ingredients", []))
    expected_steps = list(expected_structured.get("steps", []))
    ingredient_count = score_ingredient_count(
        len(structured.ingredients), len(expected_ingredients)
    )
    step_count = score_step_count(len(structured.steps), len(expected_steps))
    ingredients_detail = score_ingredients_detail(
        [ingredient.model_dump() for ingredient in structured.ingredients],
        expected_ingredients,
    )
    spans = score_source_span_ids(
        recipe.source_span_ids, list(expected.get("source_span_ids", []))
    )

    accuracy_values["title_exact"].append(float(title.exact))
    accuracy_values["title_normalized"].append(float(title.normalized))
    if expected_structured.get("yield") is not None:
        accuracy_values["yield"].append(float(yield_match))
    for field, time_score in times.items():
        if time_score.expected_minutes is not None:
            accuracy_values[field].append(float(time_score.match))
    accuracy_values["ingredient_count"].append(float(ingredient_count))
    accuracy_values["step_count"].append(float(step_count))
    accuracy_values["ingredients_detail_f1"].append(ingredients_detail.f1)
    accuracy_values["source_span_ids_f1"].append(spans.f1)

    return {
        "title": asdict(title),
        "yield": yield_match,
        **{field: asdict(time_score) for field, time_score in times.items()},
        "ingredient_count": ingredient_count,
        "step_count": step_count,
        "ingredients_detail": asdict(ingredients_detail),
        "source_span_ids": asdict(spans),
    }


async def run_extraction_eval(
    fixture_set: str,
    label: str,
    *,
    llm_provider: LLMProvider,
    judge: str | None = None,
    reports_root: Path | None = None,
    fixtures_root: Path | None = None,
    thresholds: SoftValidationThresholds | None = None,
    settings: SettingsLike | None = None,
    baseline_path: Path | None = None,
    judge_cache_root: Path | None = None,
) -> ReportRun:
    """Evaluate extraction quality over a recipe fixture set; returns the run.

    ``llm_provider`` is required and injected — this function never builds a
    provider and never reads an API key. ``judge`` names a committed judge
    prompt to run per fixture (the same injected provider serves both
    extraction and judge calls). ``fixtures_root``/``reports_root``/
    ``thresholds``/``settings``/``baseline_path``/``judge_cache_root`` default
    to the repo layout and real ``Settings`` but are injectable so tests stay
    hermetic. When a baseline exists the Epic-14 ``diff_against_baseline`` is
    invoked and its summary printed (placeholder content until Phase 15.3 fills
    the diff in).
    """
    judge_section: _JudgeSection | None = None
    if judge is not None:
        judge_section = _JudgeSection(
            load_judge(judge, llm_provider, root=fixtures_root),
            JudgeCache(root=judge_cache_root),
        )
    if thresholds is None or settings is None:
        from rag_recipes.config import get_settings

        real_settings = get_settings()
        if settings is None:
            settings = real_settings
        if thresholds is None:
            thresholds = SoftValidationThresholds(
                min_overall_confidence=real_settings.extraction_min_overall_confidence,
                min_boundary_confidence=real_settings.extraction_min_boundary_confidence,
                min_normalization_confidence=(
                    real_settings.extraction_min_normalization_confidence
                ),
                min_recipe_chars=real_settings.extraction_min_recipe_chars,
                max_recipe_chars=real_settings.extraction_max_recipe_chars,
            )

    fixtures = load_recipe_fixtures(fixture_set, root=fixtures_root)

    with ReportRun(label, reports_root=reports_root, settings=settings) as run:
        per_fixture: list[dict[str, Any]] = []
        accuracy_values: defaultdict[str, list[float]] = defaultdict(list)
        missing_counts: Counter[str] = Counter()
        confidences: list[float] = []
        ready = 0
        needs_review = 0
        extraction_failures = 0

        for fixture in fixtures:
            window = _build_synthetic_window(fixture.name, fixture.source_md)
            recipes, error = await _extract_recipes(window, fixture.name, llm_provider)
            if not recipes:
                extraction_failures += 1
                per_fixture.append(
                    {"name": fixture.name, "status": "extraction_failed", "error": error}
                )
                if judge_section is not None:
                    judge_section.skip(fixture.name, "extraction failed")
                continue
            recipe = recipes[0]
            if judge_section is not None:
                await judge_section.rate(fixture, recipe)
            warnings = validate_soft(recipe, thresholds=thresholds)
            if warnings:
                needs_review += 1
            else:
                ready += 1
            missing = _missing_fields(recipe, fixture.expected)
            missing_counts.update(missing)
            confidences.append(recipe.confidence.overall)
            per_fixture.append(
                {
                    "name": fixture.name,
                    "status": "scored",
                    "review_status": "needs_review" if warnings else "ready",
                    "warnings": [warning.code for warning in warnings],
                    "confidence_overall": recipe.confidence.overall,
                    "missing_fields": missing,
                    "scores": _score_fixture(recipe, fixture.expected, accuracy_values),
                }
            )

        accuracy_fields = (
            "title_exact",
            "title_normalized",
            "yield",
            *_TIME_FIELDS,
            "ingredient_count",
            "step_count",
            "ingredients_detail_f1",
            "source_span_ids_f1",
        )
        aggregate = {
            "fixtures": len(fixtures),
            "recipes_extracted": len(fixtures) - extraction_failures,
            "extraction_failures": extraction_failures,
            "ready": ready,
            "needs_review": needs_review,
            "average_confidence": _mean(confidences),
            "field_accuracy": {
                field: _mean(accuracy_values[field]) for field in accuracy_fields
            },
            "missing_field_counts": dict(missing_counts),
        }
        judge_payload = judge_section.payload() if judge_section is not None else None
        run.write_results(
            {
                "fixture_set": fixture_set,
                "per_fixture": per_fixture,
                "aggregate": aggregate,
                "judge": judge_payload,
                "agreement": None,
                "calibration": None,
            }
        )
        run.write_summary(_render_summary(label, fixture_set, aggregate, judge_payload))

    resolved_baseline = _DEFAULT_BASELINE_PATH if baseline_path is None else baseline_path
    if resolved_baseline.is_file():
        diff = diff_against_baseline(resolved_baseline, run.path)
        print(diff.summary)

    return run
