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

Each candidate is put through the *same* two-stage validation the persist path
uses: ``validate_hard`` first (a failure means production would persist nothing,
so the fixture is recorded ``hard_validation_failed`` and never scored), then
``validate_soft`` for the ready-vs-needs_review split (DECISIONS #4).

Judge integration (Phase 15.2, key extended in Epic 20 Phase 20.1): when
``judge`` names a committed judge prompt, the driver loads it via
``load_judge`` (reusing the *same* injected provider — never a second one),
replays cached ratings from the on-disk ``JudgeCache`` keyed by the eight-part
``JudgeCacheKey`` ``(fixture_set, fixture_id, fixture_content_hash,
extraction_prompt_version, artifact_hash, judge_name, judge_version, model)``
built at the single ``build_judge_cache_key`` site, calls the judge on misses,
and records per-fixture ratings plus an aggregate pass rate at
``results.judge``. A fixture edit, an extraction ``PROMPT_VERSION`` bump, or a
changed judged artifact each invalidate the cached rating; entries for
different artifacts of one fixture coexist (DECISIONS #10). A ``JudgeError``
(rejection, truncation, malformed verdict) marks the fixture *unrated* — never
a silent pass/fail — and the pass rate is computed over rated fixtures only.

``results.json`` schema (the Epic-14 envelope wraps the payload)::

    {
      "metadata": {...},            # ReportRun provenance
      "status": "completed",        # or "failed" for a crashed run
      "results": {
        "fixture_set": str,
        "extraction_prompt_version": str,   # eval-side copy of the extraction
                                    # PROMPT_VERSION; deliberately duplicates
                                    # metadata.prompt_version — build_metadata
                                    # lazily imports the rag_recipes constant, so
                                    # only this copy reflects the monkeypatch
                                    # seam the invalidation tests rely on
        "per_fixture": [            # one entry per fixture, keyed by name
          {"name", "status": "scored", "recipes_returned", "review_status",
           "warnings", "confidence_overall", "missing_fields", "scores": {...},
           "recipes": [...],        # full extracted payload (persisted list)
           "fixture_content_hash",  # RecipeFixture.content_hash() at run time
           "scored_recipe_index"},  # which item `scores` describe (always 0)
          {"name", "status": "extraction_failed", "error"},
          {"name", "status": "hard_validation_failed", "recipes_returned",
           "failures"},             # production would persist nothing
        ],
        "aggregate": {
          "fixtures", "recipes_extracted", "extraction_failures",
          "hard_validation_failures",
          "over_split_fixtures",      # fixtures the extractor split into >1 item
          "extraction_success_rate",  # scored / fixtures — regression-gated
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

Three distinct artifact slices, never to be conflated (DECISIONS #8):

- **scored item** — ``recipes[0]`` (``recipes[scored_recipe_index]``): the only
  item hard-validated, soft-validated, objectively scored, and whose
  ``confidence.overall`` feeds ``aggregate`` and calibration.
- **persisted list** — the full ``recipes`` payload in ``results.json``: every
  item the extractor returned, validated or not. ``scores`` describe the scored
  item only, never the whole list.
- **judged list** — the same full list as serialized for the judge; what the
  judge and the human both rate.

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
from evals.judge_cache import JudgeCache, JudgeCacheKey
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
    PROMPT_VERSION,
    ExtractedRecipe,
    RecipeExtractionOutput,
    run_extraction,
)
from rag_recipes.ingestion.pipeline.windows import Window
from rag_recipes.ingestion.validation import (
    SoftValidationThresholds,
    validate_hard,
    validate_soft,
)
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.storage.enums import ExtractionRunStatus
from rag_recipes.storage.models.source_span import SourceSpan

__all__ = [
    "artifact_hash",
    "build_judge_cache_key",
    "extraction_prompt_version",
    "run_extraction_eval",
    "serialize_extracted_artifact",
    "synthetic_span_id",
]

_TIME_FIELDS = ("prep_time", "cook_time", "total_time")

_DEFAULT_BASELINE_PATH = BASELINES_ROOT / "extraction.json"


def extraction_prompt_version() -> str:
    """The extraction ``PROMPT_VERSION`` this module keys judge caching on.

    Reads the module global at call time, so
    ``monkeypatch.setattr(evals.extraction, "PROMPT_VERSION", ...)`` is
    reflected — the seam the hermetic prompt-version-invalidation tests use.
    ``evals.alignment`` calls this accessor rather than importing the constant
    (DECISIONS #2): a second module-level binding would drift silently and make
    the monkeypatch cover only half the system.
    """
    return PROMPT_VERSION


def serialize_extracted_artifact(recipes: list[dict[str, Any]]) -> str:
    """Serialize the full extracted item list for judging: ``{"items": [...]}``.

    Literally ``RecipeExtractionOutput``'s wire shape. The one serialization
    the judge, the human (alignment), and ``artifact_hash`` all see — a second
    serialization site would make their byte-identity coincidental.
    """
    return json.dumps({"items": recipes}, indent=2)


def artifact_hash(text: str) -> str:
    """sha256 hex of the exact string handed to ``judge.judge``.

    The ``artifact_hash`` key part of :class:`JudgeCacheKey` (DECISIONS #10).
    Both the driver and alignment hash *the string they actually judge* through
    this one helper, so the key can never assert an artifact identity that was
    not judged.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_judge_cache_key(
    *,
    fixture_set: str,
    fixture_id: str,
    fixture_content_hash: str,
    extraction_prompt_version: str,
    artifact_hash: str,
    judge: Judge,
) -> JudgeCacheKey:
    """The single ``JudgeCacheKey`` construction site (DECISIONS #2).

    Every key part has exactly one source: the fixture-side provenance is
    passed in explicitly (the driver uses the working tree, alignment the run's
    recorded values), and the judge identity — including ``provider`` and
    ``model`` — comes from the ``Judge`` itself. ``evals.judge_cache`` stays
    storage-only.

    Because this is the sole construction site, ``evals.alignment`` inherited the
    Epic 23.3 ``provider`` part with no edit of its own.
    """
    return JudgeCacheKey(
        fixture_set=fixture_set,
        fixture_id=fixture_id,
        fixture_content_hash=fixture_content_hash,
        extraction_prompt_version=extraction_prompt_version,
        artifact_hash=artifact_hash,
        judge_name=judge.name,
        judge_version=judge.version,
        provider=judge.provider,
        model=judge.model,
    )


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


def _golden_structured(expected: dict[str, Any]) -> dict[str, Any]:
    """The golden ``structured_data`` block, tolerating an absent-or-null key.

    ``expected.json`` is an opaque dict (Epic-14 DECISIONS #3), so a fixture may
    legitimately write ``"structured_data": null`` or ``"ingredients": null``.
    ``dict.get(key, {})`` returns ``None`` for an explicit null, so every read
    goes through ``or`` defaults — a malformed golden file must score as a miss,
    never abort the whole run.
    """
    structured = expected.get("structured_data")
    return structured if isinstance(structured, dict) else {}


def _golden_list(structured: dict[str, Any], key: str) -> list[Any]:
    value = structured.get(key)
    return list(value) if isinstance(value, list) else []


def _missing_fields(recipe: ExtractedRecipe, expected: dict[str, Any]) -> list[str]:
    """Fields present in the golden recipe but absent from the extraction."""
    structured = recipe.structured_data
    expected_structured = _golden_structured(expected)
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

    def __init__(self, judge: Judge, cache: JudgeCache, fixture_set: str) -> None:
        self._judge = judge
        self._cache = cache
        self._fixture_set = fixture_set
        self.per_fixture: dict[str, dict[str, Any]] = {}
        self.passes = 0
        self.fails = 0
        self.unrated = 0

    async def rate(self, fixture: RecipeFixture, recipe_payloads: list[dict[str, Any]]) -> None:
        """Rate one scored fixture's *full* item list; ``JudgeError`` → unrated.

        ``recipe_payloads`` is the hoisted persisted list — never a second
        ``model_dump`` — serialized once as ``{"items": [...]}`` so
        ``boundary_correctness`` sees splits (the judged list; the *scored
        item* stays ``recipes[0]``, DECISIONS #8).
        """
        extracted = serialize_extracted_artifact(recipe_payloads)
        key = build_judge_cache_key(
            fixture_set=self._fixture_set,
            fixture_id=fixture.name,
            fixture_content_hash=fixture.content_hash(),
            # Module-global lookup at call time, so the monkeypatch seam works.
            extraction_prompt_version=PROMPT_VERSION,
            artifact_hash=artifact_hash(extracted),
            judge=self._judge,
        )
        rating = self._cache.get(key)
        if rating is None:
            try:
                rating = await self._judge.judge(
                    extracted,
                    json.dumps(fixture.expected, indent=2),
                    fixture.source_md,
                )
            except JudgeError as exc:
                self.unrated += 1
                self.per_fixture[fixture.name] = {"status": "unrated", "error": str(exc)}
                return
            # Stamped so the cache file is self-describing and alignment can
            # assert which artifact the rating belongs to (DECISIONS #10).
            rating.metadata["artifact_hash"] = key.artifact_hash
            self._cache.put(rating, key=key)
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
            # Provider as well as model: a committed baseline is read months
            # later by someone deciding whether to trust the number, and once
            # two providers can serve the same model name the model alone does
            # not say who graded it (Epic 23.3).
            "provider": self._judge.provider,
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
        f"Hard-validation failures: {aggregate['hard_validation_failures']}",
        f"Fixtures split across items: {aggregate['over_split_fixtures']}",
        f"Extraction success rate: {fmt(aggregate['extraction_success_rate'])}",
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
    expected_structured = _golden_structured(expected)

    expected_title = expected.get("title")
    title = score_title(recipe.title, expected_title if isinstance(expected_title, str) else "")
    yield_match = score_yield(structured.yield_, expected_structured.get("yield"))
    times = {
        field: score_times(getattr(structured, field), expected_structured.get(field))
        for field in _TIME_FIELDS
    }
    expected_ingredients = _golden_list(expected_structured, "ingredients")
    expected_steps = _golden_list(expected_structured, "steps")
    ingredient_count = score_ingredient_count(
        len(structured.ingredients), len(expected_ingredients)
    )
    step_count = score_step_count(len(structured.steps), len(expected_steps))
    ingredients_detail = score_ingredients_detail(
        [ingredient.model_dump() for ingredient in structured.ingredients],
        expected_ingredients,
    )
    spans = score_source_span_ids(recipe.source_span_ids, _golden_list(expected, "source_span_ids"))

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
    judge_provider: LLMProvider | None = None,
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
    prompt to run per fixture. ``judge_provider`` lets the judge run on a
    *different* provider than the model under test (Epic 23.3); ``None`` — the
    default — reuses ``llm_provider`` itself, the same object rather than an
    equivalent one, so the unset path is byte-identical to pre-23.3 behaviour.
    Setting it is what makes a cross-provider comparison meaningful: with one
    provider serving both roles, every candidate grades itself.
    ``fixtures_root``/``reports_root``/
    ``thresholds``/``settings``/``baseline_path``/``judge_cache_root`` default
    to the repo layout and real ``Settings`` but are injectable so tests stay
    hermetic. When a baseline exists the Epic-14 ``diff_against_baseline`` is
    invoked and its summary printed (placeholder content until Phase 15.3 fills
    the diff in).
    """
    judge_section: _JudgeSection | None = None
    if judge is not None:
        judge_section = _JudgeSection(
            load_judge(judge, judge_provider or llm_provider, root=fixtures_root),
            JudgeCache(root=judge_cache_root),
            fixture_set,
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
    if not fixtures:
        # The Epic-14 loader returns [] for an absent *or* empty set. Writing a
        # completed report anyway would produce zero fixtures, all-``None``
        # accuracy, and a baseline diff whose every metric is "missing" — i.e.
        # a mistyped --fixtures would print "No regressions detected".
        raise ValueError(
            f"recipe fixture set {fixture_set!r} is empty or does not exist; "
            f"nothing to evaluate"
        )

    with ReportRun(label, reports_root=reports_root, settings=settings) as run:
        per_fixture: list[dict[str, Any]] = []
        accuracy_values: defaultdict[str, list[float]] = defaultdict(list)
        missing_counts: Counter[str] = Counter()
        confidences: list[float] = []
        ready = 0
        needs_review = 0
        extraction_failures = 0
        hard_validation_failures = 0
        recipes_extracted = 0
        over_split = 0

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
            # Each synthetic fixture holds exactly one recipe (DECISIONS #1), so
            # the first item is the one scored — but the *count* is recorded so
            # an extractor that splits one recipe across items is visible in the
            # report rather than silently collapsing to "1 recipe extracted".
            recipes_extracted += len(recipes)
            if len(recipes) > 1:
                over_split += 1
            recipe = recipes[0]
            # Replay the *whole* persist-path classification, not half of it:
            # production runs validate_hard first and persists nothing when it
            # fires (persist.py raises HardValidationError). Scoring a candidate
            # it would have discarded — wrong item_type, blank title, a
            # hallucinated span id — would count it ready and hand it objective
            # scores for a recipe that never reaches the corpus.
            # The gate deliberately stays on recipes[0] — the *scored item* —
            # while the judge later sees the full *judged list* (DECISIONS #8):
            # widening the gate to every item would flip an over-split fixture
            # to hard_validation_failed and hide the very boundary error
            # boundary_correctness exists to judge.
            hard_failures = validate_hard(recipe, window)
            if hard_failures:
                hard_validation_failures += 1
                per_fixture.append(
                    {
                        "name": fixture.name,
                        "status": "hard_validation_failed",
                        "recipes_returned": len(recipes),
                        "failures": [failure.code for failure in hard_failures],
                    }
                )
                if judge_section is not None:
                    judge_section.skip(fixture.name, "hard validation failed")
                continue
            # One serialization site for the persisted list and the judged list
            # (TASK-003): a second model_dump would make the byte-identity
            # guarantee between them true only by coincidence.
            recipe_payloads = [
                item.model_dump(mode="json", by_alias=True) for item in recipes
            ]
            if judge_section is not None:
                await judge_section.rate(fixture, recipe_payloads)
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
                    "recipes_returned": len(recipes),
                    "review_status": "needs_review" if warnings else "ready",
                    "warnings": [warning.code for warning in warnings],
                    "confidence_overall": recipe.confidence.overall,
                    "missing_fields": missing,
                    "scores": _score_fixture(recipe, fixture.expected, accuracy_values),
                    "recipes": recipe_payloads,
                    "fixture_content_hash": fixture.content_hash(),
                    "scored_recipe_index": 0,
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
            "recipes_extracted": recipes_extracted,
            "extraction_failures": extraction_failures,
            "hard_validation_failures": hard_validation_failures,
            "over_split_fixtures": over_split,
            "ready": ready,
            "needs_review": needs_review,
            # Coverage is a *rate*, so it is regression-gated where the raw
            # counts are only context (DECISIONS #5): per-field accuracy is a
            # mean over the fixtures that produced a score, so 99 failures and
            # one perfect survivor would otherwise read as a clean run.
            "extraction_success_rate": (ready + needs_review) / len(fixtures),
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
                # Deliberately duplicates metadata.prompt_version: build_metadata
                # lazily imports PROMPT_VERSION from rag_recipes at call time, so
                # the metadata copy is NOT reachable by monkeypatch.setattr(
                # evals.extraction, "PROMPT_VERSION", ...). This eval-side copy is
                # the seam that makes prompt-version cache invalidation testable
                # hermetically (DECISIONS #2) — do not "simplify" it away.
                "extraction_prompt_version": PROMPT_VERSION,
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
