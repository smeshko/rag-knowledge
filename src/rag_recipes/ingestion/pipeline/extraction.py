"""LLM recipe-extraction layer (doc 4 § Strict Output Shape).

Defines the ``recipe.v1`` Pydantic output contract and the strict-mode JSON
schema fed to the LLM provider. The home of these models is this module (per the
Epic-9 phase docs — there is no ``domain/`` model for the extraction output).

**Validation boundary (DECISIONS #1):** the Pydantic layer enforces *structure
and typing only* — field presence, types, and nesting. It deliberately does NOT
enforce the ``[0, 1]`` confidence range or ``item_type == "recipe"``; those are
HARD validation checks owned by Phase 9.3 and must not flip an ``ExtractionRun``
to ``rejected`` here.
"""

from __future__ import annotations

import importlib.resources
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.windows import (
    Window,
    compute_input_hash,
    format_window_for_llm,
)
from rag_recipes.providers._observability import ProviderObservability, TraceContext
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest, StructuredOutputResponse
from rag_recipes.storage.enums import ExtractionRunStatus
from rag_recipes.storage.models.extraction_run import ExtractionRun

__all__ = [
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "ExtractedIngredient",
    "ExtractedRecipe",
    "ExtractedStep",
    "IngredientConfidence",
    "RecipeConfidence",
    "RecipeExtractionOutput",
    "RecipeFieldConfidence",
    "RecipeV1StructuredData",
    "WindowExtraction",
    "build_recipe_v1_json_schema",
    "call_provider_for_window",
    "find_cached_extraction",
    "record_window_extraction",
    "run_extraction",
]

logger = logging.getLogger(__name__)

# LLM API contract versions — module constants, NOT ``Settings`` (DECISIONS #3).
# They feed ``compute_input_hash`` and every ``ExtractionRun`` row, so they must
# travel atomically with the prompt template / schema code they describe. A
# meaningful change to either the prompt or the schema shape must bump these.
PROMPT_VERSION = "recipe-extraction-v2"
# Deliberately NOT bumped alongside PROMPT_VERSION for the TOKEN BUDGET trim.
# This constant is the *payload* contract consumers read — it is the value
# ``structured_data.schema`` carries, `evals.golden_schema` asserts, and the API
# projections default to. That contract did not change: the trimmed fields were
# derived (``ingredients_text``/``steps_text``) or unread (the four ingredient
# confidences, both step confidences), and none appear in the golden key sets
# (``STRUCTURED_KEYS``, ``STEP_KEYS``, ``INGREDIENT_SUB_FIELDS``). Bumping it
# would rename the discriminator out from under 42 committed goldens for no
# consumer-visible change. Cache invalidation is already total: PROMPT_VERSION
# is part of ``compute_input_hash`` and of the ``_find_cached_run`` key, so no
# pre-trim run can satisfy a post-trim lookup.
SCHEMA_VERSION = "recipe.v1"

_PROMPT_PLACEHOLDER = "{source_spans}"
_PROMPT_PACKAGE = "rag_recipes.ingestion.prompts"
_PROMPT_RESOURCE = "recipe_extraction_v2.md"


def _load_prompt_template() -> str:
    """Load the versioned prompt template shipped as package data.

    Loaded via ``importlib.resources`` so it resolves from the installed wheel
    as well as the source tree. Read once and cached at module scope.
    """
    return (
        importlib.resources.files(_PROMPT_PACKAGE)
        .joinpath(_PROMPT_RESOURCE)
        .read_text(encoding="utf-8")
    )


_PROMPT_TEMPLATE = _load_prompt_template()


def _render_prompt(window_text: str) -> str:
    """Inject the formatted page window into the prompt template.

    Uses ``str.replace`` (not ``str.format``) because the windowed source text
    may contain literal ``{``/``}`` that would break ``str.format``. The rendered
    string is part of the prompt contract; any observable change to the template
    must bump ``PROMPT_VERSION`` (see the template header).
    """
    return _PROMPT_TEMPLATE.replace(_PROMPT_PLACEHOLDER, window_text)


# TOKEN BUDGET — rationale lives in a comment, NOT a docstring: every class
# docstring in this module is emitted as a ``description`` in
# ``build_recipe_v1_json_schema`` and shipped to the model on every window. A
# verbose docstring here is a per-call input-token cost forever.
#
# ``IngredientConfidence`` carried overall/quantity/unit/item alongside
# normalization. Measured on a real 226-page cookbook ingest, the five floats
# were 9.5% of all output tokens — and output volume is what sets both latency
# and quota spend (a recipe-bearing window averaged 151s / ~2,700 output
# tokens). Only ``normalization`` has a reader: ``validate_soft`` takes the
# minimum across the list and warns below
# ``extraction_min_normalization_confidence``. The other four reached exactly
# one consumer — ``_confidence_pairs``, which range-checked them in [0, 1] and
# discarded them. The model was paying to grade itself on axes nothing read.
class IngredientConfidence(BaseModel):
    """How faithfully the normalized fields represent the raw line."""

    normalization: float


class ExtractedIngredient(BaseModel):
    """One parsed ingredient line (doc 4 § Ingredient Field Meaning)."""

    position: int
    raw_text: str
    quantity_text: str | None
    quantity_value: float | None
    unit_raw: str | None
    unit_normalized: str | None
    item_text: str | None
    item_normalized: str | None
    preparation: str | None
    notes: str | None
    confidence: IngredientConfidence


# TOKEN BUDGET: dropped ``confidence`` (overall + ordering). Unlike the
# ingredient case, neither float had ANY reader beyond ``_confidence_pairs``'
# range check — not validation, not ``compute_candidate_score``, not the review
# API. (Comment, not docstring: see the note above ``IngredientConfidence``.)
class ExtractedStep(BaseModel):
    """One method step (doc 4 § Strict Output Shape)."""

    step_number: int
    text: str
    source_span_ids: list[str]


class RecipeV1StructuredData(BaseModel):
    """The ``recipe.v1`` structured payload nested in each extracted recipe.

    ``schema`` is a defaulted-but-required discriminator (DECISIONS #4): a plain
    ``str`` so a wrong value is a 9.3 semantic concern, not a 9.2 structural
    rejection. It is carried by the Python field ``schema_`` with
    ``alias="schema"`` because the bare name ``schema`` shadows
    ``BaseModel.schema`` — the same keyword/attribute-collision idiom that forces
    ``yield`` to ``yield_``. ``populate_by_name=True`` lets the model be built by
    field name (tests) and by alias (the LLM payload / doc-4 example).
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(alias="schema", default="recipe.v1")
    yield_: str | None = Field(alias="yield")
    prep_time: str | None
    cook_time: str | None
    total_time: str | None
    # No ``ingredients_text`` / ``steps_text`` (TOKEN BUDGET): the model used to
    # emit both — the ingredient lines glued with newlines, then the step texts
    # glued with newlines — which is 12.5% of output tokens restating text it had
    # already written per row. ``composition.resolve_ingredients_text`` /
    # ``resolve_steps_text`` derive them from ``raw_text`` / ``text`` by exactly
    # that rule, so the bytes were bought and then regenerated locally anyway.
    #
    # They remain valid *stored* keys: ``editing`` writes them back when a
    # reviewer changes a line (mirroring the same join), and reads everywhere go
    # through ``composition`` or ``dict.get``. Pydantic ignores them as extras,
    # so pre-trim ``output_json`` still validates.
    ingredients: list[ExtractedIngredient]
    steps: list[ExtractedStep]


class RecipeFieldConfidence(BaseModel):
    """Per-field confidence for a recipe (doc 4 § Confidence Scores)."""

    model_config = ConfigDict(populate_by_name=True)

    title: float
    summary: float
    yield_: float = Field(alias="yield")
    ingredients: float
    steps: float


class RecipeConfidence(BaseModel):
    """Top-level confidence for an extracted recipe (doc 4 § Confidence Scores)."""

    overall: float
    boundary: float
    fields: RecipeFieldConfidence


class ExtractedRecipe(BaseModel):
    """One extracted recipe candidate (doc 4 § Strict Output Shape)."""

    item_type: str
    title: str
    summary: str | None
    body_text: str
    source_span_ids: list[str]
    structured_data: RecipeV1StructuredData
    confidence: RecipeConfidence
    warnings: list[str]


class RecipeExtractionOutput(BaseModel):
    """Top-level LLM output: the list of extracted recipe candidates."""

    items: list[ExtractedRecipe]


def _strictify(node: Any) -> Any:
    """Rewrite a JSON-schema tree in place into OpenAI strict-mode shape.

    For every object node, force ``additionalProperties: false`` and list every
    property in ``required`` (OpenAI strict mode requires ALL keys present;
    nullable fields stay nullable via the ``["T", "null"]`` / ``anyOf`` union
    Pydantic already emits — never by omission from ``required``). Recurses every
    nested mapping/sequence so ``$defs``, ``properties``, ``items``, and
    ``anyOf`` are all covered.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties", {})
            node["additionalProperties"] = False
            node["required"] = list(properties.keys())
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)
    return node


def build_recipe_v1_json_schema() -> dict[str, Any]:
    """Return the OpenAI strict-mode JSON schema for ``RecipeExtractionOutput``.

    Derived from the single source of truth (``model_json_schema(by_alias=True)``,
    so ``yield`` appears under its alias) and post-processed by ``_strictify`` to
    satisfy OpenAI strict mode. See DECISIONS #5.
    """
    schema = RecipeExtractionOutput.model_json_schema(by_alias=True)
    _strictify(schema)
    return schema


def _finalize(
    run: ExtractionRun,
    *,
    status: ExtractionRunStatus,
    output_json: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Write a terminal status + audit fields onto a ``RUNNING`` run.

    ``output_json`` is whole-object-assigned (always dirty-tracked on plain
    JSONB); never mutate it in place (a 9.3 concern).
    """
    run.status = status
    run.output_json = output_json
    run.error_message = error
    run.completed_at = datetime.now(tz=UTC)


async def _find_cached_run(
    session: AsyncSession,
    *,
    input_hash: str,
    provider: str,
    model: str,
) -> ExtractionRun | None:
    """Return the most recent prior ``SUCCESS`` run for this cache key, or None.

    The cache key is ``(input_hash, provider, model, prompt_version,
    schema_version)`` — ``input_hash`` is ``compute_input_hash`` (over span
    identity + versions), NOT ``FakeLLMProvider.request_hash``. Ordered by
    ``created_at`` descending so the most recent prior success wins.
    """
    stmt = (
        select(ExtractionRun)
        .where(
            ExtractionRun.status == ExtractionRunStatus.SUCCESS,
            ExtractionRun.input_hash == input_hash,
            ExtractionRun.provider == provider,
            ExtractionRun.model == model,
            ExtractionRun.prompt_version == PROMPT_VERSION,
            ExtractionRun.schema_version == SCHEMA_VERSION,
        )
        .order_by(ExtractionRun.created_at.desc())
        .limit(1)
    )
    result: ExtractionRun | None = await session.scalar(stmt)
    return result


@dataclass(frozen=True)
class WindowExtraction:
    """One window's provider outcome, captured *before* any database write.

    The seam that makes concurrent extraction possible. An ``AsyncSession`` is
    not safe for concurrent use, so windows cannot simply be gathered over
    ``run_extraction`` — its flushes would interleave on one connection. Holding
    the provider result in a plain object lets the slow part (the LLM call) fan
    out while every write stays sequential on the caller's session.

    ``error`` is a captured ``LLMTechnicalError`` rather than a raised one:
    ``asyncio.gather`` would otherwise lose the other windows' results to the
    first failure. ``record_window_extraction`` writes the ``FAILED`` audit row
    and, by default, re-raises it — ``run_extraction``'s contract. The batch loop
    passes ``raise_on_error=False`` so every sibling's row (SUCCESS included) is
    recorded and committed first, and the failure is re-raised only after the
    batch commit; otherwise the paid-for sibling results would roll back with
    the transaction and never reach the extraction cache.
    """

    window: Window
    input_hash: str
    response: StructuredOutputResponse | None
    error: LLMTechnicalError | None


async def call_provider_for_window(
    window: Window,
    *,
    provider: LLMProvider,
    document_id: str,
    observability: ProviderObservability | None = None,
) -> WindowExtraction:
    """Call the provider for ``window``. No session, no writes — safe to gather.

    Deliberately takes no ``AsyncSession``: that is the whole point of the split.
    """
    input_hash = compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION)
    request = StructuredOutputRequest(
        provider=provider.provider,
        model=provider.default_model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input=_render_prompt(format_window_for_llm(window)),
        json_schema=build_recipe_v1_json_schema(),
    )
    trace_context = TraceContext(
        session_id=document_id,
        input_hash=input_hash,
        input_source_span_ids=window.span_ids,
    )
    try:
        response = await provider.generate_structured_output(request, trace_context=trace_context)
    except LLMTechnicalError as exc:
        return WindowExtraction(window=window, input_hash=input_hash, response=None, error=exc)
    return WindowExtraction(window=window, input_hash=input_hash, response=response, error=None)


async def record_window_extraction(
    session: AsyncSession,
    outcome: WindowExtraction,
    *,
    source_version: int,
    document_id: str,
    provider: LLMProvider,
    raise_on_error: bool = True,
) -> ExtractionRun:
    """Write ``outcome``'s ``ExtractionRun`` and resolve it to a terminal status.

    The write half of the split — must run sequentially on the caller's session.
    Statuses and the re-raise-after-recording behaviour match ``run_extraction``
    exactly, because ``run_extraction`` is now implemented in terms of this.

    ``raise_on_error=False`` returns the ``FAILED`` run instead of raising, for a
    caller that must finish recording a whole batch before it aborts; the caller
    then owns re-raising ``outcome.error``.
    """
    run = ExtractionRun(
        document_id=document_id,
        source_version=source_version,
        provider=provider.provider,
        model=provider.default_model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=outcome.window.span_ids,
        input_hash=outcome.input_hash,
        status=ExtractionRunStatus.RUNNING,
        output_json=None,
    )
    session.add(run)
    await session.flush()

    if outcome.error is not None:
        _finalize(run, status=ExtractionRunStatus.FAILED, error=str(outcome.error))
        await session.flush()
        if raise_on_error:
            raise outcome.error
        return run

    response = outcome.response
    assert response is not None  # noqa: S101 - error is None, so response is set
    if response.output_json is None:
        _finalize(run, status=ExtractionRunStatus.REJECTED, error=response.parse_error)
        await session.flush()
        return run

    try:
        RecipeExtractionOutput.model_validate(response.output_json)
    except ValidationError as exc:
        _finalize(
            run,
            status=ExtractionRunStatus.REJECTED,
            output_json=response.output_json,
            error=str(exc),
        )
        await session.flush()
        return run

    _finalize(run, status=ExtractionRunStatus.SUCCESS, output_json=response.output_json)
    await session.flush()
    return run


async def find_cached_extraction(
    session: AsyncSession,
    window: Window,
    *,
    source_version: int,
    document_id: str,
    provider: LLMProvider,
) -> ExtractionRun | None:
    """Record and return a cache-hit run for ``window``, or ``None`` on a miss.

    Public so the concurrent path can run every cache check *before* dispatching
    provider calls — a hit must never cost an LLM invocation just because the
    lookup moved off the sequential path.
    """
    input_hash = compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION)
    cached = await _find_cached_run(
        session,
        input_hash=input_hash,
        provider=provider.provider,
        model=provider.default_model,
    )
    if cached is None:
        return None
    logger.debug(
        "extraction cache hit: reusing run %s for input_hash=%s provider=%s model=%s",
        cached.id,
        input_hash,
        provider.provider,
        provider.default_model,
    )
    cached_run = ExtractionRun(
        document_id=document_id,
        source_version=source_version,
        provider=provider.provider,
        model=provider.default_model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=window.span_ids,
        input_hash=input_hash,
        status=ExtractionRunStatus.SUCCESS,
        output_json=cached.output_json,
        completed_at=datetime.now(tz=UTC),
    )
    session.add(cached_run)
    await session.flush()
    return cached_run


async def run_extraction(
    session: AsyncSession,
    window: Window,
    *,
    source_version: int,
    document_id: str,
    provider: LLMProvider,
    observability: ProviderObservability | None = None,
) -> ExtractionRun:
    """Run one LLM extraction call for ``window`` and record an ``ExtractionRun``.

    The sequential composition of ``find_cached_extraction`` →
    ``call_provider_for_window`` → ``record_window_extraction``. Kept as the
    single-window entry point (``evals/extraction.py`` uses it); the ingestion
    job drives the same three steps itself so the middle one can fan out.

    Calls the injected ``provider`` and resolves a ``RUNNING`` row to exactly one
    terminal status (DECISIONS #1):

    - ``FAILED`` — the provider raised ``LLMTechnicalError`` (transport/system).
      The ``FAILED`` row is recorded, then the error is re-raised so the job's
      ``mark_failed`` path engages (DECISIONS #6).
    - ``REJECTED`` — the provider returned ``output_json=None`` (parse / refusal /
      truncation), OR the parsed object failed ``recipe.v1`` Pydantic validation.
    - ``SUCCESS`` — a valid parsed ``recipe.v1`` object.

    Only flushes; the caller owns the transaction (mirrors ``pdf_text``). The
    ``provider``/``model`` labels come from the injected provider (DECISIONS #7).
    """
    # Cache check (DECISIONS #2) first: a hit reuses a prior SUCCESS run's
    # output_json, records a fresh audit row, and never reaches the provider.
    cached_run = await find_cached_extraction(
        session,
        window,
        source_version=source_version,
        document_id=document_id,
        provider=provider,
    )
    if cached_run is not None:
        return cached_run

    outcome = await call_provider_for_window(
        window, provider=provider, document_id=document_id, observability=observability
    )
    return await record_window_extraction(
        session,
        outcome,
        source_version=source_version,
        document_id=document_id,
        provider=provider,
    )
