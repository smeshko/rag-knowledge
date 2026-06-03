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
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.windows import (
    Window,
    compute_input_hash,
    format_window_for_llm,
)
from rag_recipes.providers._observability import ProviderObservability, TraceContext
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
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
    "StepConfidence",
    "build_recipe_v1_json_schema",
    "run_extraction",
]

logger = logging.getLogger(__name__)

# LLM API contract versions — module constants, NOT ``Settings`` (DECISIONS #3).
# They feed ``compute_input_hash`` and every ``ExtractionRun`` row, so they must
# travel atomically with the prompt template / schema code they describe. A
# meaningful change to either the prompt or the schema shape must bump these.
PROMPT_VERSION = "recipe-extraction-v1"
SCHEMA_VERSION = "recipe.v1"

_PROMPT_PLACEHOLDER = "{source_spans}"
_PROMPT_PACKAGE = "rag_recipes.ingestion.prompts"
_PROMPT_RESOURCE = "recipe_extraction_v1.md"


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


class IngredientConfidence(BaseModel):
    """Per-ingredient confidence scores (doc 4 § Confidence Scores)."""

    overall: float
    quantity: float
    unit: float
    item: float
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


class StepConfidence(BaseModel):
    """Per-step confidence scores (doc 4 § Confidence Scores)."""

    overall: float
    ordering: float


class ExtractedStep(BaseModel):
    """One method step (doc 4 § Strict Output Shape)."""

    step_number: int
    text: str
    source_span_ids: list[str]
    confidence: StepConfidence


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
    ingredients_text: str | None
    ingredients: list[ExtractedIngredient]
    steps_text: str | None
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

    Inserts a ``RUNNING`` row, calls the injected ``provider``, and resolves the
    row to exactly one terminal status (DECISIONS #1):

    - ``FAILED`` — the provider raised ``LLMTechnicalError`` (transport/system).
      The ``FAILED`` row is recorded, then the error is re-raised so the job's
      ``mark_failed`` path engages (DECISIONS #6).
    - ``REJECTED`` — the provider returned ``output_json=None`` (parse / refusal /
      truncation), OR the parsed object failed ``recipe.v1`` Pydantic validation.
    - ``SUCCESS`` — a valid parsed ``recipe.v1`` object.

    Only flushes; the caller owns the transaction (mirrors ``pdf_text``). The
    ``provider``/``model`` labels come from the injected provider (DECISIONS #7).
    """
    input_text = format_window_for_llm(window)
    input_hash = compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION)
    span_ids = window.span_ids
    provider_name = provider.provider
    model_name = provider.default_model

    run = ExtractionRun(
        document_id=document_id,
        source_version=source_version,
        provider=provider_name,
        model=model_name,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=span_ids,
        input_hash=input_hash,
        status=ExtractionRunStatus.RUNNING,
        output_json=None,
    )
    session.add(run)
    await session.flush()

    request = StructuredOutputRequest(
        provider=provider_name,
        model=model_name,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input=_render_prompt(input_text),
        json_schema=build_recipe_v1_json_schema(),
    )
    trace_context = TraceContext(
        session_id=document_id,
        input_hash=input_hash,
        input_source_span_ids=span_ids,
    )

    try:
        response = await provider.generate_structured_output(request, trace_context=trace_context)
    except LLMTechnicalError as exc:
        _finalize(run, status=ExtractionRunStatus.FAILED, error=str(exc))
        await session.flush()
        raise

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
