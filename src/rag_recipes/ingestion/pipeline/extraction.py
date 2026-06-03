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

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
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
]


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
