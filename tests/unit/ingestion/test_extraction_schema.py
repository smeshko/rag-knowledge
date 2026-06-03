"""Unit tests for the recipe.v1 Pydantic schema and strict JSON-schema export.

The verbatim doc-4 § Strict Output Shape example is the contract: it must
round-trip through ``RecipeExtractionOutput.model_validate`` and the exported
JSON schema must be OpenAI strict-mode (every object node lists all properties in
``required`` and sets ``additionalProperties: false``, with nullable fields
expressed as a ``"null"``-bearing type union).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from rag_recipes.ingestion.pipeline.extraction import (
    RecipeExtractionOutput,
    build_recipe_v1_json_schema,
)


def _doc4_example() -> dict[str, Any]:
    """The verbatim doc-4 § Strict Output Shape example payload."""
    return {
        "items": [
            {
                "item_type": "recipe",
                "title": "Tomato and White Bean Soup",
                "summary": "A simple soup with pantry ingredients.",
                "body_text": "Tomato and White Bean Soup\nServes 4\nIngredients...\nSteps...",
                "source_span_ids": ["span_042", "span_043"],
                "structured_data": {
                    "schema": "recipe.v1",
                    "yield": "Serves 4",
                    "prep_time": None,
                    "cook_time": "35 minutes",
                    "total_time": None,
                    "ingredients_text": "2 tbsp olive oil\n1 onion, diced...",
                    "ingredients": [
                        {
                            "position": 1,
                            "raw_text": "2 tbsp olive oil",
                            "quantity_text": "2",
                            "quantity_value": 2,
                            "unit_raw": "tbsp",
                            "unit_normalized": "tablespoon",
                            "item_text": "olive oil",
                            "item_normalized": "olive oil",
                            "preparation": None,
                            "notes": None,
                            "confidence": {
                                "overall": 0.95,
                                "quantity": 0.98,
                                "unit": 0.96,
                                "item": 0.97,
                                "normalization": 0.9,
                            },
                        }
                    ],
                    "steps_text": "Heat the oil in a large pot...",
                    "steps": [
                        {
                            "step_number": 1,
                            "text": "Heat the oil in a large pot.",
                            "source_span_ids": ["span_043"],
                            "confidence": {"overall": 0.92, "ordering": 0.9},
                        }
                    ],
                },
                "confidence": {
                    "overall": 0.88,
                    "boundary": 0.82,
                    "fields": {
                        "title": 0.96,
                        "summary": 0.84,
                        "yield": 0.9,
                        "ingredients": 0.91,
                        "steps": 0.87,
                    },
                },
                "warnings": [],
            }
        ]
    }


def test_doc4_example_round_trips_and_resolves_yield_alias() -> None:
    parsed = RecipeExtractionOutput.model_validate(_doc4_example())

    assert len(parsed.items) == 1
    recipe = parsed.items[0]
    assert recipe.title == "Tomato and White Bean Soup"
    # The ``yield`` alias maps onto the Python field ``yield_``.
    assert recipe.structured_data.yield_ == "Serves 4"
    assert recipe.confidence.fields.yield_ == 0.9
    # The ``schema`` discriminator alias maps onto ``schema_``.
    assert recipe.structured_data.schema_ == "recipe.v1"
    # First ingredient round-trips with normalized fields.
    ingredient = recipe.structured_data.ingredients[0]
    assert ingredient.unit_normalized == "tablespoon"
    assert ingredient.quantity_value == 2.0
    assert recipe.structured_data.steps[0].source_span_ids == ["span_043"]


def test_nullable_string_fields_accept_none() -> None:
    payload = _doc4_example()
    # All the nullable string fields are already ``None`` in the doc-4 example
    # (prep_time, total_time, ingredient preparation/notes, summary set below).
    payload["items"][0]["summary"] = None
    payload["items"][0]["structured_data"]["ingredients_text"] = None
    payload["items"][0]["structured_data"]["steps_text"] = None

    parsed = RecipeExtractionOutput.model_validate(payload)
    recipe = parsed.items[0]
    assert recipe.summary is None
    assert recipe.structured_data.prep_time is None
    assert recipe.structured_data.total_time is None
    assert recipe.structured_data.ingredients_text is None
    assert recipe.structured_data.steps_text is None
    assert recipe.structured_data.ingredients[0].preparation is None
    assert recipe.structured_data.ingredients[0].notes is None


def test_missing_required_field_raises_validation_error() -> None:
    payload = _doc4_example()
    del payload["items"][0]["title"]
    with pytest.raises(ValidationError):
        RecipeExtractionOutput.model_validate(payload)


def test_wrong_type_raises_validation_error() -> None:
    payload = _doc4_example()
    payload["items"][0]["structured_data"]["ingredients"][0]["position"] = "not-an-int"
    with pytest.raises(ValidationError):
        RecipeExtractionOutput.model_validate(payload)


def test_no_semantic_constraints_enforced_at_pydantic_layer() -> None:
    """9.2 enforces structure only — out-of-range confidence and a non-recipe
    ``item_type`` still validate structurally (HARD validation is 9.3)."""
    payload = _doc4_example()
    payload["items"][0]["item_type"] = "note"
    payload["items"][0]["confidence"]["overall"] = 1.5
    parsed = RecipeExtractionOutput.model_validate(payload)
    assert parsed.items[0].item_type == "note"
    assert parsed.items[0].confidence.overall == 1.5


def _type_allows_null(node: dict[str, Any]) -> bool:
    """True if a schema node's type permits ``null`` (handles ``type`` list and
    ``anyOf``)."""
    type_field = node.get("type")
    if isinstance(type_field, list) and "null" in type_field:
        return True
    if type_field == "null":
        return True
    for branch in node.get("anyOf", []):
        if isinstance(branch, dict) and _type_allows_null(branch):
            return True
    return False


def _iter_object_nodes(node: Any) -> list[dict[str, Any]]:
    """Collect every object-typed node (has ``properties`` or ``type==object``)."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            found.append(node)
        for value in node.values():
            found.extend(_iter_object_nodes(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_iter_object_nodes(item))
    return found


def test_json_schema_is_openai_strict_mode() -> None:
    schema = build_recipe_v1_json_schema()
    object_nodes = _iter_object_nodes(schema)
    assert object_nodes, "expected at least one object node in the schema"
    for node in object_nodes:
        assert node.get("additionalProperties") is False
        assert set(node.get("required", [])) == set(node.get("properties", {}).keys())


def test_json_schema_uses_yield_alias_and_nullable_union() -> None:
    schema = build_recipe_v1_json_schema()
    structured = schema["$defs"]["RecipeV1StructuredData"]
    # The yield property appears under its alias, not the Python field name.
    assert "yield" in structured["properties"]
    assert "yield_" not in structured["properties"]
    # ``schema`` discriminator appears under its alias too.
    assert "schema" in structured["properties"]
    # A known nullable field expresses ``null`` via a type union, and is STILL
    # listed in ``required`` (strict mode), never dropped.
    assert _type_allows_null(structured["properties"]["prep_time"])
    assert "prep_time" in structured["required"]
