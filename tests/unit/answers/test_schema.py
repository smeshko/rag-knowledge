"""Unit tests for the answer.v1 JSON schema (shape only; semantics live in 17.2)."""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.answers.schema import (
    ANSWER_SCHEMA_VERSION,
    ANSWER_V1_SCHEMA,
    build_answer_v1_json_schema,
)
from rag_recipes.config import Settings


def _validate(schema: dict[str, Any], instance: Any, path: str = "$") -> None:
    """Validate ``instance`` against the subset of JSON Schema the answer schema uses.

    Supports ``type`` (object/array/string), ``properties``, ``required``,
    ``additionalProperties: false``, and array ``items`` — enough to assert the
    doc-8 § 6 shape without pulling in the optional ``jsonschema`` dependency.
    """
    expected_type = schema["type"]
    if expected_type == "object":
        if not isinstance(instance, dict):
            raise AssertionError(f"{path}: expected object, got {type(instance).__name__}")
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                raise AssertionError(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(properties)
            if extra:
                raise AssertionError(f"{path}: unexpected properties {sorted(extra)!r}")
        for key, value in instance.items():
            if key in properties:
                _validate(properties[key], value, f"{path}.{key}")
    elif expected_type == "array":
        if not isinstance(instance, list):
            raise AssertionError(f"{path}: expected array, got {type(instance).__name__}")
        for i, element in enumerate(instance):
            _validate(schema["items"], element, f"{path}[{i}]")
    elif expected_type == "string":
        if not isinstance(instance, str):
            raise AssertionError(f"{path}: expected string, got {type(instance).__name__}")
    else:  # pragma: no cover - defensive
        raise AssertionError(f"{path}: unhandled schema type {expected_type!r}")


_GOOD_PAYLOAD: dict[str, Any] = {
    "answer": {
        "style": "recommendation",
        "text": "A strong match is Tomato and White Bean Soup.",
        "citations": ["cite_1", "cite_2"],
    },
    "recommendations": [
        {
            "knowledge_item_id": "item_123",
            "reason": "Uses white beans directly and matches the soup request.",
            "citation_ids": ["cite_1"],
        }
    ],
    "citations": [
        {
            "citation_id": "cite_1",
            "knowledge_item_id": "item_123",
            "source_span_id": "span_042",
            "label": "Simple Thai Food, page 42",
        }
    ],
}


def test_schema_is_strict_object() -> None:
    assert ANSWER_V1_SCHEMA["type"] == "object"
    assert ANSWER_V1_SCHEMA["additionalProperties"] is False
    assert set(ANSWER_V1_SCHEMA["required"]) == {"answer", "recommendations", "citations"}
    # Every object level is strict (mirrors the recipe.v1 strict-mode contract).
    answer = ANSWER_V1_SCHEMA["properties"]["answer"]
    assert answer["additionalProperties"] is False
    assert set(answer["required"]) == {"style", "text", "citations"}
    rec = ANSWER_V1_SCHEMA["properties"]["recommendations"]["items"]
    assert rec["additionalProperties"] is False
    assert set(rec["required"]) == {"knowledge_item_id", "reason", "citation_ids"}
    cite = ANSWER_V1_SCHEMA["properties"]["citations"]["items"]
    assert cite["additionalProperties"] is False
    assert set(cite["required"]) == {
        "citation_id",
        "knowledge_item_id",
        "source_span_id",
        "label",
    }


def test_doc8_example_payload_validates() -> None:
    _validate(ANSWER_V1_SCHEMA, _GOOD_PAYLOAD)


def test_payload_with_extra_property_is_rejected() -> None:
    bad = {
        "answer": {**_GOOD_PAYLOAD["answer"], "unexpected": "x"},
        "recommendations": _GOOD_PAYLOAD["recommendations"],
        "citations": _GOOD_PAYLOAD["citations"],
    }
    with pytest.raises(AssertionError, match="unexpected properties"):
        _validate(ANSWER_V1_SCHEMA, bad)


def test_payload_missing_required_field_is_rejected() -> None:
    bad = {
        "answer": {"style": "recommendation", "text": "x"},  # missing citations
        "recommendations": _GOOD_PAYLOAD["recommendations"],
        "citations": _GOOD_PAYLOAD["citations"],
    }
    with pytest.raises(AssertionError, match="missing required property"):
        _validate(ANSWER_V1_SCHEMA, bad)


def test_build_returns_independent_copy() -> None:
    a = build_answer_v1_json_schema()
    a["properties"]["answer"]["required"].append("mutated")
    assert "mutated" not in ANSWER_V1_SCHEMA["properties"]["answer"]["required"]


def test_schema_version_matches_settings_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Drift guard: the module constant and the Settings default must agree so the
    # versioned StructuredOutputRequest in 17.2 stamps the right schema_version.
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.answer_schema_version == ANSWER_SCHEMA_VERSION
