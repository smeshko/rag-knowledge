"""The ``answer.v1`` JSON schema for the query-time answer structured output (doc 8 § 6).

The schema constrains only the *shape* of the model's answer. Semantic citation
validity — every cited ``citation_id`` exists in the context pack, every cited
``knowledge_item_id`` came from retrieval, every recommendation carries ≥1 citation
— is enforced in code in Phase 17.2 (doc 8 § 7), not by this schema.

Authored for OpenAI strict structured-output mode: every object lists all of its
properties in ``required`` and sets ``additionalProperties: false`` (the same
contract ``build_recipe_v1_json_schema`` satisfies for extraction).
"""

from __future__ import annotations

from typing import Any

__all__ = ["ANSWER_SCHEMA_VERSION", "ANSWER_V1_SCHEMA", "build_answer_v1_json_schema"]

#: Mirrors ``Settings.answer_schema_version`` default; a drift test guards the pair.
ANSWER_SCHEMA_VERSION = "answer.v1"


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


ANSWER_V1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "recommendations", "citations"],
    "properties": {
        "answer": {
            "type": "object",
            "additionalProperties": False,
            "required": ["style", "text", "citations"],
            "properties": {
                "style": {"type": "string"},
                "text": {"type": "string"},
                # The ``cite_N`` ids the answer text relies on.
                "citations": _string_array(),
            },
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["knowledge_item_id", "reason", "citation_ids"],
                "properties": {
                    "knowledge_item_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "citation_ids": _string_array(),
                },
            },
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "citation_id",
                    "knowledge_item_id",
                    "source_span_id",
                    "label",
                ],
                "properties": {
                    "citation_id": {"type": "string"},
                    "knowledge_item_id": {"type": "string"},
                    "source_span_id": {"type": "string"},
                    "label": {"type": "string"},
                },
            },
        },
    },
}


def build_answer_v1_json_schema() -> dict[str, Any]:
    """Return a fresh deep copy of ``ANSWER_V1_SCHEMA`` (callers may mutate safely)."""
    import copy

    return copy.deepcopy(ANSWER_V1_SCHEMA)
