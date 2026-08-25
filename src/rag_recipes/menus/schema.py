"""The ``menu_plan.v1`` and ``menu_selection.v1`` structured-output schemas.

Two LLM calls bracket the menu pipeline, so there are two schemas:

* ``menu_plan.v1`` constrains the *decomposition* — the user's multi-dish request
  turned into per-course retrieval queries. Nothing has been retrieved yet, so
  there is nothing to cite and nothing to validate against beyond shape.
* ``menu_selection.v1`` constrains the *selection* — one pick per course, drawn
  from the retrieved candidates, plus the prose explaining why they go together.

As with ``answers.schema``, these constrain only *shape*. Semantic validity — every
cited ``cite_N`` exists in the context pack, every picked ``knowledge_item_id`` came
from retrieval, every slot was actually planned, no dish is served twice — is
enforced in code by ``menus.service.validate_menu_selection``.

Authored for OpenAI strict structured-output mode: every object lists all of its
properties in ``required`` and sets ``additionalProperties: false``.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "MENU_PLAN_SCHEMA_VERSION",
    "MENU_PLAN_V1_SCHEMA",
    "MENU_SELECTION_SCHEMA_VERSION",
    "MENU_SELECTION_V1_SCHEMA",
    "build_menu_plan_v1_json_schema",
    "build_menu_selection_v1_json_schema",
]

#: Mirrors ``Settings.menu_plan_schema_version`` default; a drift test guards the pair.
MENU_PLAN_SCHEMA_VERSION = "menu_plan.v1"
#: Mirrors ``Settings.menu_selection_schema_version`` default; a drift test guards it.
MENU_SELECTION_SCHEMA_VERSION = "menu_selection.v1"


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


MENU_PLAN_V1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["theme", "courses"],
    "properties": {
        # A one-line unifying idea for the menu ("" when the request implies none).
        # Carried into the selection prompt so both calls share the same framing.
        "theme": {"type": "string"},
        "courses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["slot", "query", "note"],
                "properties": {
                    "slot": {"type": "string"},
                    "query": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
    },
}


MENU_SELECTION_V1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["menu", "courses"],
    "properties": {
        "menu": {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "text", "citations"],
            "properties": {
                "title": {"type": "string"},
                # Why these dishes work as one meal — the coherence argument.
                "text": {"type": "string"},
                "citations": _string_array(),
            },
        },
        "courses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["slot", "knowledge_item_id", "reason", "citation_ids"],
                "properties": {
                    "slot": {"type": "string"},
                    "knowledge_item_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "citation_ids": _string_array(),
                },
            },
        },
    },
}


def build_menu_plan_v1_json_schema() -> dict[str, Any]:
    """Return a fresh deep copy of ``MENU_PLAN_V1_SCHEMA`` (callers may mutate safely)."""
    return copy.deepcopy(MENU_PLAN_V1_SCHEMA)


def build_menu_selection_v1_json_schema() -> dict[str, Any]:
    """Return a fresh deep copy of ``MENU_SELECTION_V1_SCHEMA`` (safe to mutate)."""
    return copy.deepcopy(MENU_SELECTION_V1_SCHEMA)
