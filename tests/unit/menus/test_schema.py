"""Unit tests for the menu structured-output schemas."""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.config import get_settings
from rag_recipes.menus.schema import (
    MENU_PLAN_SCHEMA_VERSION,
    MENU_SELECTION_SCHEMA_VERSION,
    build_menu_plan_v1_json_schema,
    build_menu_selection_v1_json_schema,
)


def _assert_strict(node: Any, where: str) -> None:
    """Every object node must be strict-mode clean: all properties required, closed."""
    if not isinstance(node, dict):
        return
    if node.get("type") == "object":
        properties = node.get("properties", {})
        assert node.get("additionalProperties") is False, f"{where} is not closed"
        assert set(node.get("required", [])) == set(properties), (
            f"{where}: required must list every property"
        )
        for name, child in properties.items():
            _assert_strict(child, f"{where}.{name}")
    if node.get("type") == "array":
        _assert_strict(node.get("items"), f"{where}[]")


@pytest.mark.parametrize(
    "builder", [build_menu_plan_v1_json_schema, build_menu_selection_v1_json_schema]
)
def test_schemas_are_openai_strict_mode_clean(builder: Any) -> None:
    _assert_strict(builder(), builder.__name__)


@pytest.mark.parametrize(
    "builder", [build_menu_plan_v1_json_schema, build_menu_selection_v1_json_schema]
)
def test_builders_return_an_isolated_copy(builder: Any) -> None:
    """A caller mutating the returned schema must not corrupt the module constant."""
    first = builder()
    first["properties"].clear()
    assert builder()["properties"], "module-level schema was mutated by a caller"


def test_plan_schema_shape() -> None:
    schema = build_menu_plan_v1_json_schema()
    course = schema["properties"]["courses"]["items"]
    assert set(course["properties"]) == {"slot", "query", "note"}


def test_selection_schema_shape() -> None:
    schema = build_menu_selection_v1_json_schema()
    assert set(schema["properties"]["menu"]["properties"]) == {"title", "text", "citations"}
    course = schema["properties"]["courses"]["items"]
    assert set(course["properties"]) == {
        "slot",
        "knowledge_item_id",
        "reason",
        "citation_ids",
    }


def test_schema_versions_do_not_drift_from_settings() -> None:
    settings = get_settings()
    assert settings.menu_plan_schema_version == MENU_PLAN_SCHEMA_VERSION
    assert settings.menu_selection_schema_version == MENU_SELECTION_SCHEMA_VERSION
