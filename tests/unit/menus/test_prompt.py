"""Unit tests for the menu prompts and their rendered model input."""

from __future__ import annotations

from rag_recipes.answers.prompt import GROUNDING_RULES
from rag_recipes.config import get_settings
from rag_recipes.menus.prompt import (
    MENU_PLAN_PROMPT_VERSION,
    MENU_SELECTION_PROMPT,
    MENU_SELECTION_PROMPT_VERSION,
    course_from_json,
    render_menu_plan_input,
    render_menu_selection_input,
)
from rag_recipes.menus.types import Course, MenuPlan
from tests.unit.menus.test_service import PACK, PLAN


def test_prompt_versions_do_not_drift_from_settings() -> None:
    settings = get_settings()
    assert settings.menu_plan_prompt_version == MENU_PLAN_PROMPT_VERSION
    assert settings.menu_selection_prompt_version == MENU_SELECTION_PROMPT_VERSION


def test_selection_prompt_shares_the_answer_layer_grounding_rules() -> None:
    """Imported, not copied — the two layers must not drift apart."""
    assert GROUNDING_RULES in MENU_SELECTION_PROMPT


def test_plan_input_carries_the_cap_and_the_raw_request() -> None:
    rendered = render_menu_plan_input("a salad and a dessert", max_courses=4)
    assert "at most 4 courses" in rendered
    assert "Request: a salad and a dessert" in rendered


def test_selection_input_maps_slots_to_their_candidates() -> None:
    rendered = render_menu_selection_input(
        "a salad and a dessert",
        PLAN,
        PACK,
        candidate_ids_by_slot={"salad": ["ctx_1"], "dessert": ["ctx_2"]},
    )
    assert '- salad: searched "green salad" — candidates: ctx_1' in rendered
    assert '- dessert: searched "mousse" — candidates: ctx_2' in rendered
    assert "Allowed citation IDs: cite_1, cite_2" in rendered
    assert "Allowed knowledge_item_ids: salad-item, dessert-item" in rendered


def test_a_slot_with_no_candidates_is_still_listed() -> None:
    """Dropping it silently would read as 'this course was never asked for'."""
    rendered = render_menu_selection_input(
        "q", PLAN, PACK, candidate_ids_by_slot={"salad": ["ctx_1"]}
    )
    assert '- dessert: searched "mousse" — candidates: (no candidates)' in rendered


def test_selection_input_is_deterministic() -> None:
    args = ("q", PLAN, PACK)
    kwargs = {"candidate_ids_by_slot": {"salad": ["ctx_1"], "dessert": ["ctx_2"]}}
    assert render_menu_selection_input(*args, **kwargs) == render_menu_selection_input(
        *args, **kwargs
    )


def test_theme_line_is_omitted_when_empty() -> None:
    themed = MenuPlan(theme="Autumn dinner", courses=list(PLAN.courses))
    assert "Menu theme: Autumn dinner" in render_menu_selection_input(
        "q", themed, PACK, candidate_ids_by_slot={}
    )
    assert "Menu theme:" not in render_menu_selection_input(
        "q", PLAN, PACK, candidate_ids_by_slot={}
    )


def test_course_from_json_normalizes_and_rejects() -> None:
    assert course_from_json({"slot": " Dessert ", "query": " mousse ", "note": " rich "}) == (
        Course(slot="dessert", query="mousse", note="rich")
    )
    assert course_from_json({"slot": "x", "query": ""}) is None
    assert course_from_json({"query": "no slot"}) is None
    assert course_from_json("not an object") is None
