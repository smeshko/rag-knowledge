"""Unit tests for menu decomposition (``menus.plan``).

The planner is the step that makes menu search work, and every one of its failure
modes must land on the single-course fallback rather than raise — a menu request
that cannot be planned has to behave exactly like ``POST /api/v1/search``.
"""

from __future__ import annotations

import pytest

from rag_recipes.config import get_settings
from rag_recipes.menus.plan import FALLBACK_SLOT, plan_menu, single_course_plan
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputResponse, TokenUsage

pytestmark = pytest.mark.asyncio

_QUERY = "a light salad, a hearty casserole and a chocolate no-bake dessert"


def _settings(**overrides: object):
    return get_settings().model_copy(update=dict(overrides))


def _plan_payload(*courses: tuple[str, str, str], theme: str = "Cosy autumn dinner"):
    return {
        "theme": theme,
        "courses": [{"slot": slot, "query": query, "note": note} for slot, query, note in courses],
    }


async def test_plan_splits_a_multi_dish_request_into_courses() -> None:
    llm = FakeLLMProvider(
        default_output=_plan_payload(
            ("salad", "light green salad lemon vinaigrette", "light"),
            ("main", "hearty baked casserole", "hearty"),
            ("dessert", "no-bake chocolate dessert", "no-bake, chocolate"),
        )
    )
    plan = await plan_menu(_QUERY, llm_provider=llm, settings=_settings())

    assert plan.is_fallback is False
    assert plan.theme == "Cosy autumn dinner"
    assert [c.slot for c in plan.courses] == ["salad", "main", "dessert"]
    # The retrieval query must be the planner's rewrite, never the raw request.
    assert all(c.query != _QUERY for c in plan.courses)
    assert plan.courses[2].note == "no-bake, chocolate"


async def test_slots_are_lowercased_and_deduplicated() -> None:
    """A repeated slot would make the selection binding ambiguous — two picks, one course."""
    llm = FakeLLMProvider(
        default_output=_plan_payload(
            ("Salad", "green salad", ""),
            ("SALAD", "another salad", ""),
            ("Dessert", "chocolate mousse", ""),
        )
    )
    plan = await plan_menu(_QUERY, llm_provider=llm, settings=_settings())

    assert [c.slot for c in plan.courses] == ["salad", "dessert"]
    assert plan.courses[0].query == "green salad"


async def test_courses_are_capped_at_menu_max_courses() -> None:
    llm = FakeLLMProvider(
        default_output=_plan_payload(*[(f"slot{i}", f"query {i}", "") for i in range(9)])
    )
    plan = await plan_menu(_QUERY, llm_provider=llm, settings=_settings(menu_max_courses=3))

    assert len(plan.courses) == 3
    assert [c.slot for c in plan.courses] == ["slot0", "slot1", "slot2"]


async def test_entries_missing_a_slot_or_query_are_dropped() -> None:
    llm = FakeLLMProvider(
        default_output={
            "theme": "",
            "courses": [
                {"slot": "salad", "query": "green salad", "note": ""},
                {"slot": "", "query": "orphaned query", "note": ""},
                {"slot": "dessert", "query": "   ", "note": ""},
                {"slot": "drink", "query": "sparkling lemonade", "note": ""},
                "not an object",
            ],
        }
    )
    plan = await plan_menu(_QUERY, llm_provider=llm, settings=_settings())

    assert [c.slot for c in plan.courses] == ["salad", "drink"]


@pytest.mark.parametrize(
    "payload",
    [
        {"theme": "x"},  # no courses key at all
        {"theme": "x", "courses": "not a list"},
        {"theme": "x", "courses": []},
        {"theme": "x", "courses": [{"slot": "", "query": ""}]},
    ],
)
async def test_unusable_planner_output_falls_back_to_a_single_course(
    payload: dict[str, object],
) -> None:
    plan = await plan_menu(
        _QUERY, llm_provider=FakeLLMProvider(default_output=payload), settings=_settings()
    )

    assert plan.is_fallback is True
    assert [c.slot for c in plan.courses] == [FALLBACK_SLOT]
    assert plan.courses[0].query == _QUERY


async def test_technical_llm_failure_falls_back_instead_of_raising() -> None:
    plan = await plan_menu(
        _QUERY, llm_provider=FakeLLMProvider(fail_technically=True), settings=_settings()
    )

    assert plan.is_fallback is True
    assert plan.courses[0].query == _QUERY


async def test_parse_failure_falls_back() -> None:
    rejected = StructuredOutputResponse(
        output_json=None,
        parse_error="not JSON",
        raw_text="{oops",
        usage=TokenUsage(input_tokens=1, output_tokens=0),
        provider="fake",
        model="fake-model",
    )
    plan = await plan_menu(
        _QUERY, llm_provider=FakeLLMProvider(default_output=rejected), settings=_settings()
    )

    assert plan.is_fallback is True


async def test_planner_request_carries_the_configured_prompt_and_schema_versions() -> None:
    llm = FakeLLMProvider(default_output=_plan_payload(("salad", "green salad", "")))
    settings = _settings()
    await plan_menu(_QUERY, llm_provider=llm, settings=settings)

    (call,) = llm.calls
    assert call.prompt_version == settings.menu_plan_prompt_version
    assert call.schema_version == settings.menu_plan_schema_version
    assert _QUERY in call.input


async def test_single_course_plan_is_marked_as_a_fallback() -> None:
    plan = single_course_plan("anything")
    assert plan.is_fallback is True
    assert plan.theme == ""
    assert plan.courses[0].slot == FALLBACK_SLOT
