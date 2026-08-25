"""Menu decomposition: one multi-dish request → one retrieval query per course.

This is the step that makes menu search work at all. Retrieval quality per course
is already good — "hearty casserole" and "chocolate no-bake dessert" each return
the right dishes — but the two asked together return neither, because one embedding
of a four-dish sentence sits near none of them. ``plan_menu`` is therefore the whole
fix in one function: it converts the request into N queries the existing facade
already answers well.

Every failure mode returns a **single-course fallback** carrying the raw query
rather than raising. A menu request that cannot be planned then behaves exactly like
``POST /api/v1/search`` does today — degraded, but never worse than the endpoint it
builds on, and never a 5xx. That mirrors the answer layer's safe-fallback contract.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_recipes.config import Settings
from rag_recipes.menus.prompt import course_from_json, render_menu_plan_input
from rag_recipes.menus.schema import build_menu_plan_v1_json_schema
from rag_recipes.menus.types import Course, MenuPlan
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest

__all__ = ["FALLBACK_SLOT", "plan_menu", "single_course_plan"]

logger = logging.getLogger(__name__)

#: The slot name a fallback plan's sole course carries.
FALLBACK_SLOT = "menu"


def single_course_plan(query: str) -> MenuPlan:
    """The degraded plan: one course whose query is the request, verbatim."""
    return MenuPlan(
        theme="",
        courses=[Course(slot=FALLBACK_SLOT, query=query, note="")],
        is_fallback=True,
    )


async def plan_menu(
    query: str,
    *,
    llm_provider: LLMProvider,
    settings: Settings,
) -> MenuPlan:
    """Decompose ``query`` into courses, or return a single-course fallback.

    The plan is normalized before it is trusted: slots are lowercased and
    de-duplicated (a repeated slot would make the selection binding ambiguous —
    two picks claiming one course), unusable entries are dropped, and the list is
    capped at ``settings.menu_max_courses``. A plan left with no usable course
    falls back rather than proceeding with an empty course list.
    """
    try:
        response = await llm_provider.generate_structured_output(
            StructuredOutputRequest(
                provider=llm_provider.provider,
                model=llm_provider.default_model,
                prompt_version=settings.menu_plan_prompt_version,
                schema_version=settings.menu_plan_schema_version,
                input=render_menu_plan_input(query, max_courses=settings.menu_max_courses),
                json_schema=build_menu_plan_v1_json_schema(),
            )
        )
    except LLMTechnicalError:
        logger.warning("menu planner failed technically; falling back to a single course")
        return single_course_plan(query)

    plan_json = response.output_json
    if response.parse_error is not None or plan_json is None:
        logger.warning("menu planner returned unparseable output; falling back")
        return single_course_plan(query)

    return _plan_from_json(plan_json, query, max_courses=settings.menu_max_courses)


def _plan_from_json(plan_json: dict[str, Any], query: str, *, max_courses: int) -> MenuPlan:
    """Normalize planner output into a ``MenuPlan`` (pure; falls back when unusable).

    ``parse_error is None`` guarantees only a JSON *object*, not a conforming one —
    the provider does no post-parse schema validation — so every field is coerced
    defensively here rather than trusted.
    """
    raw_courses = plan_json.get("courses")
    if not isinstance(raw_courses, list):
        logger.warning("menu planner emitted no courses list; falling back")
        return single_course_plan(query)

    courses: list[Course] = []
    seen_slots: set[str] = set()
    for raw in raw_courses:
        course = course_from_json(raw)
        if course is None or course.slot in seen_slots:
            continue
        seen_slots.add(course.slot)
        courses.append(course)
        if len(courses) >= max_courses:
            break

    if not courses:
        logger.warning("menu planner produced no usable course; falling back")
        return single_course_plan(query)

    theme = plan_json.get("theme")
    return MenuPlan(
        theme=theme.strip() if isinstance(theme, str) else "",
        courses=courses,
        is_fallback=False,
    )
