"""Dependency-free planning types for the menu layer.

A *menu* request names several dishes at once ("a light salad, a hearty casserole,
a chocolate no-bake dessert and something to drink that goes together"). Retrieval
cannot serve that as one query: a single embedding of a multi-dish sentence lands
near none of the dishes, and ``group_by_item`` ranks purely by score, so whichever
course the embedding drifts toward sweeps the whole result list.

The menu layer therefore splits the request into *courses* first, retrieves each
one independently through the existing facade, and only then asks an LLM to pick a
combination that goes together. These are the planning types for that first step;
the result envelope lives in ``menus.service`` (mirroring ``answers.service``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Course", "MenuPlan"]


@dataclass(frozen=True)
class Course:
    """One course of a planned menu.

    ``slot`` is the short label the course is addressed by ("salad", "main",
    "dessert", "drink") — it is the key the selection step binds a pick to, so it
    must be unique within a plan. ``query`` is a self-contained retrieval query in
    cookbook vocabulary, NOT the user's original sentence: it is what actually goes
    to ``retrieval.search.search``. ``note`` carries the constraints the planner
    read off the request ("no-bake", "light") for the selection prompt's benefit;
    it never reaches retrieval.
    """

    slot: str
    query: str
    note: str = ""


@dataclass(frozen=True)
class MenuPlan:
    """The decomposed request: an optional unifying theme plus the ordered courses.

    ``is_fallback`` marks a plan that did not come from the planner LLM — a single
    course carrying the raw query. That degrades a menu request to exactly the
    behaviour of ``POST /api/v1/search`` today rather than failing the request,
    which is the same "never worse than retrieval alone" contract the answer layer
    keeps with its safe fallback.
    """

    theme: str = ""
    courses: list[Course] = field(default_factory=list)
    is_fallback: bool = False
