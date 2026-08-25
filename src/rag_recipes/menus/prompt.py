"""The menu decomposition and selection prompts.

The menu layer makes two LLM calls with opposite jobs, so the prompts pull in
opposite directions and are versioned separately:

* **Plan** (``menu-plan-v1``) is a *rewrite* task with no retrieved context. Its
  whole value is turning each clause of a multi-dish request into a retrieval query
  that works on its own. Echoing the user's clause verbatim is the failure mode to
  avoid — "something to drink that goes together" retrieves nothing useful, while
  "refreshing non-alcoholic drink to serve with dinner" does.
* **Select** (``menu-selection-v1``) is a *grounded* task and therefore shares the
  answer layer's ``GROUNDING_RULES`` verbatim (imported, not copied, so the two
  layers can never drift apart). It sees per-slot candidates and must pick exactly
  one per slot using only ``cite_N`` ids from the pack.

``render_menu_selection_input`` prints the slot → candidate mapping *above* the pack
so the model is told which items are eligible for which course; the pack itself is a
flat list, and nothing but this mapping distinguishes a dessert candidate from a
salad one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from rag_recipes.answers.context_pack import ContextPack
from rag_recipes.answers.prompt import GROUNDING_RULES
from rag_recipes.menus.types import Course, MenuPlan

__all__ = [
    "MENU_PLAN_PROMPT",
    "MENU_PLAN_PROMPT_VERSION",
    "MENU_SELECTION_PROMPT",
    "MENU_SELECTION_PROMPT_VERSION",
    "course_from_json",
    "render_menu_plan_input",
    "render_menu_selection_input",
]

#: Mirrors ``Settings.menu_plan_prompt_version``; a drift test guards the pair.
MENU_PLAN_PROMPT_VERSION = "menu-plan-v1"
#: Mirrors ``Settings.menu_selection_prompt_version``; a drift test guards the pair.
MENU_SELECTION_PROMPT_VERSION = "menu-selection-v1"


MENU_PLAN_PROMPT = """\
You plan menus for a personal recipe library. Given a request that names several \
dishes at once, split it into one course per dish that was asked for.

Rules:
- Emit exactly one course per dish the request names. Do not add courses the \
request did not ask for, and do not merge two requested dishes into one course.
- slot is a short lowercase label naming the course's role: "starter", "salad", \
"soup", "main", "side", "dessert", "drink". Each slot must be unique.
- query is a standalone retrieval query for a recipe search engine, written in \
cookbook vocabulary — dish type, key ingredients, technique. Rewrite the user's \
words rather than copying them: drop conversational framing ("I want", "something \
that goes together") and keep the words a recipe's own title, ingredients, or \
method would actually contain.
- Carry every constraint the request states into the query itself when it is a \
property of the dish ("no-bake", "chocolate", "light", "hearty", "vegetarian"), \
because retrieval only sees the query.
- note is a short free-text restatement of that course's constraints, for the \
later selection step. Leave it "" when the request states none.
- theme is one line describing what would make these dishes work as a single meal, \
or "" if the request implies no particular occasion or cuisine.

Return only the courses; nothing has been retrieved yet, so do not name recipes."""


MENU_SELECTION_PROMPT = f"""\
You assemble a menu for a personal recipe library. Each course below has its own \
list of candidate recipes retrieved from the library. Choose exactly one recipe per \
course and explain why the chosen dishes work as one meal.

Grounding rules:
{GROUNDING_RULES}

Produce a structured menu that conforms to the menu_selection.v1 schema:
- courses[] has one entry per slot listed below, in that order, each naming the \
slot, the chosen knowledge_item_id, a short reason, and its citation_ids \
(at least one).
- Choose each course's item from that slot's candidates only. Never choose the same \
recipe for two courses.
- menu.title names the menu in a few words.
- menu.text explains in two or three sentences how the courses balance each other \
— flavour, richness, temperature, effort, and any shared or clashing ingredients. \
Ground every claim about a dish in that dish's context.
- menu.citations lists the citation IDs menu.text relies on (at least one).

If a slot's candidates contain nothing that honestly fits, still pick that slot's \
closest candidate and say plainly in menu.text why it is a compromise. Do not \
invent a recipe, and do not leave a slot unfilled.

Only use citation IDs and knowledge_item_ids that appear in the context pack below."""


def render_menu_plan_input(query: str, *, max_courses: int) -> str:
    """Compose the planner input: the prompt, the course cap, and the raw request."""
    return f"{MENU_PLAN_PROMPT}\n\nEmit at most {max_courses} courses.\n\nRequest: {query}"


def render_menu_selection_input(
    query: str,
    plan: MenuPlan,
    pack: ContextPack,
    *,
    candidate_ids_by_slot: Mapping[str, Sequence[str]],
) -> str:
    """Compose the selection input: prompt + slot/candidate mapping + serialized pack.

    Deterministic for fixed inputs (the pack is dumped with dataclass field order and
    the id lists are emitted in pack order), matching ``render_answer_input`` — the
    allowed-id footer is the same id space ``validate_menu_selection`` checks the
    model's output against.

    Slots whose candidate list is empty are still printed, marked ``(no candidates)``.
    Dropping them silently would leave the model to infer the course was never asked
    for; the service treats an unfillable slot as a course with no selection.
    """
    pack_json = json.dumps(asdict(pack), indent=2, ensure_ascii=False)

    lines: list[str] = []
    for course in plan.courses:
        ids = list(candidate_ids_by_slot.get(course.slot, ()))
        note = f" [{course.note}]" if course.note else ""
        candidates = ", ".join(ids) if ids else "(no candidates)"
        lines.append(f'- {course.slot}: searched "{course.query}"{note} — candidates: {candidates}')

    allowed_citation_ids: list[str] = []
    allowed_item_ids: list[str] = []
    for item in pack.items:
        if item.knowledge_item_id not in allowed_item_ids:
            allowed_item_ids.append(item.knowledge_item_id)
        for citation in item.citations:
            if citation.citation_id not in allowed_citation_ids:
                allowed_citation_ids.append(citation.citation_id)

    theme_line = f"Menu theme: {plan.theme}\n" if plan.theme else ""
    return (
        f"{MENU_SELECTION_PROMPT}\n\n"
        f"User request: {query}\n"
        f"{theme_line}\n"
        f"Courses to fill (in order):\n" + "\n".join(lines) + "\n\n"
        f"Context pack:\n{pack_json}\n\n"
        f"Allowed citation IDs: {', '.join(allowed_citation_ids) or '(none)'}\n"
        f"Allowed knowledge_item_ids: {', '.join(allowed_item_ids) or '(none)'}"
    )


def course_from_json(raw: object) -> Course | None:
    """Coerce one planner ``courses[]`` entry to a ``Course`` (``None`` when unusable).

    A course with no ``slot`` or no ``query`` cannot be retrieved for or bound to a
    selection, so it is dropped rather than repaired — the planner is re-runnable and
    a half-specified course would silently distort the menu.
    """
    if not isinstance(raw, dict):
        return None
    slot = raw.get("slot")
    query = raw.get("query")
    note = raw.get("note")
    if not isinstance(slot, str) or not slot.strip():
        return None
    if not isinstance(query, str) or not query.strip():
        return None
    return Course(
        slot=slot.strip().lower(),
        query=query.strip(),
        note=note.strip() if isinstance(note, str) else "",
    )
