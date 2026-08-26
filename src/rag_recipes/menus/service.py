"""Menu service: plan → per-course retrieval → cross-course assignment → selection.

``generate_menu`` orchestrates the multi-dish flow and returns an internal
``MenuResult`` the route projects to ``MenuResponse``. It reuses the retrieval facade
and the answer layer's context-pack machinery wholesale; what is new here is the
three things a single query cannot do:

1. **Decomposition** (``menus.plan``) — one retrieval query per course, so each
   course is searched with an embedding that actually points at it.
2. **Cross-course assignment** — a recipe may rank for two courses at once (a
   "Dark Chocolate Strawberry Bowl" is both a salad-shaped bowl and a dessert), so
   candidates are assigned to at most one course before selection. Without this the
   model can serve the same dish twice, which is the multi-course version of the
   very failure the layer exists to fix.
3. **Coherence selection** — one grounded LLM call picks one dish per course and
   argues why they work together.

The grounding guarantee is the answer layer's, enforced here rather than trusted to
the model: every ``cite_N`` must exist in the pack, every pick must be a pack item
of *its own* slot's candidates, each pick's citations must belong to that pick, and
no dish may fill two slots. Response ``citations[]`` and every ``selection.title``
are reconstructed from the pack.

Every failure — planner, retrieval-yields-nothing, parse, invalid citations, or
``LLMTechnicalError`` — degrades to the same safe fallback: the top-scoring
candidate per course with no rationale, plus a warning. A menu request therefore
never returns a 5xx and never returns less structure than retrieval alone provides.

**Sequential retrieval is deliberate.** The per-course searches are independent and
look parallelizable, but they share one ``AsyncSession``, which is not safe for
concurrent use. Fanning out would need a session-per-course from the app's
``session_factory``; that is a worthwhile optimization (latency is currently the sum
of N embedding round-trips) but it is not free, and correctness comes first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.answers.context_pack import ContextPack, build_context_pack
from rag_recipes.answers.service import build_response_citations
from rag_recipes.answers.shared import as_list, fetch_chunk_inputs, previews_from_structured
from rag_recipes.api.schemas.answers import AnswerCitation
from rag_recipes.api.schemas.menus import (
    CourseSelection,
    MenuBody,
    MenuCourse,
)
from rag_recipes.api.schemas.search import KnowledgeItemResult, RetrievalDebugInfo
from rag_recipes.api.search_projection import (
    build_retrieval_debug,
    fetch_item_structured_data,
    project_results,
)
from rag_recipes.config import Settings
from rag_recipes.menus.plan import plan_menu
from rag_recipes.menus.prompt import render_menu_selection_input
from rag_recipes.menus.schema import build_menu_selection_v1_json_schema
from rag_recipes.menus.types import Course, MenuPlan
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from rag_recipes.providers.reranker.base import RerankerProvider
from rag_recipes.retrieval.search import search
from rag_recipes.retrieval.types import (
    KnowledgeItemResult as RetrievalItemResult,
)
from rag_recipes.retrieval.types import (
    SearchDebug,
    SearchRequest,
    SearchResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FALLBACK_WARNING",
    "MenuDebugFields",
    "MenuResult",
    "generate_menu",
    "validate_menu_selection",
]

#: Shown when a structured menu was retrieved but no citation-safe selection is possible.
FALLBACK_WARNING = (
    "I found candidates for each course, but could not compose a citation-safe menu. "
    "Here is the closest match for each course instead."
)
#: Shown when retrieval returned nothing for any course.
NO_RESULTS_WARNING = "No relevant results were found for any course of this menu."
#: Shown when the planner could not split the request and it was searched as one query.
PLAN_FALLBACK_WARNING = (
    "This request could not be split into courses, so it was searched as a single query."
)


@dataclass
class MenuDebugFields:
    """Dev-only menu diagnostics the service computes; the route gates their exposure."""

    retrieval_mode: str
    model: str
    plan_prompt_version: str
    selection_prompt_version: str
    plan_is_fallback: bool = False
    course_count: int = 0
    candidate_count: int = 0
    context_item_count: int = 0
    citation_count: int = 0
    retrieval_debug_by_slot: dict[str, RetrievalDebugInfo] = field(default_factory=dict)


@dataclass
class MenuResult:
    """Internal result the menus route projects into ``MenuResponse``."""

    query: str
    theme: str
    menu: MenuBody
    courses: list[MenuCourse] = field(default_factory=list)
    citations: list[AnswerCitation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_fallback: bool = False
    debug: MenuDebugFields | None = None


@dataclass
class _CourseCandidates:
    """A course plus the retrieval items assigned to it, best first."""

    course: Course
    items: list[RetrievalItemResult] = field(default_factory=list)


async def generate_menu(
    session: AsyncSession,
    request: SearchRequest,
    *,
    include_candidates: bool,
    max_courses: int,
    candidates_per_course: int,
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    reranker: RerankerProvider | None = None,
) -> MenuResult:
    """Plan a menu for ``request.query``, retrieve per course, and select coherently.

    ``request`` is a template: its filters/mode/category apply to every course, and
    only ``query`` and ``limit`` are replaced per course. ``max_courses`` and
    ``candidates_per_course`` are the route-normalized positive effective values.
    """
    plan = await plan_menu(
        request.query,
        llm_provider=llm_provider,
        settings=replace_settings_max_courses(settings, max_courses),
    )
    debug = MenuDebugFields(
        retrieval_mode=request.mode,
        model=llm_provider.default_model,
        plan_prompt_version=settings.menu_plan_prompt_version,
        selection_prompt_version=settings.menu_selection_prompt_version,
        plan_is_fallback=plan.is_fallback,
        course_count=len(plan.courses),
    )
    warnings: list[str] = [PLAN_FALLBACK_WARNING] if plan.is_fallback else []

    per_course = await _retrieve_courses(
        session,
        request,
        plan,
        candidates_per_course=candidates_per_course,
        embedding_provider=embedding_provider,
        settings=settings,
        reranker=reranker,
        debug=debug,
    )
    assigned = _assign_across_courses(per_course, cap=candidates_per_course)
    flat_items = [item for bucket in assigned for item in bucket.items]
    debug.candidate_count = len(flat_items)

    if not flat_items:
        return _empty_result(
            request.query, plan, warnings=[*warnings, NO_RESULTS_WARNING], debug=debug
        )

    # One synthetic envelope over the union of assigned items lets the whole
    # projection layer (structured data, citation locators, result envelope) be
    # reused unchanged — every helper it feeds reads only `.items`.
    flat_result = SearchResult(items=flat_items, debug=_synthetic_debug(request))
    structured = await fetch_item_structured_data(session, flat_result)
    projected = await project_results(session, flat_result, structured=structured)
    projected_by_id = {row.item.id: row for row in projected}

    pack = await _build_menu_pack(
        session,
        request.query,
        flat_items,
        structured=structured,
        chunks_per_item=settings.menu_matched_chunks_per_item,
    )
    debug.context_item_count = len(pack.items)
    debug.citation_count = sum(len(item.citations) for item in pack.items)

    slot_by_item_id = {
        item.item.knowledge_item_id: bucket.course.slot
        for bucket in assigned
        for item in bucket.items
    }
    # Derived from the *pack*, not from the assignment: the pack drops items with no
    # citable chunk, and an item the model cannot cite is not a candidate it may pick.
    candidate_ids_by_slot: dict[str, list[str]] = {c.slot: [] for c in plan.courses}
    for pack_item in pack.items:
        slot = slot_by_item_id.get(pack_item.knowledge_item_id)
        if slot is not None:
            candidate_ids_by_slot[slot].append(pack_item.context_item_id)

    if not pack.items:
        return _fallback_result(
            request.query,
            plan,
            assigned,
            projected_by_id,
            include_candidates=include_candidates,
            warnings=[*warnings, FALLBACK_WARNING],
            debug=debug,
        )

    try:
        response = await llm_provider.generate_structured_output(
            StructuredOutputRequest(
                provider=llm_provider.provider,
                model=llm_provider.default_model,
                prompt_version=settings.menu_selection_prompt_version,
                schema_version=settings.menu_selection_schema_version,
                input=render_menu_selection_input(
                    request.query, plan, pack, candidate_ids_by_slot=candidate_ids_by_slot
                ),
                json_schema=build_menu_selection_v1_json_schema(),
            )
        )
    except LLMTechnicalError:
        return _fallback_result(
            request.query,
            plan,
            assigned,
            projected_by_id,
            include_candidates=include_candidates,
            warnings=[*warnings, FALLBACK_WARNING],
            debug=debug,
        )

    selection_json = response.output_json
    if response.parse_error is not None or selection_json is None:
        return _fallback_result(
            request.query,
            plan,
            assigned,
            projected_by_id,
            include_candidates=include_candidates,
            warnings=[*warnings, FALLBACK_WARNING],
            debug=debug,
        )

    errors = validate_menu_selection(
        selection_json, pack, plan=plan, slot_by_item_id=slot_by_item_id
    )
    if errors:
        logger.info("menu selection rejected: %s", "; ".join(errors[:5]))
        return _fallback_result(
            request.query,
            plan,
            assigned,
            projected_by_id,
            include_candidates=include_candidates,
            warnings=[*warnings, FALLBACK_WARNING],
            debug=debug,
        )

    # validate_menu_selection checks membership, binding and structure, but not every
    # scalar type (a truthy non-string `text`/`reason` would trip Pydantic) — guard
    # construction so a residual surprise degrades to the fallback, never a 500.
    try:
        menu_body, courses, citations = _build_success_payload(
            selection_json,
            plan,
            pack,
            assigned,
            projected_by_id,
            include_candidates=include_candidates,
        )
    except (ValidationError, TypeError):
        return _fallback_result(
            request.query,
            plan,
            assigned,
            projected_by_id,
            include_candidates=include_candidates,
            warnings=[*warnings, FALLBACK_WARNING],
            debug=debug,
        )

    return MenuResult(
        query=request.query,
        theme=plan.theme,
        menu=menu_body,
        courses=courses,
        citations=citations,
        warnings=warnings,
        is_fallback=False,
        debug=debug,
    )


def replace_settings_max_courses(settings: Settings, max_courses: int) -> Settings:
    """Return ``settings`` with ``menu_max_courses`` overridden for this request.

    The planner reads its cap from ``Settings`` (so a deployment can bound it) but
    the request may lower it; copying the model is cheaper and safer than threading
    a second cap argument through the planner's signature.
    """
    if max_courses == settings.menu_max_courses:
        return settings
    return settings.model_copy(update={"menu_max_courses": max_courses})


async def _retrieve_courses(
    session: AsyncSession,
    request: SearchRequest,
    plan: MenuPlan,
    *,
    candidates_per_course: int,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    reranker: RerankerProvider | None,
    debug: MenuDebugFields,
) -> list[_CourseCandidates]:
    """Search once per course, sequentially (shared session — see module docstring).

    Over-fetches ``candidates_per_course * 2`` per course so cross-course assignment
    has slack: an item taken by an earlier-scoring course must be replaceable, or a
    contested dish would shrink the losing course's candidate list.
    """
    over_fetch = max(candidates_per_course * 2, candidates_per_course + 1)
    buckets: list[_CourseCandidates] = []
    for course in plan.courses:
        course_request = replace(request, query=course.query, limit=over_fetch)
        result = await search(
            session,
            course_request,
            provider=embedding_provider,
            settings=settings,
            reranker=reranker,
        )
        debug.retrieval_debug_by_slot[course.slot] = build_retrieval_debug(result, settings)
        buckets.append(_CourseCandidates(course=course, items=list(result.items)))
    return buckets


def _assign_across_courses(
    buckets: list[_CourseCandidates], *, cap: int
) -> list[_CourseCandidates]:
    """Assign each retrieved item to at most one course, then cap each course.

    Globally greedy by ``item_score``: the highest-scoring (course, item) pair wins
    first, so a dish that ranks for two courses lands in the one that wanted it more.
    Scores come from different queries and so are not strictly comparable in an
    absolute sense, but they are on the same RRF scale, and the ordering only ever
    decides *contested* items — an uncontested item is assigned regardless of rank.
    Course order is preserved; ties break on course order then item id, so the
    assignment is deterministic for fixed inputs.
    """
    ranked = sorted(
        (
            (-item.item_score, index, item.item.knowledge_item_id, index, item)
            for index, bucket in enumerate(buckets)
            for item in bucket.items
        ),
        key=lambda row: (row[0], row[1], row[2]),
    )
    taken: set[str] = set()
    assigned: list[list[RetrievalItemResult]] = [[] for _ in buckets]
    for _score, _order, item_id, index, item in ranked:
        if item_id in taken or len(assigned[index]) >= cap:
            continue
        taken.add(item_id)
        assigned[index].append(item)

    # Re-sort each course's picks by score: greedy assignment visits them in global
    # score order, which is already per-course descending, but an explicit sort keeps
    # the invariant local rather than incidental.
    return [
        _CourseCandidates(
            course=bucket.course,
            items=sorted(items, key=lambda i: (-i.item_score, i.item.knowledge_item_id)),
        )
        for bucket, items in zip(buckets, assigned, strict=True)
    ]


async def _build_menu_pack(
    session: AsyncSession,
    query: str,
    items: list[RetrievalItemResult],
    *,
    structured: dict[str, dict[str, Any]],
    chunks_per_item: int,
) -> ContextPack:
    """Build one context pack over every assigned candidate across all courses.

    A single pack (rather than one per course) keeps ``cite_N`` ids unique across the
    whole menu, which is what lets ``menu.text`` cite dishes from several courses in
    one sentence when arguing that they go together.
    """
    chunk_ids = [ref.chunk_id for item in items for ref in item.matched_chunks[:chunks_per_item]]
    chunk_input_by_id = await fetch_chunk_inputs(session, chunk_ids)
    previews = previews_from_structured(structured)
    return build_context_pack(
        query,
        items,
        chunk_input_by_id,
        item_limit=len(items),
        chunks_per_item=chunks_per_item,
        structured_preview_by_item_id=previews,
    )


def validate_menu_selection(
    selection_json: dict[str, Any],
    pack: ContextPack,
    *,
    plan: MenuPlan,
    slot_by_item_id: dict[str, str],
) -> list[str]:
    """Return selection-validation errors (empty ⇒ valid). Membership, binding, slots.

    Rules: (a) every cited ``citation_id`` exists in the pack; (b) every course entry
    names a planned slot, with no slot filled twice; (c) every picked
    ``knowledge_item_id`` is a pack item **assigned to that entry's slot** — picking a
    dessert for the salad course is a slot violation even though the id is in the
    pack; (d) no recipe fills two courses; (e) each entry carries ≥1 citation, and
    every one of them belongs to *that* entry's item; (f) every slot that has at
    least one pack candidate is filled — an unfilled fillable slot means the caller
    asked for a course the menu silently dropped; (g) the menu carries ≥1 valid
    citation overall.

    Structurally malformed but parseable output is itself a validation error, so a
    non-conforming response funnels to the safe fallback rather than crashing.
    """
    cite_owner: dict[str, str] = {}
    for item in pack.items:
        for citation in item.citations:
            cite_owner[citation.citation_id] = item.knowledge_item_id
    pack_item_ids = {item.knowledge_item_id for item in pack.items}
    planned_slots = {course.slot for course in plan.courses}
    fillable_slots = {
        slot_by_item_id[item_id] for item_id in pack_item_ids if item_id in slot_by_item_id
    }

    errors: list[str] = []
    valid_cited: set[str] = set()

    menu_block = selection_json.get("menu")
    if not isinstance(menu_block, dict):
        errors.append("menu block is missing or not an object")
        menu_block = {}
    # `isinstance` is checked before the membership lookup so an unhashable nested
    # value (e.g. a list) can never reach `cid in cite_owner` and raise TypeError.
    for cid in as_list(menu_block.get("citations")):
        if not isinstance(cid, str) or cid not in cite_owner:
            errors.append(f"menu cites unknown citation_id {cid!r}")
        else:
            valid_cited.add(cid)

    courses = selection_json.get("courses")
    if not isinstance(courses, list):
        errors.append("courses is missing or not a list")
        courses = []

    seen_slots: set[str] = set()
    seen_items: set[str] = set()
    for index, entry in enumerate(courses):
        if not isinstance(entry, dict):
            errors.append(f"course[{index}] is not an object")
            continue
        slot = entry.get("slot")
        item_id = entry.get("knowledge_item_id")
        citation_ids = as_list(entry.get("citation_ids"))

        if not isinstance(slot, str) or slot not in planned_slots:
            errors.append(f"course[{index}] names unplanned slot {slot!r}")
            slot = None
        elif slot in seen_slots:
            errors.append(f"course[{index}] fills slot {slot!r} twice")
        else:
            seen_slots.add(slot)

        if not isinstance(item_id, str) or item_id not in pack_item_ids:
            errors.append(f"course[{index}] picks unknown knowledge_item_id {item_id!r}")
        elif item_id in seen_items:
            errors.append(f"course[{index}] serves {item_id!r} in more than one course")
        else:
            seen_items.add(item_id)
            if slot is not None and slot_by_item_id.get(item_id) != slot:
                errors.append(
                    f"course[{index}] picks {item_id!r}, which is a candidate for "
                    f"slot {slot_by_item_id.get(item_id)!r}, not {slot!r}"
                )

        if not citation_ids:
            errors.append(f"course[{index}] has no citations")
        for cid in citation_ids:
            if not isinstance(cid, str) or cid not in cite_owner:
                errors.append(f"course[{index}] cites unknown citation_id {cid!r}")
            elif cite_owner[cid] != item_id:
                errors.append(
                    f"course[{index}] cites {cid!r} which belongs to a different item "
                    f"({cite_owner[cid]!r}, not {item_id!r})"
                )
            else:
                valid_cited.add(cid)

    for slot in sorted(fillable_slots - seen_slots):
        errors.append(f"slot {slot!r} has candidates but was left unfilled")

    if not valid_cited:
        errors.append("menu has no valid citations")

    return errors


def _build_success_payload(
    selection_json: dict[str, Any],
    plan: MenuPlan,
    pack: ContextPack,
    assigned: list[_CourseCandidates],
    projected_by_id: dict[str, KnowledgeItemResult],
    *,
    include_candidates: bool,
) -> tuple[MenuBody, list[MenuCourse], list[AnswerCitation]]:
    """Reconstruct the menu body, courses, and citations from the pack (pure, no I/O).

    Courses are emitted in **plan** order, not the model's, and each ``title`` comes
    from the pack — the model supplies only ids and prose.
    """
    menu_block = selection_json.get("menu")
    menu_block = menu_block if isinstance(menu_block, dict) else {}
    title_by_item = {item.knowledge_item_id: item.title for item in pack.items}

    entry_by_slot: dict[str, dict[str, Any]] = {}
    for entry in as_list(selection_json.get("courses")):
        if isinstance(entry, dict) and isinstance(entry.get("slot"), str):
            entry_by_slot.setdefault(entry["slot"], entry)

    menu_citation_ids = [c for c in as_list(menu_block.get("citations")) if isinstance(c, str)]
    used_cite_ids: list[str] = list(menu_citation_ids)

    candidates_by_slot = {
        bucket.course.slot: [
            projected_by_id[item.item.knowledge_item_id]
            for item in bucket.items
            if item.item.knowledge_item_id in projected_by_id
        ]
        for bucket in assigned
    }

    courses: list[MenuCourse] = []
    for course in plan.courses:
        entry = entry_by_slot.get(course.slot)
        selection: CourseSelection | None = None
        if entry is not None:
            item_id = entry.get("knowledge_item_id", "")
            citation_ids = [c for c in as_list(entry.get("citation_ids")) if isinstance(c, str)]
            for cid in citation_ids:
                if cid not in used_cite_ids:
                    used_cite_ids.append(cid)
            selection = CourseSelection(
                knowledge_item_id=item_id,
                title=title_by_item.get(item_id, "") if isinstance(item_id, str) else "",
                reason=entry.get("reason") or "",
                citation_ids=citation_ids,
            )
        courses.append(
            MenuCourse(
                slot=course.slot,
                query=course.query,
                note=course.note,
                selection=selection,
                candidates=candidates_by_slot.get(course.slot, []) if include_candidates else [],
            )
        )

    menu_body = MenuBody(
        title=menu_block.get("title") or "",
        text=menu_block.get("text") or "",
        citations=menu_citation_ids,
    )
    return menu_body, courses, build_response_citations(used_cite_ids, pack)


def _fallback_result(
    query: str,
    plan: MenuPlan,
    assigned: list[_CourseCandidates],
    projected_by_id: dict[str, KnowledgeItemResult],
    *,
    include_candidates: bool,
    warnings: list[str],
    debug: MenuDebugFields | None,
) -> MenuResult:
    """The safe fallback: each course's best-scoring candidate, with no rationale.

    Strictly more useful than the answer layer's fallback, because per-course
    retrieval already produced a usable menu shape — only the coherence argument is
    missing. ``reason`` and ``citation_ids`` are left empty rather than fabricated,
    and the warning says so.
    """
    courses: list[MenuCourse] = []
    for bucket in assigned:
        best = bucket.items[0] if bucket.items else None
        selection: CourseSelection | None = None
        if best is not None:
            projected = projected_by_id.get(best.item.knowledge_item_id)
            selection = CourseSelection(
                knowledge_item_id=best.item.knowledge_item_id,
                title=projected.item.title if projected is not None else best.item.title,
                reason="",
                citation_ids=[],
            )
        courses.append(
            MenuCourse(
                slot=bucket.course.slot,
                query=bucket.course.query,
                note=bucket.course.note,
                selection=selection,
                candidates=[
                    projected_by_id[item.item.knowledge_item_id]
                    for item in bucket.items
                    if item.item.knowledge_item_id in projected_by_id
                ]
                if include_candidates
                else [],
            )
        )
    return MenuResult(
        query=query,
        theme=plan.theme,
        menu=MenuBody(title="", text=warnings[-1] if warnings else FALLBACK_WARNING, citations=[]),
        courses=courses,
        citations=[],
        warnings=warnings,
        is_fallback=True,
        debug=debug,
    )


def _empty_result(
    query: str, plan: MenuPlan, *, warnings: list[str], debug: MenuDebugFields | None
) -> MenuResult:
    """Nothing retrieved for any course — report the plan with every course unfilled."""
    return MenuResult(
        query=query,
        theme=plan.theme,
        menu=MenuBody(
            title="", text=warnings[-1] if warnings else NO_RESULTS_WARNING, citations=[]
        ),
        courses=[
            MenuCourse(slot=c.slot, query=c.query, note=c.note, selection=None, candidates=[])
            for c in plan.courses
        ],
        citations=[],
        warnings=warnings,
        is_fallback=True,
        debug=debug,
    )


def _synthetic_debug(request: SearchRequest) -> SearchDebug:
    """A placeholder ``SearchDebug`` for the union envelope fed to the projection layer.

    The projection helpers read only ``.items``; the real per-course diagnostics are
    kept in ``MenuDebugFields.retrieval_debug_by_slot``. Counts are zeroed rather
    than summed so nothing downstream mistakes this for a real search's debug.
    """
    return SearchDebug(
        mode=request.mode,
        normalized_query=request.query,
        keyword_candidates=0,
        vector_candidates=0,
        merged_chunks=0,
        grouped_items=0,
    )
