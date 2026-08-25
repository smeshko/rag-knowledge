"""Request/response Pydantic models for POST /api/v1/menus.

Pure API projections, parallel to ``api/schemas/answers.py``. Following that
module's rule — *the envelope wraps retrieval, it does not re-model it* — this one
reuses ``SearchFilters`` for the request, the search ``KnowledgeItemResult`` for
per-course candidates, and the answer layer's ``AnswerCitation`` for citation
detail. The citation shape is identical to the answer layer's and is rebuilt by the
very same ``build_response_citations``, so giving it a second model would be two
definitions guaranteed to drift.

``mode`` is a plain ``str`` so an invalid value lands in the doc-6
``invalid_request`` envelope via handler validation rather than FastAPI's raw 422.

As in the answer layer, ``citations[]`` and each ``selection.title`` are
reconstructed by the backend from the context pack, never trusted from the LLM —
the model emits only ``cite_N`` references and ``knowledge_item_id``s.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from rag_recipes.api.schemas.answers import AnswerCitation
from rag_recipes.api.schemas.search import (
    KnowledgeItemResult,
    RetrievalDebugInfo,
    SearchFilters,
)

__all__ = [
    "MenuBody",
    "MenuCourse",
    "MenuDebugInfo",
    "MenuOptions",
    "MenuRequestBody",
    "MenuResponse",
    "MenuRetrievalOptions",
    "CourseSelection",
]


# --- Request ---


class MenuRetrievalOptions(BaseModel):
    mode: str = "hybrid"
    #: Candidates retrieved *per course*, not for the menu as a whole. Defaults to
    #: ``Settings.menu_candidates_per_course``.
    candidates_per_course: int | None = None


class MenuOptions(BaseModel):
    #: Off by default: a menu response is the picks, and echoing every course's
    #: candidate list multiplies the payload by the candidate cap for a caller that
    #: usually only renders the selections.
    include_candidates: bool = False
    include_debug: bool = False
    #: Caps the planner. Defaults to ``Settings.menu_max_courses``.
    max_courses: int | None = None


class MenuRequestBody(BaseModel):
    query: str
    category: str = "recipes"
    subcategory: str | None = None
    filters: SearchFilters = SearchFilters()
    retrieval: MenuRetrievalOptions = MenuRetrievalOptions()
    menu: MenuOptions = MenuOptions()


# --- Response ---


class MenuBody(BaseModel):
    """The menu as a whole: its name, the coherence argument, and its citations."""

    title: str
    text: str
    citations: list[str]


class CourseSelection(BaseModel):
    """The one recipe chosen for a course. ``title`` is rebuilt from the pack."""

    knowledge_item_id: str
    title: str
    reason: str
    citation_ids: list[str]


class MenuCourse(BaseModel):
    """One course: what was planned, what was retrieved, and what was chosen.

    ``selection`` is ``null`` when the course could not be filled — retrieval
    returned nothing for its query, or the fallback path had no candidate to pick.
    A course is never dropped from the response for being unfillable: the caller
    asked for it, so it is reported empty rather than silently omitted.
    """

    slot: str
    query: str
    note: str = ""
    selection: CourseSelection | None = None
    candidates: list[KnowledgeItemResult] = []


class MenuDebugInfo(BaseModel):
    """Dev-only menu diagnostics, gated exactly like the search/answer debug blocks."""

    retrieval_mode: str
    model: str
    plan_prompt_version: str
    selection_prompt_version: str
    #: True when the planner fell back to one course carrying the raw query.
    plan_is_fallback: bool
    course_count: int
    candidate_count: int
    context_item_count: int
    citation_count: int
    #: Per-course retrieval diagnostics, keyed by slot.
    retrieval_debug_by_slot: dict[str, RetrievalDebugInfo] | None = None


class MenuResponse(BaseModel):
    query: str
    theme: str = ""
    menu: MenuBody
    courses: list[MenuCourse] = []
    citations: list[AnswerCitation] = []
    warnings: list[str] = []
    debug: MenuDebugInfo | None = None

    @model_serializer(mode="wrap")
    def _drop_absent_debug(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize with the ``debug`` key absent (not null) when gated off (D4).

        Every other legitimately-null field keeps appearing — this is NOT
        ``exclude_none``. Requires ``separate_input_output_schemas=False`` on
        the app or this serializer collapses the model's OpenAPI schema.
        """
        data: dict[str, Any] = handler(self)
        if data.get("debug") is None:
            data.pop("debug", None)
        return data
