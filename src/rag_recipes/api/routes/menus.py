"""POST /api/v1/menus — compose a multi-course menu from one request.

A thin validate → orchestrate → project handler, mirroring ``answer``. All menu
logic (decomposition, per-course retrieval, cross-course assignment, coherence
selection, citation validation, safe fallback) lives in
``menus.service.generate_menu``; this handler validates the request, normalizes the
positive effective caps, calls the service, and projects its ``MenuResult`` into the
``MenuResponse`` envelope.

No ``LLMTechnicalError`` / 502 mapping here — the service turns *every* planner and
selection failure into a safe-fallback ``MenuResult``, so the route raises only
request-validation 400s and inherits the app-wide 401.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import (
    get_embedding_provider,
    get_llm_provider,
    get_reranker_provider,
    get_session,
    get_settings,
)
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.schemas.menus import (
    MenuDebugInfo,
    MenuRequestBody,
    MenuResponse,
)
from rag_recipes.config import Settings
from rag_recipes.menus.service import generate_menu
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.reranker.base import RerankerProvider
from rag_recipes.retrieval.types import SearchRequest

router = APIRouter(tags=["menus"])

_VALID_MODES = frozenset({"hybrid", "keyword", "vector"})


@router.post("/menus", response_model=MenuResponse)
async def compose_menu(
    body: MenuRequestBody,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),  # noqa: B008
    llm_provider: LLMProvider = Depends(get_llm_provider),  # noqa: B008
    reranker: RerankerProvider | None = Depends(get_reranker_provider),  # noqa: B008
) -> Any:
    if not body.query.strip():
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="A non-empty 'query' is required.",
            details={"field": "query"},
        )
    if body.retrieval.mode not in _VALID_MODES:
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="Invalid value for 'mode'.",
            details={"field": "mode", "value": body.retrieval.mode},
        )

    # Both caps are clamped to a positive effective value the same way `search()`
    # clamps `limit`: a non-positive override falls back to the configured default
    # rather than producing an empty plan (0) or an unbounded fan-out (negative).
    max_courses = _positive(body.menu.max_courses, settings.menu_max_courses)
    candidates_per_course = _positive(
        body.retrieval.candidates_per_course, settings.menu_candidates_per_course
    )

    request = SearchRequest(
        query=body.query,
        category=body.category,
        subcategory=body.subcategory,
        item_type=body.filters.item_type or "recipe",
        document_ids=list(body.filters.document_ids),
        mode=body.retrieval.mode,
        # Per-course limits are set inside the service; this template value is
        # replaced for every course search and is never used as-is.
        limit=candidates_per_course,
        exclude_needs_review=body.filters.exclude_needs_review,
    )

    result = await generate_menu(
        session,
        request,
        include_candidates=body.menu.include_candidates,
        max_courses=max_courses,
        candidates_per_course=candidates_per_course,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        settings=settings,
        reranker=reranker,
    )

    # Dev-only menu debug, gated exactly like the search/answer debug blocks: present
    # only when BOTH opted-in AND enabled; the key is dropped (absent, not null)
    # otherwise. The service always computes the fields; the route decides inclusion.
    debug = None
    if body.menu.include_debug and settings.debug_endpoints_enabled and result.debug is not None:
        d = result.debug
        debug = MenuDebugInfo(
            retrieval_mode=d.retrieval_mode,
            model=d.model,
            plan_prompt_version=d.plan_prompt_version,
            selection_prompt_version=d.selection_prompt_version,
            plan_is_fallback=d.plan_is_fallback,
            course_count=d.course_count,
            candidate_count=d.candidate_count,
            context_item_count=d.context_item_count,
            citation_count=d.citation_count,
            retrieval_debug_by_slot=d.retrieval_debug_by_slot or None,
        )

    # The absent-when-None `debug` key is handled by MenuResponse's wrap serializer.
    return MenuResponse(
        query=result.query,
        theme=result.theme,
        menu=result.menu,
        courses=result.courses,
        citations=result.citations,
        warnings=result.warnings,
        debug=debug,
    )


def _positive(value: int | None, default: int) -> int:
    """Return ``value`` when it is a positive int, else ``default``."""
    return value if (value is not None and value > 0) else default
