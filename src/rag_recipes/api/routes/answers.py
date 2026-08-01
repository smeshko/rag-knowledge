"""POST /api/v1/answers — the query-time answer endpoint (doc 8).

A thin validate → orchestrate → project handler, mirroring ``search_documents``.
All synthesis logic (retrieval reuse, context pack, LLM call, citation validation,
safe fallback) lives in ``answers.service.generate_answer``; this handler validates
the request, normalizes a single positive effective limit, calls the service, and
projects its ``AnswerResult`` into the doc-8 § 6 envelope.

No ``LLMTechnicalError`` / 502 mapping here — the service turns *every* generation
failure (parse, invalid/misattributed citation, technical) into a safe-fallback
``AnswerResult``, so the route only raises request-validation 400s and inherits the
app-wide 401.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.answers.service import generate_answer
from rag_recipes.api.dependencies import (
    get_embedding_provider,
    get_llm_provider,
    get_session,
    get_settings,
)
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.schemas.answers import (
    AnswerDebugInfo,
    AnswerRequestBody,
    AnswerResponse,
)
from rag_recipes.config import Settings
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.retrieval.types import SearchRequest

router = APIRouter(tags=["answers"])

_VALID_MODES = frozenset({"hybrid", "keyword", "vector"})
# All four doc 8 § 9 styles share the same pipeline; only the prompt variant differs.
_SUPPORTED_STYLES = frozenset({"recommendation", "summary", "comparison", "direct_answer"})


@router.post("/answers", response_model=AnswerResponse)
async def answer(
    body: AnswerRequestBody,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),  # noqa: B008
    llm_provider: LLMProvider = Depends(get_llm_provider),  # noqa: B008
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
    if body.answer.style not in _SUPPORTED_STYLES:
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="Unsupported answer style.",
            details={"field": "style", "value": body.answer.style},
        )

    # One normalized positive limit drives both retrieval and the context-pack cap
    # (mirrors search()'s internal clamp); search_default_limit is Field(ge=1).
    limit = body.retrieval.limit
    effective_limit = limit if (limit is not None and limit > 0) else settings.search_default_limit

    request = SearchRequest(
        query=body.query,
        category=body.category,
        subcategory=body.subcategory,
        item_type=body.filters.item_type or "recipe",
        document_ids=list(body.filters.document_ids),
        mode=body.retrieval.mode,
        limit=effective_limit,
        exclude_needs_review=body.filters.exclude_needs_review,
    )

    result = await generate_answer(
        session,
        request,
        style=body.answer.style,
        include_results=body.answer.include_results,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        settings=settings,
    )

    # Dev-only answer debug, gated exactly like the search debug (doc 8 § 11): present
    # only when BOTH opted-in AND enabled; the key is dropped (absent, not null)
    # otherwise. The service always computes the fields; the route decides inclusion.
    debug = None
    if body.answer.include_debug and settings.debug_endpoints_enabled and result.debug is not None:
        d = result.debug
        debug = AnswerDebugInfo(
            retrieval_mode=d.retrieval_mode,
            model=d.model,
            prompt_version=d.prompt_version,
            context_item_count=d.context_item_count,
            citation_count=d.citation_count,
            retrieval_debug=d.retrieval_debug,
        )

    # The service already applies the success-path `include_results` drop and keeps
    # `results` on a fallback, so the route maps the AnswerResult straight through.
    # The absent-when-None `debug` key is handled by AnswerResponse's wrap
    # serializer (D4).
    return AnswerResponse(
        query=result.query,
        answer=result.answer,
        recommendations=result.recommendations,
        citations=result.citations,
        results=result.results,
        warnings=result.warnings,
        debug=debug,
    )
