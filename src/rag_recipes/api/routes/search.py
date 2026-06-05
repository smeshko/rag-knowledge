"""POST /api/v1/search — the public search endpoint (doc 6 § 7).

A thin validate → call → project pipeline over the Epic 12 retrieval facade
(``retrieval.search.search``). All retrieval logic — normalization, metadata
filters, keyword/vector SQL, RRF merge, chunk-type boosts, item grouping, citation
labels — lives in Epic 12; this handler only validates the request, calls the
facade, and projects its ``SearchResult`` into the doc-6 response envelope. The
per-item projection lives in ``api/search_projection`` (shared with the answers
route so both return byte-identical ``results``). The ``debug`` block is gated on
BOTH ``include_debug`` AND ``Settings.debug_endpoints_enabled``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import (
    get_embedding_provider,
    get_reranker_provider,
    get_session,
    get_settings,
)
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.schemas.search import (
    SearchRequestBody,
    SearchResponse,
)
from rag_recipes.api.search_projection import build_retrieval_debug, project_results
from rag_recipes.config import Settings
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.reranker.base import RerankerProvider
from rag_recipes.retrieval.search import search
from rag_recipes.retrieval.types import SearchRequest

router = APIRouter(tags=["search"])

_VALID_MODES = frozenset({"hybrid", "keyword", "vector"})


@router.post("/search")
async def search_documents(
    body: SearchRequestBody,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
    provider: EmbeddingProvider = Depends(get_embedding_provider),  # noqa: B008
    reranker: RerankerProvider | None = Depends(get_reranker_provider),  # noqa: B008
) -> Any:
    if not body.query.strip():
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="A non-empty 'query' is required.",
            details={"field": "query"},
        )
    if body.mode not in _VALID_MODES:
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="Invalid value for 'mode'.",
            details={"field": "mode", "value": body.mode},
        )

    request = SearchRequest(
        query=body.query,
        category=body.category,
        subcategory=body.subcategory,
        item_type=body.filters.item_type or "recipe",
        document_ids=list(body.filters.document_ids),
        mode=body.mode,
        limit=body.limit if body.limit is not None else settings.search_default_limit,
        exclude_needs_review=body.filters.exclude_needs_review,
    )
    result = await search(
        session, request, provider=provider, settings=settings, reranker=reranker
    )

    response = SearchResponse(
        query=result.debug.normalized_query,
        results=await project_results(session, result),
        debug=build_retrieval_debug(result, settings)
        if (body.include_debug and settings.debug_endpoints_enabled)
        else None,
    )

    # Serialize with the schema/yield aliases; drop `debug` entirely when gated off
    # (the key is absent, not null), keeping other legitimately-null fields.
    data = response.model_dump(by_alias=True)
    if data.get("debug") is None:
        data.pop("debug", None)
    return data
