"""POST /api/v1/search — the public search endpoint (doc 6 § 7).

A thin validate → call → project pipeline over the Epic 12 retrieval facade
(``retrieval.search.search``). All retrieval logic — normalization, metadata
filters, keyword/vector SQL, RRF merge, chunk-type boosts, item grouping, citation
labels — lives in Epic 12; this handler only validates the request, calls the
facade, and projects its ``SearchResult`` into the doc-6 response envelope. Every
Epic-12 attribute read is isolated in the private ``_build_*`` helpers (one
adaptation seam). The ``debug`` block is gated on BOTH ``include_debug`` AND
``Settings.debug_endpoints_enabled``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import (
    get_embedding_provider,
    get_session,
    get_settings,
)
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.schemas.search import (
    DisplayProjection,
    KnowledgeItemResult,
    MatchedChunk,
    ResultDocument,
    ResultItem,
    RetrievalDebugInfo,
    SearchRequestBody,
    SearchResponse,
    SourceCitation,
    StructuredPreview,
)
from rag_recipes.config import Settings
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.retrieval.search import search
from rag_recipes.retrieval.types import KnowledgeItemResult as FacadeItemResult
from rag_recipes.retrieval.types import SearchRequest, SearchResult
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan

router = APIRouter(tags=["search"])

_VALID_MODES = frozenset({"hybrid", "keyword", "vector"})
_TOP_INGREDIENTS = 5
_SNIPPET_MAX = 240


@router.post("/search")
async def search_documents(
    body: SearchRequestBody,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
    provider: EmbeddingProvider = Depends(get_embedding_provider),  # noqa: B008
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
    result = await search(session, request, provider=provider, settings=settings)

    structured = await _fetch_item_structured_data(session, result)
    locators = await _fetch_citation_locators(session, result)

    response = SearchResponse(
        query=result.debug.normalized_query,
        results=[
            _build_result(item, structured.get(item.item.knowledge_item_id, {}), locators)
            for item in result.items
        ],
        debug=_build_debug(result, settings)
        if (body.include_debug and settings.debug_endpoints_enabled)
        else None,
    )

    # Serialize with the schema/yield aliases; drop `debug` entirely when gated off
    # (the key is absent, not null), keeping other legitimately-null fields.
    data = response.model_dump(by_alias=True)
    if data.get("debug") is None:
        data.pop("debug", None)
    return data


async def _fetch_item_structured_data(
    session: AsyncSession, result: SearchResult
) -> dict[str, dict[str, Any]]:
    """Batch-fetch ``structured_data`` + ``confidence`` for the result items.

    The facade's item projection is minimal; the API envelope needs the recipe
    ``structured_data`` (for the preview/badges) and ``confidence``. Returns a map
    of item id → ``{"structured_data": ..., "confidence": ...}``.
    """
    item_ids = [item.item.knowledge_item_id for item in result.items]
    if not item_ids:
        return {}
    rows = (
        await session.execute(
            select(
                KnowledgeItem.id,
                KnowledgeItem.structured_data,
                KnowledgeItem.confidence,
            ).where(KnowledgeItem.id.in_(item_ids))
        )
    ).all()
    return {
        row.id: {"structured_data": row.structured_data or {}, "confidence": row.confidence}
        for row in rows
    }


async def _fetch_citation_locators(
    session: AsyncSession, result: SearchResult
) -> dict[str, dict[str, Any]]:
    """Batch-fetch the ``locator`` JSONB for every cited source span."""
    span_ids = {
        cite.source_span_id for item in result.items for cite in item.source_citations
    }
    if not span_ids:
        return {}
    rows = (
        await session.execute(
            select(SourceSpan.id, SourceSpan.locator).where(SourceSpan.id.in_(span_ids))
        )
    ).all()
    return {row.id: row.locator for row in rows}


def _build_result(
    facade_item: FacadeItemResult,
    extra: dict[str, Any],
    locators: dict[str, dict[str, Any]],
) -> KnowledgeItemResult:
    structured_data: dict[str, Any] = extra.get("structured_data", {})
    confidence = extra.get("confidence")
    citations = [
        SourceCitation(
            source_span_id=cite.source_span_id,
            label=cite.label,
            locator=locators.get(cite.source_span_id),
        )
        for cite in facade_item.source_citations
    ]
    return KnowledgeItemResult(
        # model_validate (alias-keyed) — the `schema`/`yield` fields can't be passed
        # by keyword (`yield` is a Python keyword) and mypy infers the init by alias.
        item=ResultItem.model_validate(
            {
                "id": facade_item.item.knowledge_item_id,
                "item_type": facade_item.item.item_type,
                "schema": structured_data.get("schema", "recipe.v1"),
                "title": facade_item.item.title,
                "summary": facade_item.item.summary,
                "status": facade_item.item.status,
                "confidence": confidence,
            }
        ),
        display=_build_display(facade_item, structured_data, citations),
        structured_preview=_build_structured_preview(structured_data),
        document=ResultDocument(
            id=facade_item.document.document_id,
            title=facade_item.document.title,
            author=facade_item.document.author,
        ),
        matched_chunks=[
            MatchedChunk(
                chunk_id=mc.chunk_id, chunk_type=mc.chunk_type.value, score=mc.score
            )
            for mc in facade_item.matched_chunks
        ],
        source_citations=citations,
    )


def _build_structured_preview(structured_data: dict[str, Any]) -> StructuredPreview:
    ingredients = structured_data.get("ingredients") or []
    top: list[str] = []
    for ing in ingredients:
        if not isinstance(ing, dict):
            continue
        label = ing.get("item_normalized") or ing.get("item_text") or ing.get("raw_text")
        if label:
            top.append(label)
        if len(top) >= _TOP_INGREDIENTS:
            break
    return StructuredPreview.model_validate(
        {
            "schema": "recipe.preview.v1",
            "yield": structured_data.get("yield"),
            "top_ingredients": top,
        }
    )


def _build_display(
    facade_item: FacadeItemResult,
    structured_data: dict[str, Any],
    citations: list[SourceCitation],
) -> DisplayProjection:
    primary_label = citations[0].label if citations else None
    doc_title = facade_item.document.title
    subtitle = f"{doc_title} · {primary_label}" if primary_label else (doc_title or None)
    summary = facade_item.item.summary
    snippet = summary[:_SNIPPET_MAX] if summary else None
    return DisplayProjection(
        title=facade_item.item.title,
        subtitle=subtitle,
        snippet=snippet,
        badges=_build_badges(structured_data),
    )


def _build_badges(structured_data: dict[str, Any]) -> list[str]:
    """Derive display badges from structured_data (doc 6 § 7; DECISIONS #2).

    A non-empty ``yield`` contributes one badge; the first present of
    ``cook_time`` / ``total_time`` / ``prep_time`` contributes one time badge.
    Absent/empty fields contribute nothing.
    """
    badges: list[str] = []
    recipe_yield = structured_data.get("yield")
    if recipe_yield:
        badges.append(str(recipe_yield))
    for key in ("cook_time", "total_time", "prep_time"):
        value = structured_data.get(key)
        if value:
            badges.append(str(value))
            break
    return badges


def _build_debug(result: SearchResult, settings: Settings) -> RetrievalDebugInfo:
    debug = result.debug
    return RetrievalDebugInfo(
        retrieval_mode=debug.mode,
        normalized_query=debug.normalized_query,
        embedding_model=settings.embedding_model,
        keyword_top_k=settings.search_keyword_top_k,
        vector_top_k=settings.search_vector_top_k,
        keyword_candidates=debug.keyword_candidates,
        vector_candidates=debug.vector_candidates,
        merged_candidates=debug.merged_chunks,
        grouped_items=debug.grouped_items,
    )
