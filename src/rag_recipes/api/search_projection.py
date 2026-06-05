"""Shared projection of a retrieval ``SearchResult`` into the API result envelope.

Both ``POST /api/v1/search`` and ``POST /api/v1/answers`` must return the *same*
``results[]`` for the same query — so the per-item projection (item / display /
structured_preview / document / matched_chunks / source_citations+locators /
confidence) and its batched DB fetches live here, imported by both routes, rather
than duplicated. Every Epic-12 attribute read is isolated in these helpers (one
adaptation seam).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.schemas.search import (
    DisplayProjection,
    KnowledgeItemResult,
    MatchedChunk,
    ResultDocument,
    ResultItem,
    RetrievalDebugInfo,
    SourceCitation,
    StructuredPreview,
)
from rag_recipes.config import Settings
from rag_recipes.retrieval.types import KnowledgeItemResult as FacadeItemResult
from rag_recipes.retrieval.types import SearchResult
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan

TOP_INGREDIENTS = 5
SNIPPET_MAX = 240


def build_retrieval_debug(result: SearchResult, settings: Settings) -> RetrievalDebugInfo:
    """Project the Epic-12 debug payload into the API ``RetrievalDebugInfo`` (doc 7 § 12).

    Shared by ``/search`` (its top-level ``debug``) and ``/answers`` (the nested
    ``retrieval_debug`` of its answer debug), so both report identical retrieval
    diagnostics for the same query.
    """
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


async def project_results(
    session: AsyncSession,
    result: SearchResult,
    *,
    structured: dict[str, dict[str, Any]] | None = None,
) -> list[KnowledgeItemResult]:
    """Project every item in ``result`` into the API ``KnowledgeItemResult`` envelope.

    ``structured`` (item id → ``{structured_data, confidence}``) may be supplied by
    a caller that already fetched it (the answer service fetches it to build the
    context-pack preview) to avoid a second round-trip; otherwise it is fetched here.
    """
    if structured is None:
        structured = await fetch_item_structured_data(session, result)
    locators = await fetch_citation_locators(session, result)
    return [
        build_result(item, structured.get(item.item.knowledge_item_id, {}), locators)
        for item in result.items
    ]


async def fetch_item_structured_data(
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


async def fetch_citation_locators(
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


def build_result(
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
        display=build_display(facade_item, structured_data, citations),
        structured_preview=build_structured_preview(structured_data),
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


def build_structured_preview(structured_data: dict[str, Any]) -> StructuredPreview:
    ingredients = structured_data.get("ingredients") or []
    top: list[str] = []
    for ing in ingredients:
        if not isinstance(ing, dict):
            continue
        label = ing.get("item_normalized") or ing.get("item_text") or ing.get("raw_text")
        if label:
            top.append(label)
        if len(top) >= TOP_INGREDIENTS:
            break
    return StructuredPreview.model_validate(
        {
            "schema": "recipe.preview.v1",
            "yield": structured_data.get("yield"),
            "top_ingredients": top,
        }
    )


def build_display(
    facade_item: FacadeItemResult,
    structured_data: dict[str, Any],
    citations: list[SourceCitation],
) -> DisplayProjection:
    primary_label = citations[0].label if citations else None
    doc_title = facade_item.document.title
    subtitle = f"{doc_title} · {primary_label}" if primary_label else (doc_title or None)
    summary = facade_item.item.summary
    snippet = summary[:SNIPPET_MAX] if summary else None
    return DisplayProjection(
        title=facade_item.item.title,
        subtitle=subtitle,
        snippet=snippet,
        badges=build_badges(structured_data),
    )


def build_badges(structured_data: dict[str, Any]) -> list[str]:
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
