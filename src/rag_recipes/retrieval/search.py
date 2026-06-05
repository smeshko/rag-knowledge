"""The retrieval facade — ``search()`` composes the whole layer (doc 7 §§ 6-11).

Normalizes the query, resolves filters, branches on ``mode`` (hybrid/keyword/vector),
fuses the legs via RRF + chunk-type boosts, groups by KnowledgeItem, fetches the
items/documents/chunks/source-spans for the top results (batched, no N+1), and builds
the result envelope with matched-chunk scores and page-citation labels. Returns a
stdlib ``SearchResult`` dataclass; Epic 13 maps it onto the HTTP response model and
owns the dev-only gating of the debug payload (DECISIONS #4).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.config import Settings
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.retrieval.filters import build_filters
from rag_recipes.retrieval.group import group_by_item
from rag_recipes.retrieval.keyword import keyword_search
from rag_recipes.retrieval.merge import merge_candidates
from rag_recipes.retrieval.normalize import normalize_query
from rag_recipes.retrieval.types import (
    ChunkCandidate,
    ItemResult,
    KnowledgeItemResult,
    MatchedChunkRef,
    ResultDocument,
    ResultItem,
    SearchDebug,
    SearchRequest,
    SearchResult,
    SourceCitation,
)
from rag_recipes.retrieval.vector import vector_search
from rag_recipes.storage.enums import ChunkType
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan

_EN_DASH = "–"


def _keyword_boost_table(settings: Settings) -> Mapping[ChunkType, float]:
    return {
        ChunkType.RECIPE_TITLE: settings.recipe_keyword_boost_title,
        ChunkType.RECIPE_INGREDIENTS: settings.recipe_keyword_boost_ingredients,
        ChunkType.RECIPE_STEPS: settings.recipe_keyword_boost_steps,
        ChunkType.RECIPE_SUMMARY: settings.recipe_keyword_boost_summary,
        ChunkType.RECIPE_FULL: settings.recipe_keyword_boost_full,
    }


def _vector_boost_table(settings: Settings) -> Mapping[ChunkType, float]:
    return {
        ChunkType.RECIPE_SUMMARY: settings.recipe_vector_boost_summary,
        ChunkType.RECIPE_FULL: settings.recipe_vector_boost_full,
        ChunkType.RECIPE_STEPS: settings.recipe_vector_boost_steps,
        ChunkType.RECIPE_INGREDIENTS: settings.recipe_vector_boost_ingredients,
        ChunkType.RECIPE_TITLE: settings.recipe_vector_boost_title,
    }


def _citation_label(locator: dict[str, Any]) -> str:
    """Render a SourceSpan locator as ``"page 42"`` or ``"pages 42–43"`` (en dash).

    The span writer (Epic 8) stamps ``page_start`` / ``page_end`` into the JSONB
    locator. A single-page span (start == end, or no end) reads "page N"; a range
    reads "pages N–M".
    """
    start = locator.get("page_start")
    end = locator.get("page_end")
    if start is None:
        return ""
    if end is None or end == start:
        return f"page {start}"
    return f"pages {start}{_EN_DASH}{end}"


async def search(
    session: AsyncSession,
    request: SearchRequest,
    *,
    provider: EmbeddingProvider,
    settings: Settings,
) -> SearchResult:
    """Run the full retrieval pipeline and return the grouped, fetched result envelope."""
    nq = normalize_query(request.query)
    filters = build_filters(request)
    limit = request.limit or settings.search_default_limit
    keyword_top_k = max(limit * 5, settings.search_keyword_top_k)
    vector_top_k = max(limit * 5, settings.search_vector_top_k)

    keyword_candidates: list[ChunkCandidate] = []
    vector_candidates: list[ChunkCandidate] = []
    if request.mode in ("keyword", "hybrid"):
        keyword_candidates = await keyword_search(session, nq, filters, top_k=keyword_top_k)
    if request.mode in ("vector", "hybrid"):
        vector_candidates = await vector_search(
            session,
            nq,
            filters,
            provider=provider,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            top_k=vector_top_k,
        )

    # All three modes fuse through the same boost-application + grouping (a single
    # leg just passes an empty list for the absent side), so item scores are
    # consistent across modes (doc 7 § 6).
    merged = merge_candidates(
        keyword_candidates,
        vector_candidates,
        keyword_boosts=_keyword_boost_table(settings),
        vector_boosts=_vector_boost_table(settings),
        rrf_k=settings.search_rrf_k,
        keyword_source_weight=settings.keyword_source_weight,
        vector_source_weight=settings.vector_source_weight,
    )
    grouped = group_by_item(
        merged,
        supporting_bonus=settings.search_supporting_chunk_bonus,
        supporting_bonus_cap=settings.search_supporting_chunk_bonus_cap,
    )
    top_items = grouped[:limit]

    items = await _fetch_and_build(session, top_items)
    return SearchResult(
        items=items,
        debug=SearchDebug(
            mode=request.mode,
            normalized_query=nq.keyword,
            keyword_candidates=len(keyword_candidates),
            vector_candidates=len(vector_candidates),
            merged_chunks=len(merged),
            grouped_items=len(grouped),
        ),
    )


async def _fetch_and_build(
    session: AsyncSession, top_items: list[ItemResult]
) -> list[KnowledgeItemResult]:
    """Batch-fetch the items/documents/chunks/spans for ``top_items`` (no N+1)."""
    if not top_items:
        return []

    item_ids = [r.knowledge_item_id for r in top_items]
    chunk_ids = [m.chunk_id for r in top_items for m in r.matched_chunks]

    items_by_id = {
        ki.id: ki
        for ki in (
            await session.execute(
                select(KnowledgeItem).where(KnowledgeItem.id.in_(item_ids))
            )
        )
        .scalars()
        .all()
    }
    doc_ids = {ki.document_id for ki in items_by_id.values()}
    docs_by_id = {
        d.id: d
        for d in (
            await session.execute(select(Document).where(Document.id.in_(doc_ids)))
        )
        .scalars()
        .all()
    }
    chunks_by_id = {
        c.id: c
        for c in (
            await session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
        )
        .scalars()
        .all()
    }
    span_ids = {
        sid for c in chunks_by_id.values() for sid in c.source_span_ids
    }
    spans_by_id = {
        s.id: s
        for s in (
            await session.execute(
                select(SourceSpan).where(SourceSpan.id.in_(span_ids))
            )
        )
        .scalars()
        .all()
    }

    results: list[KnowledgeItemResult] = []
    for grouped_item in top_items:
        ki = items_by_id.get(grouped_item.knowledge_item_id)
        if ki is None:
            continue  # defensive: a deleted item between ranking and fetch
        doc = docs_by_id.get(ki.document_id)
        results.append(
            KnowledgeItemResult(
                item=ResultItem(
                    knowledge_item_id=ki.id,
                    item_type=ki.item_type,
                    title=ki.title,
                    summary=ki.summary,
                    status=ki.status.value,
                ),
                document=ResultDocument(
                    document_id=doc.id if doc else ki.document_id,
                    title=doc.title if doc else "",
                    author=doc.author if doc else "",
                ),
                item_score=grouped_item.item_score,
                matched_chunks=[
                    MatchedChunkRef(
                        chunk_id=m.chunk_id, chunk_type=m.chunk_type, score=m.score
                    )
                    for m in grouped_item.matched_chunks
                ],
                source_citations=_citations_for(
                    grouped_item, chunks_by_id, spans_by_id
                ),
            )
        )
    return results


def _citations_for(
    grouped_item: ItemResult,
    chunks_by_id: dict[str, Chunk],
    spans_by_id: dict[str, SourceSpan],
) -> list[SourceCitation]:
    """Distinct page citations for an item's matched chunks, in first-seen order."""
    citations: list[SourceCitation] = []
    seen: set[str] = set()
    for matched in grouped_item.matched_chunks:
        chunk = chunks_by_id.get(matched.chunk_id)
        if chunk is None:
            continue
        for span_id in chunk.source_span_ids:
            if span_id in seen:
                continue
            span = spans_by_id.get(span_id)
            if span is None:
                continue
            seen.add(span_id)
            citations.append(
                SourceCitation(
                    source_span_id=span_id, label=_citation_label(span.locator)
                )
            )
    return citations
