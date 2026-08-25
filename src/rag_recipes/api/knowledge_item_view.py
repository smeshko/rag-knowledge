"""The canonical ``KnowledgeItemResponse`` assembly (doc 6 § 8).

Extracted from ``routes/knowledge_items.py`` in Epic 22.2 so the read endpoint
and the edit ``PATCH`` return byte-identical envelopes: the same citations, the
same display subtitle, the same freshly-derived ``review_reasons``. Written once
means an edit cannot answer with a shape the detail endpoint would not have
produced.

Parent document and source spans are loaded by explicit queries, never via an
async lazy relationship load.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.review_reasons import (
    build_review_reasons,
    build_review_thresholds,
)
from rag_recipes.api.schemas.knowledge_items import (
    KnowledgeItemDetail,
    KnowledgeItemDisplay,
    KnowledgeItemResponse,
    KnowledgeItemSourceCitation,
)
from rag_recipes.ingestion.pipeline.persist import thresholds_from_settings
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.knowledge_item_favourite import KnowledgeItemFavourite
from rag_recipes.storage.models.source_span import SourceSpan

__all__ = ["build_knowledge_item_response", "pdf_page_label"]

_EN_DASH = "–"


def pdf_page_label(locator: dict[str, Any]) -> str:
    """Render a ``pdf_page_range`` locator as ``"page 42"`` / ``"pages 42–43"``.

    Grounded in doc 7 § 10 and byte-equivalent to the Epic 12 search label
    (``page_start`` / ``page_end`` from the Epic 8 span writer; en dash U+2013).
    """
    start = locator.get("page_start")
    end = locator.get("page_end")
    if start is None:
        return ""
    if end is None or end == start:
        return f"page {start}"
    return f"pages {start}{_EN_DASH}{end}"


async def build_knowledge_item_response(
    session: AsyncSession, item: KnowledgeItem
) -> KnowledgeItemResponse:
    """Assemble the doc-6 §8 response envelope for ``item``.

    ``review_reasons`` are derived from the item's *current*
    ``structured_data["warnings"]``, so a caller that has just rewritten them
    gets the recomputed flag set for free.
    """
    document = await session.get(Document, item.document_id)
    # A scalar read rather than a relationship: the star lives in its own table
    # (see the model docstring) and the PATCH path calls this builder inside a
    # transaction that has just rewritten the item, so a lazy load here would be
    # an async lazy load on a dirty session.
    favourited_at = (
        await session.execute(
            select(KnowledgeItemFavourite.created_at).where(
                KnowledgeItemFavourite.knowledge_item_id == item.id
            )
        )
    ).scalar_one_or_none()
    span_ids: list[str] = list(item.source_span_ids or [])
    spans_by_id: dict[str, SourceSpan] = {}
    if span_ids:
        rows = (
            (await session.execute(select(SourceSpan).where(SourceSpan.id.in_(span_ids))))
            .scalars()
            .all()
        )
        spans_by_id = {span.id: span for span in rows}

    citations = [
        KnowledgeItemSourceCitation(
            source_span_id=span_id,
            label=pdf_page_label(spans_by_id[span_id].locator),
            locator=spans_by_id[span_id].locator,
        )
        for span_id in span_ids
        if span_id in spans_by_id
    ]

    primary_label = citations[0].label if citations else None
    doc_title = document.title if document is not None else ""
    subtitle = f"{doc_title} · {primary_label}" if primary_label else (doc_title or None)

    # Current bounds, not the ones ingest used (those are not persisted) — the
    # reasons say so on the wire via `threshold`.
    thresholds = thresholds_from_settings()
    return KnowledgeItemResponse(
        knowledge_item=KnowledgeItemDetail(
            id=item.id,
            document_id=item.document_id,
            item_type=item.item_type,
            title=item.title,
            summary=item.summary,
            status=item.status.value,
            source_span_ids=span_ids,
            confidence=item.confidence,
            structured_data=item.structured_data or {},
            review_reasons=build_review_reasons(
                item.status.value,
                item.structured_data or {},
                confidence=item.confidence,
                thresholds=thresholds,
            ),
            review_thresholds=build_review_thresholds(item.status.value, thresholds),
            edited_at=item.edited_at,
            favourited_at=favourited_at,
        ),
        display=KnowledgeItemDisplay(title=item.title, subtitle=subtitle),
        source_citations=citations,
    )
