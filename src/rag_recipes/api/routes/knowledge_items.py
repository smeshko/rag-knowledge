"""GET /api/v1/knowledge-items/{item_id} — the canonical item detail (doc 6 § 8).

Returns a KnowledgeItem with its FULL, untruncated ``structured_data`` (every
ingredient and step, plus any unknown keys, passed through verbatim), a small
doc-6 §8 ``display`` block, and item-level source citations. A direct audit-friendly
lookup: it returns the item regardless of status (ready / needs_review / superseded /
extracting) — 404 is reserved for a genuinely-unknown id. Parent document and source
spans are loaded by explicit queries (never via an async lazy relationship load).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_session
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.schemas.knowledge_items import (
    KnowledgeItemDetail,
    KnowledgeItemDisplay,
    KnowledgeItemResponse,
    KnowledgeItemSourceCitation,
)
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan

router = APIRouter(tags=["knowledge-items"])

_EN_DASH = "–"


def _pdf_page_label(locator: dict[str, Any]) -> str:
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


@router.get("/knowledge-items/{item_id}")
async def get_knowledge_item(
    item_id: str,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    item = await session.get(KnowledgeItem, item_id)
    if item is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
            message=f"Knowledge item {item_id!r} not found.",
            details={"item_id": item_id},
        )

    # Explicit loads — never the async-lazy `item.document` / span relationships.
    document = await session.get(Document, item.document_id)
    span_ids: list[str] = list(item.source_span_ids or [])
    spans_by_id: dict[str, SourceSpan] = {}
    if span_ids:
        rows = (
            await session.execute(
                select(SourceSpan).where(SourceSpan.id.in_(span_ids))
            )
        ).scalars().all()
        spans_by_id = {span.id: span for span in rows}

    citations = [
        KnowledgeItemSourceCitation(
            source_span_id=span_id,
            label=_pdf_page_label(spans_by_id[span_id].locator),
            locator=spans_by_id[span_id].locator,
        )
        for span_id in span_ids
        if span_id in spans_by_id
    ]

    primary_label = citations[0].label if citations else None
    doc_title = document.title if document is not None else ""
    subtitle = f"{doc_title} · {primary_label}" if primary_label else (doc_title or None)

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
        ),
        display=KnowledgeItemDisplay(title=item.title, subtitle=subtitle),
        source_citations=citations,
    )
