"""Review-queue surface (Epic 21.3): the global review queue's read side.

``GET /review-items`` (contract §1, plan D4): cross-document listing of
``needs_review`` knowledge items on *terminal* documents — mid-reprocess
documents' items are excluded by the ``TERMINAL_STATUSES`` guard alone. There
is deliberately **no version/generation scoping** (D9): staleness is a
decision-time concern (the POST's 409), never a listing filter, so the queue,
``count_knowledge_items`` and the documents-list derivation share one
identical ``needs_review`` domain.

Module boundary (D4): the review surface groups its read and write endpoints
here (shared schemas + enqueue dependency) rather than splitting by URL
prefix; ``knowledge_items.py`` stays the read-only audit endpoint.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_session
from rag_recipes.api.review_reasons import build_review_reasons
from rag_recipes.api.routes._params import parse_int
from rag_recipes.api.schemas.review import (
    ReviewItem,
    ReviewItemDocument,
    ReviewItemExtraction,
    ReviewItemListResponse,
    ReviewItemSourcePages,
)
from rag_recipes.api.search_projection import top_ingredients
from rag_recipes.ingestion.status import TERMINAL_STATUSES
from rag_recipes.storage.enums import KnowledgeItemStatus
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan

router = APIRouter(tags=["review"])

_LIST_LIMIT_DEFAULT = 50
_LIST_LIMIT_MAX = 200
_LIST_OFFSET_DEFAULT = 0


def _source_pages(
    span_ids: list[str], locators_by_id: dict[str, dict[str, Any]]
) -> ReviewItemSourcePages:
    """Min/max page bounds over the item's resolved span locators.

    Keys are read with ``.get`` and skipped when absent (the
    ``_pdf_page_label`` precedent) — a degraded locator must never 500 the
    whole listing. Both bounds are ``None`` when nothing resolves.
    """
    starts: list[int] = []
    ends: list[int] = []
    for span_id in span_ids:
        locator = locators_by_id.get(span_id)
        if not isinstance(locator, dict):
            continue
        start = locator.get("page_start")
        end = locator.get("page_end")
        if isinstance(start, int):
            starts.append(start)
        if isinstance(end, int):
            ends.append(end)
    return ReviewItemSourcePages(
        page_start=min(starts) if starts else None,
        page_end=max(ends) if ends else None,
    )


@router.get("/review-items", response_model=ReviewItemListResponse)
async def list_review_items(
    document_id: str | None = None,
    limit: str | None = None,
    offset: str | None = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    """List pending-review knowledge items, newest first (contract §1, D4).

    Every ``needs_review`` item of a terminal document — including a
    ``failed`` one (D9 recorded consequence) and every live generation of a
    twice-reviewed document (D9: staleness is handled at decision time, not by
    hiding rows here). Unknown ``document_id`` → naturally 200 + empty list.
    """
    limit_int = parse_int(
        limit,
        field="limit",
        default=_LIST_LIMIT_DEFAULT,
        minimum=1,
        maximum=_LIST_LIMIT_MAX,
    )
    offset_int = parse_int(
        offset,
        field="offset",
        default=_LIST_OFFSET_DEFAULT,
        minimum=0,
    )

    stmt = (
        select(KnowledgeItem, Document.id, Document.title)
        .join(Document, KnowledgeItem.document_id == Document.id)
        .where(
            KnowledgeItem.status == KnowledgeItemStatus.NEEDS_REVIEW,
            Document.status.in_(TERMINAL_STATUSES),
        )
    )
    if document_id:
        stmt = stmt.where(Document.id == document_id)
    stmt = (
        stmt.order_by(KnowledgeItem.created_at.desc(), KnowledgeItem.id.desc())
        .limit(limit_int)
        .offset(offset_int)
    )
    rows = (await session.execute(stmt)).all()

    # Chunk-free source_pages path (D4): needs_review items have no chunks, so
    # spans are resolved off KnowledgeItem.source_span_ids itself — one batched
    # fetch for the whole page.
    page_span_ids = {
        span_id for item, _, _ in rows for span_id in (item.source_span_ids or [])
    }
    locators_by_id: dict[str, dict[str, Any]] = {}
    if page_span_ids:
        span_rows = (
            await session.execute(
                select(SourceSpan.id, SourceSpan.locator).where(
                    SourceSpan.id.in_(page_span_ids)
                )
            )
        ).all()
        locators_by_id = {row.id: row.locator for row in span_rows}

    review_items: list[ReviewItem] = []
    for item, doc_id, doc_title in rows:
        structured = item.structured_data or {}
        review_items.append(
            ReviewItem(
                id=item.id,
                title=item.title,
                summary=item.summary,
                item_type=item.item_type,
                document=ReviewItemDocument(id=doc_id, title=doc_title),
                source_pages=_source_pages(
                    list(item.source_span_ids or []), locators_by_id
                ),
                # model_validate (alias-keyed): `schema`/`yield` cannot be
                # passed by keyword (`yield` is a Python keyword).
                extraction=ReviewItemExtraction.model_validate(
                    {
                        "schema": structured.get("schema", "recipe.v1"),
                        "yield": structured.get("yield"),
                        "top_ingredients": top_ingredients(structured),
                        "confidence_overall": (item.confidence or {}).get("overall"),
                    }
                ),
                flags=build_review_reasons(item.status.value, structured),
            )
        )
    return ReviewItemListResponse(review_items=review_items)
