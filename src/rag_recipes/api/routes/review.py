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

import logging
from typing import Any

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_arq_redis, get_session
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.review_reasons import build_review_reasons
from rag_recipes.api.routes._params import parse_int
from rag_recipes.api.schemas.review import (
    ReviewDecision,
    ReviewedKnowledgeItem,
    ReviewItem,
    ReviewItemDocument,
    ReviewItemExtraction,
    ReviewItemListResponse,
    ReviewItemSourcePages,
    ReviewRequest,
    ReviewResponse,
)
from rag_recipes.api.search_projection import top_ingredients
from rag_recipes.ingestion.queue import enqueue_job
from rag_recipes.ingestion.status import TERMINAL_STATUSES
from rag_recipes.storage.enums import KnowledgeItemStatus
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan

router = APIRouter(tags=["review"])

logger = logging.getLogger(__name__)

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


@router.post("/knowledge-items/{item_id}/review", response_model=ReviewResponse)
async def review_knowledge_item(
    item_id: str,
    body: ReviewRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    arq_redis: ArqRedis = Depends(get_arq_redis),  # noqa: B008
) -> Any:
    """Decide a pending-review item (contract §2, plan D1/D6/D9).

    Approve flips ``needs_review → indexing`` and enqueues
    ``index_knowledge_item``; reject flips to terminal ``rejected``. The two
    409 guards run first (advisory read-then-act); the atomic guarded UPDATE
    is what actually closes the decide/decide race.

    The DB commit and the Redis enqueue are *not* one transaction — the
    enqueue-failure revert is best-effort compensation, not atomicity (D1).
    The reachable end states are: decided-and-enqueued; reverted to
    ``needs_review`` (compensating UPDATE, user re-approves); or a stuck
    ``indexing`` row that the item-level sweep returns to ``needs_review``.
    Either way indexing completes or the item returns to the queue.
    """
    target = (
        KnowledgeItemStatus.INDEXING
        if body.decision is ReviewDecision.APPROVED
        else KnowledgeItemStatus.REJECTED
    )

    # Resolve the item + its document first (explicit queries, never the
    # async-lazy item.document), so the 409 guards are reachable rather than
    # masked by the 404 path (D6 evaluation order).
    item_row = (
        await session.execute(
            select(KnowledgeItem.document_id, KnowledgeItem.source_version).where(
                KnowledgeItem.id == item_id
            )
        )
    ).one_or_none()
    if item_row is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
            message=f"Knowledge item {item_id!r} not found.",
            details={"item_id": item_id},
        )

    # Guard 1 (both decisions): a non-terminal document is mid-reprocess — the
    # pipeline is rewriting its items underneath the queue (mirrors the DELETE
    # /documents/{id} guard).
    doc_status = (
        await session.execute(
            select(Document.status).where(Document.id == item_row.document_id)
        )
    ).scalar_one()
    if doc_status not in TERMINAL_STATUSES:
        raise ApiError(
            status_code=409,
            code=ErrorCode.INGESTION_ALREADY_RUNNING,
            message="Document is not in a terminal state.",
            details={
                "document_id": item_row.document_id,
                "status": doc_status.value,
            },
        )

    # Guard 2 (approve only, D9): approving a stale generation would flip the
    # active version backwards and supersede the newer live generation.
    # Rejecting a stale item stays allowed — that is how the queue is cleared.
    if body.decision is ReviewDecision.APPROVED:
        current_version = (
            await session.execute(
                select(func.max(KnowledgeItem.source_version)).where(
                    KnowledgeItem.document_id == item_row.document_id
                )
            )
        ).scalar_one()
        if item_row.source_version < current_version:
            raise ApiError(
                status_code=409,
                code=ErrorCode.REVIEW_ITEM_STALE,
                message=(
                    "Knowledge item belongs to a stale extraction generation; "
                    "approve is refused (reject to clear it)."
                ),
                details={
                    "item_id": item_id,
                    "source_version": item_row.source_version,
                    "current_source_version": current_version,
                },
            )

    # Atomic guarded UPDATE (the reprocess_document race-closure pattern): two
    # concurrent decisions cannot both win — the loser matches 0 rows. One
    # result path: RETURNING read via scalar_one_or_none (never also rowcount).
    document_id = (
        await session.execute(
            update(KnowledgeItem)
            .where(
                KnowledgeItem.id == item_id,
                KnowledgeItem.status == KnowledgeItemStatus.NEEDS_REVIEW,
            )
            .values(status=target)
            .execution_options(synchronize_session=False)
            .returning(KnowledgeItem.document_id)
        )
    ).scalar_one_or_none()
    if document_id is None:
        # Explicit re-select, not session.get: the UPDATE ran with
        # synchronize_session=False and an identity-mapped object could still
        # report needs_review, putting a wrong status in the 404 body (D6).
        current_status = (
            await session.execute(
                select(KnowledgeItem.status).where(KnowledgeItem.id == item_id)
            )
        ).scalar_one_or_none()
        if current_status is None:
            raise ApiError(
                status_code=404,
                code=ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
                message=f"Knowledge item {item_id!r} not found.",
                details={"item_id": item_id},
            )
        raise ApiError(
            status_code=404,
            code=ErrorCode.REVIEW_NOT_PENDING,
            message=f"Knowledge item {item_id!r} is not awaiting review.",
            details={"item_id": item_id, "status": current_status.value},
        )

    await session.commit()

    if body.decision is ReviewDecision.APPROVED:
        try:
            await enqueue_job(
                arq_redis, "index_knowledge_item", item_id, session_id=document_id
            )
        except Exception:
            logger.exception(
                "Failed to enqueue index_knowledge_item for %s; reverting to "
                "needs_review",
                item_id,
            )
            # Best-effort compensation (D1): Redis may have accepted the job
            # before erroring, in which case this revert races a live worker —
            # safe, because the job's `status is INDEXING` guard no-ops on the
            # reverted row. If the revert itself fails, the item-level sweep
            # returns the row to needs_review.
            try:
                await session.execute(
                    update(KnowledgeItem)
                    .where(
                        KnowledgeItem.id == item_id,
                        KnowledgeItem.status == KnowledgeItemStatus.INDEXING,
                    )
                    .values(status=KnowledgeItemStatus.NEEDS_REVIEW)
                    .execution_options(synchronize_session=False)
                )
                await session.commit()
            except Exception:
                logger.exception(
                    "Compensating revert failed for %s; the stuck-indexing "
                    "sweep will return it to needs_review",
                    item_id,
                )
            raise ApiError(
                status_code=500,
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to enqueue the indexing job.",
            ) from None

    return ReviewResponse(
        knowledge_item=ReviewedKnowledgeItem(
            id=item_id, document_id=document_id, status=target.value
        ),
        decision=body.decision.value,
    )
