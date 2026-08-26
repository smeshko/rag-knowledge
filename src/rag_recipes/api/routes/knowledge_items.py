"""Knowledge-item lifecycle outside the review surface.

- ``POST /api/v1/knowledge-items`` — write a recipe by hand. The only creation
  path that does not run an extraction; it lands the item on the shared
  handwritten shelf and indexes it straight away.
- ``GET /api/v1/knowledge-items/{item_id}`` — the canonical item detail (doc 6
  § 8): the FULL, untruncated ``structured_data`` (every ingredient and step,
  plus any unknown keys, passed through verbatim), a small doc-6 §8 ``display``
  block, and item-level source citations. A direct audit-friendly lookup: it
  returns the item regardless of status (ready / needs_review / superseded /
  extracting) — 404 is reserved for a genuinely-unknown id.
- ``GET /api/v1/documents/{document_id}/knowledge-items`` — a book's contents,
  every status, which is the one thing ``GET /review-items`` cannot answer.
- ``DELETE /api/v1/knowledge-items/{item_id}`` — the per-recipe hard delete.

Epic 21.3's D4 recorded this module as read-only, with the review surface
owning every write. That boundary held while the only writes *were* review
verbs (decide, and 22.2's edit ``PATCH``, both of which share ``review.py``'s
guard stack, schemas and enqueue dependency). The per-recipe DELETE is not a
review verb — it is item lifecycle, reachable from the library rather than the
queue, and it shares the cascade repository with ``DELETE /documents/{id}``
rather than anything in ``review.py`` — so it lands here and D4 now reads
"review *decisions* live in review.py". The manual POST lands here on the same
rule: it is authoring, not reviewing, and its only overlap with the review
surface is the ``index_knowledge_item`` job both end up enqueuing.

Response assembly is shared, not re-implemented: ``api/knowledge_item_view``
for the detail envelope, ``api/knowledge_item_list`` for listing rows.
"""

from __future__ import annotations

import logging
from typing import Any

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, Response
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_arq_redis, get_session
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.knowledge_item_list import build_summaries
from rag_recipes.api.knowledge_item_view import build_knowledge_item_response
from rag_recipes.api.routes._params import parse_enum, parse_int
from rag_recipes.api.schemas.knowledge_items import (
    KnowledgeItemCreateRequest,
    KnowledgeItemResponse,
)
from rag_recipes.api.schemas.review import KnowledgeItemListResponse
from rag_recipes.ingestion.manual import authored_recipe, ensure_manual_shelf
from rag_recipes.ingestion.pipeline.persist import thresholds_from_settings
from rag_recipes.ingestion.queue import enqueue_job
from rag_recipes.ingestion.status import TERMINAL_STATUSES
from rag_recipes.storage.enums import KnowledgeItemStatus
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.repositories.documents import DocumentRepository

router = APIRouter(tags=["knowledge-items"])

logger = logging.getLogger(__name__)

_LIST_LIMIT_DEFAULT = 50
_LIST_LIMIT_MAX = 200
_LIST_OFFSET_DEFAULT = 0

#: Statuses hidden from an unfiltered book listing: ``superseded`` items belong
#: to a dead extraction generation and ``rejected`` ones were thrown away. Both
#: are reachable with an explicit ``?status=``.
_LIST_HIDDEN_STATUSES = (
    KnowledgeItemStatus.SUPERSEDED,
    KnowledgeItemStatus.REJECTED,
)


@router.post("/knowledge-items", status_code=201, response_model=KnowledgeItemResponse)
async def create_knowledge_item(
    body: KnowledgeItemCreateRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    arq_redis: ArqRedis = Depends(get_arq_redis),  # noqa: B008
) -> Any:
    """Write a recipe by hand, with no PDF and no extraction behind it.

    The one way a knowledge item enters this system without an ingest run. It
    lands on the shared **handwritten shelf** — a single Document that
    ``ensure_manual_shelf`` creates on first use and every manual recipe
    thereafter shares (see ``ingestion/manual`` for why the FKs demand a source
    chain at all, and why there is one rather than one per recipe).

    **Straight to the shelf, never to the queue.** The item is born ``indexing``
    and an ``index_knowledge_item`` job chunks, embeds and flips it to ``ready``
    — the approve path's second half, reached without the first, because there
    is no extraction to second-guess. Soft-validation warnings are still derived
    and stored (``authored_recipe``), they just do not gate the status: a typed
    recipe with no method carries ``no_steps`` for a later edit to clear,
    without being held back from search over it.

    Two consequences of that, both shared with approve and with editing a
    shelved item. The recipe is **absent from search until the worker
    finishes**, which is the price of not blocking the request on an embedding
    round-trip — the 201 says ``indexing`` and means it. And if the job exhausts
    its retries, ``sweep_stuck_jobs``' item-level pass returns the row to
    ``needs_review``, which is honest: the row genuinely has no chunks, and that
    is exactly what ``needs_review`` means.

    The DB commit and the Redis enqueue are not one transaction. On an enqueue
    failure the compensation **deletes the row** rather than parking it in the
    review queue the way approve and edit do — their compensation returns an
    item to a state it came from, while here there is no earlier state to return
    to. A 500 that also left a half-created recipe behind would make the obvious
    retry a duplicate. If the delete itself fails, the stuck-indexing sweep
    still rescues the row into ``needs_review``, so nothing is unrecoverable.
    """
    shelf = await ensure_manual_shelf(session)
    authored = authored_recipe(
        title=body.title,
        summary=body.summary,
        yield_=body.yield_,
        prep_time=body.prep_time,
        cook_time=body.cook_time,
        total_time=body.total_time,
        ingredients=body.ingredients,
        steps=body.steps,
        thresholds=thresholds_from_settings(),
    )
    item = KnowledgeItem(
        document_id=shelf.document_id,
        extraction_run_id=shelf.extraction_run_id,
        source_version=shelf.source_version,
        item_type="recipe",
        title=authored.title,
        normalized_title=authored.normalized_title,
        summary=authored.summary,
        body_text=authored.body_text,
        # Empty, not a fabricated span: no page was read. It is what makes the
        # detail envelope's `source_citations` empty and its subtitle the shelf
        # title alone.
        source_span_ids=[],
        structured_data=authored.structured_data,
        confidence=authored.confidence,
        # `pre_edit_snapshot` and `edited_at` stay NULL: they mean "the original
        # extraction, before a human touched it", and there was no extraction.
        # A first edit will fill them with THIS text, which is correct — it is
        # the original.
        status=KnowledgeItemStatus.INDEXING,
    )
    session.add(item)
    await session.flush()
    item_id = item.id
    await session.commit()

    try:
        await enqueue_job(
            arq_redis, "index_knowledge_item", item_id, session_id=shelf.document_id
        )
    except Exception:
        logger.exception(
            "Failed to enqueue index_knowledge_item for new item %s; deleting it",
            item_id,
        )
        try:
            # Guarded on INDEXING: Redis may have accepted the job before
            # erroring, so a worker could be running. It either has not started
            # (and then finds the item gone, hitting its `item is None` no-op)
            # or has already flipped the row to `ready` — in which case the
            # recipe is genuinely on the shelf and deleting it would throw away
            # a working save because the enqueue call *reported* a failure.
            await session.execute(
                delete(KnowledgeItem).where(
                    KnowledgeItem.id == item_id,
                    KnowledgeItem.status == KnowledgeItemStatus.INDEXING,
                )
            )
            await session.commit()
        except Exception:
            logger.exception(
                "Compensating delete failed for %s; the stuck-indexing sweep "
                "will return it to needs_review",
                item_id,
            )
        raise ApiError(
            status_code=500,
            code=ErrorCode.INTERNAL_ERROR,
            message="Failed to enqueue the indexing job.",
        ) from None

    logger.info(
        "Created manual knowledge item %s on the handwritten shelf %s",
        item_id,
        shelf.document_id,
    )
    # Re-read so the response reflects what Postgres holds — including a status
    # the worker may already have advanced to `ready` between the commit and
    # here, which is a true answer rather than a stale one.
    await session.refresh(item)
    return await build_knowledge_item_response(session, item)


@router.get("/knowledge-items/{item_id}", response_model=KnowledgeItemResponse)
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

    return await build_knowledge_item_response(session, item)


@router.get(
    "/documents/{document_id}/knowledge-items",
    response_model=KnowledgeItemListResponse,
)
async def list_document_knowledge_items(
    document_id: str,
    status: str | None = None,
    limit: str | None = None,
    offset: str | None = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    """List one book's knowledge items, newest first.

    The listing ``GET /review-items`` cannot be: it answers for *every* status,
    which is what makes a book browsable once its recipes are on the shelf.

    Unfiltered, it hides ``superseded`` and ``rejected`` rows — a dead
    generation and a thrown-away item are not part of the book's contents — but
    an explicit ``?status=`` reaches either, so nothing is unreachable. Unlike
    ``/review-items``, the document is *addressed* rather than filtered, so an
    unknown id is a 404 (the ``GET /documents/{id}`` rule), not an empty list.

    There is deliberately no document-status guard: unlike the review queue,
    which excludes mid-reprocess books because deciding their items is unsafe,
    reading a book's contents while it reprocesses is harmless. The write verbs
    keep their own 409.
    """
    status_filter = parse_enum(KnowledgeItemStatus, status, field="status")
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

    # Existence is checked before the listing so an unknown book reports as
    # unknown rather than as an empty one.
    document_title = (
        await session.execute(select(Document.title).where(Document.id == document_id))
    ).scalar_one_or_none()
    if document_title is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.DOCUMENT_NOT_FOUND,
            message=f"Document {document_id!r} not found.",
            details={"document_id": document_id},
        )

    stmt = select(KnowledgeItem, Document.id, Document.title).join(
        Document, KnowledgeItem.document_id == Document.id
    )
    stmt = stmt.where(KnowledgeItem.document_id == document_id)
    if status_filter is not None:
        stmt = stmt.where(KnowledgeItem.status == status_filter)
    else:
        stmt = stmt.where(KnowledgeItem.status.notin_(_LIST_HIDDEN_STATUSES))
    stmt = (
        stmt.order_by(KnowledgeItem.created_at.desc(), KnowledgeItem.id.desc())
        .limit(limit_int)
        .offset(offset_int)
    )
    rows = [tuple(row) for row in (await session.execute(stmt)).all()]

    return KnowledgeItemListResponse(knowledge_items=await build_summaries(session, rows))


@router.delete("/knowledge-items/{item_id}", status_code=204)
async def delete_knowledge_item(
    item_id: str,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Response:
    """Delete one recipe and everything derived from it.

    The per-item counterpart to ``DELETE /documents/{document_id}``, and the
    same shape: 404 unknown → 409 mid-reprocess → cascade → commit → log the
    per-table counts (a body-less 204 leaves no other audit trail).

    ``SELECT ... FOR UPDATE`` takes the **document** row, not the item — the
    lock order every writer here uses (``jobs.py``: document first, item
    second), and inverting it risks a deadlock against a concurrent reprocess.
    It also does the real work: it serializes this handler against
    ``index_knowledge_item``, so deleting an item that is mid-``indexing`` is
    safe without any new machinery — the job either has not started (and then
    finds the item gone, hitting its ``item is None`` no-op) or has finished.

    Allowed at any item status, ``ready`` included. That is the point: a
    shelved recipe had no removal path at all, and ``rejected`` — the soft
    delete the review surface offers — reaches only ``needs_review`` items.

    Not closed here: deleting the last ``ready`` item of the active generation
    leaves ``documents.active_source_version`` pointing at a generation with
    nothing in it. Not corrupt — search simply finds nothing and the counts
    read zero — and reprocessing is the existing way back.
    """
    document_id = (
        await session.execute(
            select(KnowledgeItem.document_id).where(KnowledgeItem.id == item_id)
        )
    ).scalar_one_or_none()
    if document_id is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
            message=f"Knowledge item {item_id!r} not found.",
            details={"item_id": item_id},
        )

    document = await session.get(Document, document_id, with_for_update=True)
    # documents.id is the item's FK target — the row cannot be missing.
    assert document is not None
    if document.status not in TERMINAL_STATUSES:
        raise ApiError(
            status_code=409,
            code=ErrorCode.INGESTION_ALREADY_RUNNING,
            message="Document is not in a terminal state.",
            details={"document_id": document_id, "status": document.status.value},
        )

    # Re-read under the document lock: a delete that raced another delete for
    # the same item would otherwise commit a no-op cascade and answer 204 for a
    # row it did not remove.
    still_present = (
        await session.execute(select(KnowledgeItem.id).where(KnowledgeItem.id == item_id))
    ).scalar_one_or_none()
    if still_present is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
            message=f"Knowledge item {item_id!r} not found.",
            details={"item_id": item_id},
        )

    counts = await DocumentRepository(session).delete_knowledge_item_cascade(item_id)
    await session.commit()
    logger.info(
        "Deleted knowledge item %s from document %s; per-table deleted rows: %s",
        item_id,
        document_id,
        counts,
    )
    return Response(status_code=204)
