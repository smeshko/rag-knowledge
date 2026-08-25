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
prefix; ``knowledge_items.py`` stays the read-only audit endpoint. Epic 22.2's
edit ``PATCH`` lands here for the same reason — it is a review-surface write
sharing this module's guard stack and transaction shape.
"""

from __future__ import annotations

import logging
from typing import Any

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends
from sqlalchemy import case, func, literal, select, type_coerce, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_arq_redis, get_session
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.knowledge_item_list import build_summaries
from rag_recipes.api.knowledge_item_view import build_knowledge_item_response
from rag_recipes.api.routes._params import parse_int
from rag_recipes.api.schemas.knowledge_items import (
    KnowledgeItemResponse,
    KnowledgeItemUpdateRequest,
)
from rag_recipes.api.schemas.review import (
    ReviewDecision,
    ReviewedKnowledgeItem,
    ReviewItemListResponse,
    ReviewRequest,
    ReviewResponse,
)
from rag_recipes.ingestion.editing import (
    UNSET,
    RecipeEdit,
    Unset,
    apply_edit,
    warnings_for_item,
)
from rag_recipes.ingestion.pipeline.persist import thresholds_from_settings
from rag_recipes.ingestion.queue import enqueue_job
from rag_recipes.ingestion.status import TERMINAL_STATUSES
from rag_recipes.storage.enums import KnowledgeItemStatus
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.repositories.documents import DocumentRepository

router = APIRouter(tags=["review"])

logger = logging.getLogger(__name__)

_LIST_LIMIT_DEFAULT = 50
_LIST_LIMIT_MAX = 200
_LIST_OFFSET_DEFAULT = 0

#: The two statuses ``PATCH /knowledge-items/{id}`` accepts. ``needs_review``
#: is a pure row rewrite; ``ready`` additionally drops the item's index rows and
#: re-indexes it. Everything else — ``indexing``, ``extracting``,
#: ``superseded``, ``rejected`` — is either mid-flight or dead.
_EDITABLE_STATUSES = (
    KnowledgeItemStatus.NEEDS_REVIEW,
    KnowledgeItemStatus.READY,
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
    rows = [tuple(row) for row in (await session.execute(stmt)).all()]

    # Chunk-free source_pages path (D4), now shared with the per-book listing:
    # spans are resolved off KnowledgeItem.source_span_ids itself, one batched
    # fetch for the whole page.
    return ReviewItemListResponse(review_items=await build_summaries(session, rows))


def _recipe_edit_from(body: KnowledgeItemUpdateRequest) -> RecipeEdit:
    """Project the request onto the domain edit, preserving absent-vs-null.

    ``model_fields_set`` is the ``exclude_unset`` semantics the contract needs:
    a field the client never mentioned stays ``UNSET`` and is left alone, while
    one sent as ``null`` is a real instruction to clear it.

    Written out field by field rather than through a ``getattr`` helper so the
    types survive: a helper returning ``Any`` would let a ``None`` reach a field
    the domain object types as non-optional without mypy noticing.
    """
    supplied = body.model_fields_set

    def clearable(name: str, value: str | None) -> str | None | Unset:
        return value if name in supplied else UNSET

    # The schema rejects an explicit null on these three, so "supplied" implies
    # a real value.
    title: str | Unset = body.title if "title" in supplied and body.title else UNSET
    ingredients: list[str] | Unset = (
        body.ingredients if "ingredients" in supplied and body.ingredients is not None else UNSET
    )
    steps: list[str] | Unset = (
        body.steps if "steps" in supplied and body.steps is not None else UNSET
    )

    return RecipeEdit(
        title=title,
        summary=clearable("summary", body.summary),
        yield_=clearable("yield_", body.yield_),
        prep_time=clearable("prep_time", body.prep_time),
        cook_time=clearable("cook_time", body.cook_time),
        total_time=clearable("total_time", body.total_time),
        ingredients=ingredients,
        steps=steps,
    )


@router.patch("/knowledge-items/{item_id}", response_model=KnowledgeItemResponse)
async def update_knowledge_item(
    item_id: str,
    body: KnowledgeItemUpdateRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    arq_redis: ArqRedis = Depends(get_arq_redis),  # noqa: B008
) -> Any:
    """Correct a knowledge item in place (Epic 22.2; ``ready`` items since).

    Content warnings are re-derived from the corrected text either way — so a
    reviewer who fixes "no ingredients" stops seeing the flag that said so,
    while ``low_overall_confidence`` / ``low_boundary_confidence`` survive,
    because retyping a line does not attest that the recipe was cut out of the
    page correctly.

    Two statuses, two very different transactions:

    - ``needs_review`` — the original path, and still a pure row rewrite.
      Editing never decides: the item is *still* ``needs_review`` afterwards,
      and the edited text is what gets chunked when the reviewer approves.
      These items have no chunks and no embeddings, so there is nothing else to
      keep in step.
    - ``ready`` — the item is indexed, so a row rewrite alone would leave
      ``chunks`` and ``chunk_embeddings`` describing the *old* text and the
      recipe findable by words it no longer contains. The handler therefore
      drops its index rows in the same transaction, flips it to ``indexing``,
      and enqueues ``index_knowledge_item`` to rebuild and re-embed from the
      saved text — the delete-and-re-embed path this docstring used to say did
      not exist.

    Two consequences of the ``ready`` path, both deliberate. The recipe is
    **absent from search until the worker finishes** (search requires chunks),
    which is the price of not blocking the request on an embedding round-trip.
    And if the job exhausts its retries, the item-level pass in
    ``sweep_stuck_jobs`` returns the row to ``needs_review`` — an edit can
    therefore demote a shelved recipe into the review queue. That is honest
    rather than lossy: the row genuinely has no chunks at that point, which is
    exactly what ``needs_review`` means.

    Guards run in the same order as ``POST …/review`` so each stays reachable
    rather than masked: 404 unknown id → 409 mid-reprocess document → 409 stale
    generation → guarded UPDATE → 404 not editable. The empty-body 400 sits
    *after* the 404 so an unknown id reports as unknown whatever the body says.

    Not closed here (and not asked for by the epic): two concurrent PATCHes are
    last-write-wins on content. The guarded UPDATE closes the edit-vs-decide
    race, and the COALESCE keeps the snapshot at the original extraction, but
    neither is a content-level precondition — two reviewers editing the same
    item in two tabs would see the second edit replace the first wholesale.
    """
    item = await session.get(KnowledgeItem, item_id)
    if item is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
            message=f"Knowledge item {item_id!r} not found.",
            details={"item_id": item_id},
        )

    edit = _recipe_edit_from(body)
    if edit.is_empty():
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="Request body names no editable field.",
            details={"item_id": item_id},
        )

    # Guard 1: a non-terminal document is mid-reprocess — the pipeline is
    # rewriting its items underneath the queue, so an edit would be lost.
    doc_status = (
        await session.execute(select(Document.status).where(Document.id == item.document_id))
    ).scalar_one()
    if doc_status not in TERMINAL_STATUSES:
        raise ApiError(
            status_code=409,
            code=ErrorCode.INGESTION_ALREADY_RUNNING,
            message="Document is not in a terminal state.",
            details={"document_id": item.document_id, "status": doc_status.value},
        )

    # Guard 2: a stale generation is superseded content — correcting it would
    # write into a version no approve can ever accept (the POST's D9 rule).
    current_version = (
        await session.execute(
            select(func.max(KnowledgeItem.source_version)).where(
                KnowledgeItem.document_id == item.document_id
            )
        )
    ).scalar_one()
    if item.source_version < current_version:
        raise ApiError(
            status_code=409,
            code=ErrorCode.REVIEW_ITEM_STALE,
            message=(
                "Knowledge item belongs to a stale extraction generation; "
                "editing is refused (reject to clear it)."
            ),
            details={
                "item_id": item_id,
                "source_version": item.source_version,
                "current_source_version": current_version,
            },
        )

    structured_before = item.structured_data or {}
    edited = apply_edit(
        title=item.title,
        summary=item.summary,
        body_text=item.body_text,
        structured_data=structured_before,
        edit=edit,
    )
    structured_after = {
        **edited.structured_data,
        "warnings": warnings_for_item(
            title=edited.title,
            summary=edited.summary,
            body_text=edited.body_text,
            source_span_ids=list(item.source_span_ids or []),
            structured_data=edited.structured_data,
            confidence=item.confidence,
            thresholds=thresholds_from_settings(),
            item_type=item.item_type,
        ),
    }

    # The original extraction, captured on the FIRST edit only — COALESCE, not a
    # Python `if`, so a second edit cannot overwrite it even if it read the row
    # before the first one committed. It is the undo path and the ground truth
    # the extraction evals must keep scoring.
    snapshot = {
        "title": item.title,
        "summary": item.summary,
        "body_text": item.body_text,
        "structured_data": structured_before,
        "confidence": item.confidence,
    }

    # Guarded UPDATE with whole-object JSONB assignment (the POST's race-closure
    # pattern): a PATCH racing a decide has exactly one winner, and an in-place
    # mutation of the loaded row's `structured_data` — which is NOT dirty-tracked
    # and would vanish at commit — is structurally impossible here.
    #
    # The status is decided IN the same statement rather than read first and
    # written second: a CASE over the OLD value (standard SQL — the SET
    # expression sees the pre-UPDATE row) sends a `ready` item to `indexing`
    # and leaves a `needs_review` one alone, and RETURNING the new value is how
    # the handler learns which of the two branches actually won. Reading the
    # status before the UPDATE and branching in Python would reopen exactly the
    # race the guarded UPDATE exists to close.
    updated = (
        await session.execute(
            update(KnowledgeItem)
            .where(
                KnowledgeItem.id == item_id,
                KnowledgeItem.status.in_(_EDITABLE_STATUSES),
            )
            .values(
                title=edited.title,
                normalized_title=edited.normalized_title,
                summary=edited.summary,
                body_text=edited.body_text,
                structured_data=structured_after,
                pre_edit_snapshot=func.coalesce(
                    KnowledgeItem.pre_edit_snapshot, type_coerce(snapshot, JSONB)
                ),
                edited_at=func.now(),
                status=case(
                    (
                        KnowledgeItem.status == KnowledgeItemStatus.READY,
                        literal(
                            KnowledgeItemStatus.INDEXING,
                            type_=KnowledgeItem.status.type,
                        ),
                    ),
                    else_=KnowledgeItem.status,
                ),
            )
            .execution_options(synchronize_session=False)
            .returning(KnowledgeItem.id, KnowledgeItem.status)
        )
    ).one_or_none()
    if updated is None:
        # Explicit re-select, not the identity-mapped object: the UPDATE ran with
        # synchronize_session=False, so `item.status` could still report
        # needs_review and put a wrong status in the 404 body (the POST's D6).
        current_status = (
            await session.execute(select(KnowledgeItem.status).where(KnowledgeItem.id == item_id))
        ).scalar_one_or_none()
        if current_status is None:
            raise ApiError(
                status_code=404,
                code=ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
                message=f"Knowledge item {item_id!r} not found.",
                details={"item_id": item_id},
            )
        # Same code, wider meaning since the ready path landed: the FE keys its
        # REFUSED_COPY table on the code, so the code is load-bearing and the
        # message is what widens.
        raise ApiError(
            status_code=404,
            code=ErrorCode.REVIEW_NOT_PENDING,
            message=(
                f"Knowledge item {item_id!r} is not editable "
                f"(status {current_status.value!r})."
            ),
            details={"item_id": item_id, "status": current_status.value},
        )

    _, new_status = updated
    reindexing = new_status is KnowledgeItemStatus.INDEXING
    if reindexing:
        # Same transaction as the content write: the edited row and the absence
        # of its stale index rows commit together or not at all.
        #
        # Not optional. `index_knowledge_item` skips `build_chunks` entirely
        # when the item already has chunks (its defensive-idempotency guard),
        # so an edit that left them in place would re-embed the OLD text and
        # report success — the worst available failure mode.
        chunk_counts = await DocumentRepository(session).delete_knowledge_item_chunks(
            item_id
        )
        logger.info(
            "Edit of ready item %s dropped its index rows for re-embedding: %s",
            item_id,
            chunk_counts,
        )

    await session.commit()

    if reindexing:
        try:
            await enqueue_job(
                arq_redis, "index_knowledge_item", item_id, session_id=item.document_id
            )
        except Exception:
            logger.exception(
                "Failed to enqueue index_knowledge_item after editing %s; "
                "reverting to needs_review",
                item_id,
            )
            # The approve path's compensation, verbatim, and for the same
            # reasons (D1) — including the revert target. `needs_review` rather
            # than back to `ready`: the chunks are already gone, so a row
            # labelled ready would claim to be on the shelf while being
            # unfindable. Chunk-free IS what needs_review means.
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
                message="Failed to enqueue the re-indexing job.",
            ) from None

    # Re-read through the ORM so the response reflects what Postgres now holds
    # (server-side `now()`, the COALESCEd snapshot, the new status) rather than
    # what we sent.
    await session.refresh(item)
    return await build_knowledge_item_response(session, item)


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
