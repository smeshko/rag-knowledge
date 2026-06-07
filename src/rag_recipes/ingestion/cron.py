"""Cron coroutines registered on the arq worker.

``sweep_stuck_jobs`` is the Phase 7.2 recovery mechanism: documents stuck
in a non-terminal status longer than ``stuck_job_timeout_minutes`` are
marked ``failed`` with reason ``stuck_job_timeout`` and a forensic row in
``ingestion_failures``. Per-doc transition rejections (e.g. the doc
became terminal between the SELECT and the row lock) are logged at
WARNING and skipped via a savepoint rollback so the rest of the batch
still lands.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from rag_recipes.config import Settings
from rag_recipes.ingestion.status import (
    TERMINAL_STATUSES,
    InvalidTransitionError,
    mark_failed,
)
from rag_recipes.storage.enums import DocumentStatus, ExtractionBatchItemStatus
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem

logger = logging.getLogger(__name__)

# A document in EXTRACTING_ITEMS with an *in-flight* batch item may legitimately
# sit far past the sweep threshold (Anthropic batches run up to 24h), so it is
# exempt. Crucially this is NOT extended to PENDING: a doc whose only items are
# PENDING means the submitter never claimed them (disabled/broken), so it must be
# surfaced (reaped), never hidden forever (Epic 19.2 DECISIONS #3).
_IN_FLIGHT_BATCH_ITEM_STATUSES = (
    ExtractionBatchItemStatus.SUBMITTING,
    ExtractionBatchItemStatus.SUBMITTED,
)


# The sweep treats every non-terminal status as "stuck", so a status that a
# successful document *rests* at — done for now, but with no consumer to advance
# it until a later phase lands — must be exempted, or a successful document would
# be marked failed once it ages past the timeout.
#
# - CREATING_SOURCE_SPANS (Phase 8.1): extraction done, spans persisted, awaiting
#   Epic 9's process_extraction_run. (Epic 9 has since landed and advances past
#   it, so this is now only reached transiently; kept until the handoff is retired.)
# After Phase 10.3 the pipeline runs all the way to a terminal status
# (READY/NEEDS_REVIEW), so its post-extraction stages (CREATING_CHUNKS,
# EMBEDDING_CHUNKS, INDEXING) are transient: a crash in any of them rolls back to
# the status it started from, which is a resumable entry point in
# jobs._resume_or_fresh. A document wedged in one of those past the timeout is
# genuinely stuck (the worker never re-delivered it) and should be swept to FAILED
# so it is visible for reprocessing — so none of them are exempt.
#
# CREATING_SOURCE_SPANS (Phase 8.1) remains exempt: a successful extraction can
# still rest there with no consumer until Epic 9's handoff is fully retired.
# REMOVE it too once that handoff no longer rests a successful document there.
_SWEEP_EXEMPT_STATUSES: frozenset[DocumentStatus] = frozenset(
    {DocumentStatus.CREATING_SOURCE_SPANS}
)


async def sweep_stuck_jobs(ctx: dict[str, Any]) -> int:
    """Mark documents stalled past ``stuck_job_timeout_minutes`` as failed.

    Staleness keys on real progress, not just any row write: a document
    advancing ``last_progress_at`` (Phase 9.5 commits the heartbeat once per
    extraction batch) is healthy and never reaped, even when ``updated_at`` is
    old. ``coalesce(last_progress_at, updated_at)`` falls back to ``updated_at``
    for pre-extraction stages, whose heartbeat is null, so their semantics are
    unchanged. The timeout default stays at 30 minutes — the progress-aware
    predicate makes a shorter value *safe*, but tuning it is a separate ops
    decision (DECISIONS #6).
    """
    settings: Settings = ctx["settings"]
    session_factory: async_sessionmaker[Any] = ctx["session_factory"]
    timeout_minutes = settings.stuck_job_timeout_minutes
    threshold = datetime.now(tz=UTC) - timedelta(minutes=timeout_minutes)

    count = 0
    async with session_factory() as session:
        result = await session.execute(
            select(Document.id, Document.status)
            .where(Document.status.notin_(TERMINAL_STATUSES))
            .where(Document.status.notin_(_SWEEP_EXEMPT_STATUSES))
            .where(
                func.coalesce(Document.last_progress_at, Document.updated_at)
                < threshold
            )
        )
        stuck: list[str] = [doc_id for doc_id, _ in result.all()]

        for doc_id in stuck:
            try:
                async with session.begin_nested():
                    # The SELECT snapshot is stale. Between it and this row lock the
                    # worker can advance the document — the atomic finalize commits
                    # straight into CREATING_CHUNKS — or refresh its heartbeat. Re-read
                    # the now-locked row and skip if it has become terminal, reached a
                    # sweep-exempt resting state, or progressed past the threshold.
                    # Without this, mark_failed only checks the transition is *legal*,
                    # and EXTRACTING_ITEMS -> CREATING_CHUNKS -> FAILED is legal, so a
                    # successfully chunked document could still be failed (review #3).
                    doc = await session.get(Document, doc_id, with_for_update=True)
                    if doc is None:
                        continue
                    last_progress = doc.last_progress_at or doc.updated_at
                    if (
                        doc.status in TERMINAL_STATUSES
                        or doc.status in _SWEEP_EXEMPT_STATUSES
                        or last_progress >= threshold
                    ):
                        continue
                    # Batch-aware exemption (Epic 19.2): an EXTRACTING_ITEMS doc
                    # with an in-flight (SUBMITTING/SUBMITTED) batch item may
                    # legitimately wait up to 24h for results — don't reap it. A
                    # stale-PENDING-only doc (submitter never claimed) and a
                    # synchronous EXTRACTING_ITEMS doc with no items both fall
                    # through and are reaped (DECISIONS #3).
                    if doc.status is DocumentStatus.EXTRACTING_ITEMS:
                        in_flight = await session.execute(
                            select(ExtractionBatchItem.id)
                            .where(
                                ExtractionBatchItem.document_id == doc_id,
                                ExtractionBatchItem.status.in_(
                                    _IN_FLIGHT_BATCH_ITEM_STATUSES
                                ),
                            )
                            .limit(1)
                        )
                        if in_flight.first() is not None:
                            continue
                    await mark_failed(
                        session,
                        doc_id,
                        reason="stuck_job_timeout",
                        metadata_json={
                            "timeout_minutes": timeout_minutes,
                            "last_seen_status": doc.status.value,
                        },
                    )
                count += 1
            except InvalidTransitionError as exc:
                logger.warning("Sweep skipped %s: %s", doc_id, exc)
                continue

        await session.commit()

    logger.info("Stuck-job sweep marked %d documents failed", count)
    return count
