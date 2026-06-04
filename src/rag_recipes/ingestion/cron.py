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
from rag_recipes.storage.enums import DocumentStatus
from rag_recipes.storage.models.document import Document

logger = logging.getLogger(__name__)


# Phase 8.1 deliberately rests a successfully-extracted document at
# CREATING_SOURCE_SPANS: extraction is done and spans are persisted, but there
# is no consumer to advance it until Epic 9's process_extraction_run lands. The
# sweep treats every non-terminal status as "stuck", so without this exemption
# a *successful* extraction would be marked failed once it ages past the
# timeout. Exempt the handoff state until Epic 9 owns it.
#
# REMOVE this exemption when process_extraction_run exists — at that point a
# document wedged in CREATING_SOURCE_SPANS (spans written, item-extraction
# stalled) is genuinely stuck and should be swept again.
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
        stuck: list[tuple[str, DocumentStatus]] = list(result.all())

        for doc_id, last_status in stuck:
            try:
                async with session.begin_nested():
                    await mark_failed(
                        session,
                        doc_id,
                        reason="stuck_job_timeout",
                        metadata_json={
                            "timeout_minutes": timeout_minutes,
                            "last_seen_status": last_status.value,
                        },
                    )
                count += 1
            except InvalidTransitionError as exc:
                logger.warning("Sweep skipped %s: %s", doc_id, exc)
                continue

        await session.commit()

    logger.info("Stuck-job sweep marked %d documents failed", count)
    return count
