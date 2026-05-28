"""Centralised ``Document.status`` transition helpers (doc 2 § status).

Every legal ``Document.status`` mutation flows through this module. The
``VALID_TRANSITIONS`` matrix mirrors doc 2 § status verbatim; rejecting an
edge raises ``InvalidTransitionError`` so the cron and Epic 8+ jobs share
one enforcement point. Callers own the transaction — ``transition_to``
flushes but does not commit, so ``mark_failed``'s two writes land
atomically under the caller's outer ``async with session.begin():``.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import DocumentStatus
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.repositories.failures import FailuresRepository

logger = logging.getLogger(__name__)


TERMINAL_STATUSES: frozenset[DocumentStatus] = frozenset(
    {DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW, DocumentStatus.FAILED}
)


VALID_TRANSITIONS: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DocumentStatus.QUEUED: frozenset(
        {DocumentStatus.EXTRACTING_TEXT, DocumentStatus.FAILED}
    ),
    DocumentStatus.EXTRACTING_TEXT: frozenset(
        {DocumentStatus.CREATING_SOURCE_SPANS, DocumentStatus.FAILED}
    ),
    DocumentStatus.CREATING_SOURCE_SPANS: frozenset(
        {DocumentStatus.EXTRACTING_ITEMS, DocumentStatus.FAILED}
    ),
    DocumentStatus.EXTRACTING_ITEMS: frozenset(
        {DocumentStatus.VALIDATING_ITEMS, DocumentStatus.FAILED}
    ),
    DocumentStatus.VALIDATING_ITEMS: frozenset(
        {DocumentStatus.CREATING_CHUNKS, DocumentStatus.FAILED}
    ),
    DocumentStatus.CREATING_CHUNKS: frozenset(
        {DocumentStatus.EMBEDDING_CHUNKS, DocumentStatus.FAILED}
    ),
    DocumentStatus.EMBEDDING_CHUNKS: frozenset(
        {DocumentStatus.INDEXING, DocumentStatus.FAILED}
    ),
    DocumentStatus.INDEXING: frozenset(
        {
            DocumentStatus.READY,
            DocumentStatus.NEEDS_REVIEW,
            DocumentStatus.FAILED,
        }
    ),
    DocumentStatus.READY: frozenset({DocumentStatus.QUEUED}),
    DocumentStatus.NEEDS_REVIEW: frozenset({DocumentStatus.QUEUED}),
    DocumentStatus.FAILED: frozenset({DocumentStatus.QUEUED}),
}


def is_terminal(status: DocumentStatus) -> bool:
    return status in TERMINAL_STATUSES


class InvalidTransitionError(Exception):
    def __init__(
        self,
        *,
        current: DocumentStatus,
        attempted: DocumentStatus,
        allowed: frozenset[DocumentStatus],
    ) -> None:
        self.current = current
        self.attempted = attempted
        self.allowed = allowed
        allowed_repr = sorted(s.value for s in allowed)
        super().__init__(
            f"Invalid Document.status transition: "
            f"{current.value} -> {attempted.value} "
            f"(allowed: {allowed_repr})"
        )


async def transition_to(
    session: AsyncSession,
    document_id: str,
    new_status: DocumentStatus,
    *,
    message: str | None = None,
) -> Document:
    """Transition a document to ``new_status`` under a row-level lock.

    Raises ``LookupError`` if the document does not exist and
    ``InvalidTransitionError`` if the edge is not in ``VALID_TRANSITIONS``.
    Flushes the change so server-side ``updated_at`` bumps land; the caller
    commits or rolls back.
    """
    doc = await session.get(Document, document_id, with_for_update=True)
    if doc is None:
        raise LookupError(f"Document not found: {document_id}")

    allowed = VALID_TRANSITIONS[doc.status]
    if new_status not in allowed:
        raise InvalidTransitionError(
            current=doc.status, attempted=new_status, allowed=allowed
        )

    log_msg = f"Document {document_id} {doc.status.value} -> {new_status.value}"
    if message:
        log_msg = f"{log_msg} ({message})"
    logger.info(log_msg)

    doc.status = new_status
    await session.flush()
    return doc


async def mark_failed(
    session: AsyncSession,
    document_id: str,
    *,
    reason: str,
    error_message: str | None = None,
    metadata_json: dict[str, Any] | None = None,
) -> Document:
    """Record a failure row, then transition the document to ``failed``.

    The failure row is written first so that even if the transition is
    rejected (e.g. the document is already terminal because another sweep
    or job won the race), the caller's rollback unwinds both writes
    atomically and a commit on the success path lands them both.
    """
    current = await session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    if current is None:
        raise LookupError(f"Document not found: {document_id}")

    await FailuresRepository(session).add_failure(
        document_id=document_id,
        last_status=current,
        reason=reason,
        error_message=error_message,
        metadata_json=metadata_json,
    )
    return await transition_to(
        session,
        document_id,
        DocumentStatus.FAILED,
        message=f"mark_failed: {reason}",
    )
