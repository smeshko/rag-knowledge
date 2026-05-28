"""Data-access primitives for ``IngestionFailure``.

Append-only forensic log for the stuck-job-recovery cron (and, from Epic
8 onwards, real ingestion jobs). Caller owns the transaction; methods
insert + flush so server defaults (``failed_at``, ``metadata_json``) are
populated before the caller commits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import DocumentStatus
from rag_recipes.storage.models.ingestion_failure import IngestionFailure


class FailuresRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_failure(
        self,
        *,
        document_id: str,
        last_status: DocumentStatus,
        reason: str,
        error_message: str | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> IngestionFailure:
        failure = IngestionFailure(
            document_id=document_id,
            last_status=last_status,
            reason=reason,
            error_message=error_message,
        )
        if metadata_json is not None:
            failure.metadata_json = metadata_json
        self._session.add(failure)
        await self._session.flush()
        await self._session.refresh(failure, ["failed_at", "metadata_json"])
        return failure

    async def list_failures(
        self, document_id: str, limit: int = 10
    ) -> Sequence[IngestionFailure]:
        result = await self._session.execute(
            select(IngestionFailure)
            .where(IngestionFailure.document_id == document_id)
            .order_by(IngestionFailure.failed_at.desc(), IngestionFailure.id.desc())
            .limit(limit)
        )
        return result.scalars().all()
