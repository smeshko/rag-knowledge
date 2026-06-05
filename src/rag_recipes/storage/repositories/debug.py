"""Read-only loaders for the dev-only debug endpoints (doc 6 § 9).

Explicit queries only (no async lazy relationship loads). Extraction runs and
source spans are append-only audit records — these are read-only views.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Integer, cast, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import ExtractionRunStatus
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.source_span import SourceSpan


class DebugRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_extraction_runs(
        self,
        document_id: str,
        *,
        source_version: int | None,
        status: ExtractionRunStatus | None,
    ) -> Sequence[ExtractionRun]:
        stmt = select(ExtractionRun).where(ExtractionRun.document_id == document_id)
        if source_version is not None:
            stmt = stmt.where(ExtractionRun.source_version == source_version)
        if status is not None:
            stmt = stmt.where(ExtractionRun.status == status)
        stmt = stmt.order_by(ExtractionRun.created_at.desc(), ExtractionRun.id.desc())
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get_extraction_run(self, run_id: str) -> ExtractionRun | None:
        return await self._session.get(ExtractionRun, run_id)

    async def list_source_spans(
        self,
        document_id: str,
        *,
        source_version: int | None,
        page_start: int | None,
        page_end: int | None,
    ) -> Sequence[SourceSpan]:
        # page_start/page_end live in the JSONB locator; cast the ->> text to int
        # (the get_pages_progress idiom). A span is returned when its own page range
        # OVERLAPS the requested [page_start, page_end] window.
        span_start = cast(SourceSpan.locator["page_start"].astext, Integer)
        span_end = cast(SourceSpan.locator["page_end"].astext, Integer)
        stmt = select(SourceSpan).where(SourceSpan.document_id == document_id)
        if source_version is not None:
            stmt = stmt.where(SourceSpan.source_version == source_version)
        if page_start is not None:
            stmt = stmt.where(span_end >= page_start)
        if page_end is not None:
            stmt = stmt.where(span_start <= page_end)
        stmt = stmt.order_by(span_start, SourceSpan.id)
        result = await self._session.execute(stmt)
        return result.scalars().all()
