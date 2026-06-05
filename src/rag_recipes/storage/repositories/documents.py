"""Data-access primitives for ``SourceAsset`` + ``Document``.

The repository owns no transaction control: callers (the upload route)
flush/commit/rollback explicitly. Methods that insert ``session.add`` +
``await session.flush()`` so server defaults and IDs are populated before
the caller takes the next step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import CursorResult, Integer, cast, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import (
    DocumentStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan


@dataclass(frozen=True)
class KnowledgeItemCounts:
    total: int
    ready: int
    needs_review: int


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_source_asset_by_content_hash(
        self, content_hash: str
    ) -> SourceAsset | None:
        result = await self._session.execute(
            select(SourceAsset).where(SourceAsset.content_hash == content_hash)
        )
        return result.scalar_one_or_none()

    async def get_document_by_asset_id(self, asset_id: str) -> Document | None:
        # Explicit query — never `await`-trigger the lazy SourceAsset.document
        # relationship, which is a default-lazy load and breaks on AsyncSession.
        result = await self._session.execute(
            select(Document).where(Document.asset_id == asset_id)
        )
        return result.scalar_one_or_none()

    async def get_source_asset_for_document(
        self, document: Document
    ) -> SourceAsset | None:
        # Explicit query — never `await document.asset` from an AsyncSession
        # (lazy-load trap, same as get_document_by_asset_id).
        result = await self._session.execute(
            select(SourceAsset).where(SourceAsset.id == document.asset_id)
        )
        return result.scalar_one_or_none()

    async def supersede_prior_items(
        self, document_id: str, *, keep_extraction_run_ids: set[str]
    ) -> int:
        """Supersede a document's prior KnowledgeItems, sparing the kept runs.

        Single bulk indexed UPDATE flipping every non-superseded ``KnowledgeItem``
        for ``document_id`` whose ``extraction_run_id`` is NOT in
        ``keep_extraction_run_ids`` to ``SUPERSEDED``; returns the affected row
        count. The keep-set framing ("supersede everything *not* produced by the
        accepted run") is exact and clock-skew-proof versus a timestamp filter
        (DECISIONS #4): Epic 10.3 calls it with the just-accepted run ids at the
        READY transition and Epic 11 with the new source-version's runs. For first
        extraction the keep-set covers the current pass's runs, so nothing matches
        and 0 is returned. Already-``SUPERSEDED`` rows are excluded by the status
        guard and never re-touched.

        Caller owns the transaction (consistent with the other repo writes and
        ``transition_to``); this flush-level write does not commit.
        ``synchronize_session=False`` keeps it a single round-trip — the caller
        does not rely on the in-session ORM objects reflecting the new status.
        """
        stmt = (
            update(KnowledgeItem)
            .where(
                KnowledgeItem.document_id == document_id,
                KnowledgeItem.status != KnowledgeItemStatus.SUPERSEDED,
                KnowledgeItem.extraction_run_id.notin_(keep_extraction_run_ids),
            )
            .values(status=KnowledgeItemStatus.SUPERSEDED)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(stmt)
        # AsyncSession.execute is typed Result[Any]; an UPDATE yields a
        # CursorResult, whose rowcount is the affected-row count. Narrow for the
        # typed int access.
        assert isinstance(result, CursorResult)
        return result.rowcount

    async def add_source_asset(
        self,
        *,
        id: str,
        source_type: SourceType,
        original_filename: str,
        storage_provider: str,
        storage_key: str,
        content_hash: str,
        upload_status: UploadStatus,
    ) -> SourceAsset:
        asset = SourceAsset(
            id=id,
            source_type=source_type,
            original_filename=original_filename,
            storage_provider=storage_provider,
            storage_key=storage_key,
            content_hash=content_hash,
            upload_status=upload_status,
        )
        self._session.add(asset)
        await self._session.flush()
        return asset

    async def add_document(
        self,
        *,
        asset_id: str,
        category: str,
        subcategory: str | None,
        title: str,
        author: str,
        source_type: SourceType,
        language: str | None,
        active_source_version: int | None,
        status: DocumentStatus,
    ) -> Document:
        # Set source_type as a scalar column; do not attach the asset
        # relationship — that would trigger the cross-validator against a
        # possibly-half-built graph (see PLAN Risks).
        document = Document(
            asset_id=asset_id,
            category=category,
            subcategory=subcategory,
            title=title,
            author=author,
            source_type=source_type,
            language=language,
            active_source_version=active_source_version,
            status=status,
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def get_document(self, document_id: str) -> Document | None:
        return await self._session.get(Document, document_id)

    async def list_documents(
        self,
        *,
        category: str | None,
        status: DocumentStatus | None,
        source_type: SourceType | None,
        limit: int,
        offset: int,
    ) -> Sequence[Document]:
        stmt = select(Document)
        if category is not None:
            stmt = stmt.where(Document.category == category)
        if status is not None:
            stmt = stmt.where(Document.status == status)
        if source_type is not None:
            stmt = stmt.where(Document.source_type == source_type)
        # id desc as a stable tiebreak when created_at collides.
        stmt = stmt.order_by(Document.created_at.desc(), Document.id.desc())
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def count_source_spans(self, document_id: str) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(SourceSpan).where(
                SourceSpan.document_id == document_id
            )
        )
        return result.scalar_one()

    async def get_latest_extraction_run(
        self, document_id: str
    ) -> ExtractionRun | None:
        """Return the document's most recent ExtractionRun, or None.

        Used by the Epic 11.2 ``auto`` reprocess selector to compare the last run's
        ``prompt_version`` / ``schema_version`` against the current constants.
        Ordered by ``created_at`` desc with an ``id`` tiebreak for stability.
        """
        result = await self._session.execute(
            select(ExtractionRun)
            .where(ExtractionRun.document_id == document_id)
            .order_by(ExtractionRun.created_at.desc(), ExtractionRun.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_version_extractor_identity(
        self, document_id: str, source_version: int
    ) -> str | None:
        """Return the extractor identity stamped on a version's spans, or None.

        Reads ``locator["meta"]["extractor_identity"]`` from one span of the given
        version (every span of a version carries the same identity). Returns None
        when the version has no spans, or for legacy spans written before Epic 11.2
        stamped the identity — the ``auto`` selector treats a missing identity as
        "differs from the current extractor" and defaults to a new-version run.
        """
        result = await self._session.execute(
            select(SourceSpan.locator["meta"]["extractor_identity"].astext)
            .where(
                SourceSpan.document_id == document_id,
                SourceSpan.source_version == source_version,
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def max_source_version(self, document_id: str) -> int | None:
        """Return the document's highest ``SourceSpan.source_version``, or None.

        Used by the reprocess endpoint to resolve the version a reuse run reuses
        when the document has no ``active_source_version`` yet (e.g. a FAILED doc
        whose first extraction persisted spans but never reached READY). The
        aggregate yields exactly one row whose value is NULL → ``None`` when the
        document has no spans at all.
        """
        result = await self._session.execute(
            select(func.max(SourceSpan.source_version)).where(
                SourceSpan.document_id == document_id
            )
        )
        return result.scalar_one()

    async def get_pages_progress(
        self, document_id: str, source_version: int
    ) -> tuple[int, int | None]:
        # Combined count + max(page_end) in one round-trip. page_end lives in
        # the JSONB locator column; the ->> accessor returns text, which we
        # cast to int. max over zero rows is NULL → (0, None).
        result = await self._session.execute(
            select(
                func.count(SourceSpan.id),
                func.max(cast(SourceSpan.locator["page_end"].astext, Integer)),
            ).where(
                SourceSpan.document_id == document_id,
                SourceSpan.source_version == source_version,
            )
        )
        count, max_page = result.one()
        return (count, max_page)

    async def count_chunks(self, document_id: str) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(Chunk).where(
                Chunk.document_id == document_id
            )
        )
        return result.scalar_one()

    async def count_knowledge_items(self, document_id: str) -> KnowledgeItemCounts:
        # Single GROUP BY query — one round-trip for total / ready / needs_review.
        result = await self._session.execute(
            select(KnowledgeItem.status, func.count())
            .where(KnowledgeItem.document_id == document_id)
            .group_by(KnowledgeItem.status)
        )
        by_status: dict[KnowledgeItemStatus, int] = {
            status: count for status, count in result.all()
        }
        return KnowledgeItemCounts(
            total=sum(by_status.values()),
            ready=by_status.get(KnowledgeItemStatus.READY, 0),
            needs_review=by_status.get(KnowledgeItemStatus.NEEDS_REVIEW, 0),
        )
