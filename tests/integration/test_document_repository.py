from __future__ import annotations

import hashlib

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository


async def _add_run(
    session: AsyncSession, *, document_id: str, source_version: int = 1
) -> ExtractionRun:
    run = ExtractionRun(
        document_id=document_id,
        source_version=source_version,
        provider="fake",
        model="fake-model",
        prompt_version="test-prompt-v1",
        schema_version="test-schema-v1",
        input_source_span_ids=[],
        input_hash=new_id("hash"),
        status=ExtractionRunStatus.SUCCESS,
        output_json=None,
    )
    session.add(run)
    await session.flush()
    return run


async def _add_item(
    session: AsyncSession,
    *,
    document_id: str,
    extraction_run_id: str,
    status: KnowledgeItemStatus,
    source_version: int = 1,
) -> KnowledgeItem:
    item = KnowledgeItem(
        document_id=document_id,
        extraction_run_id=extraction_run_id,
        source_version=source_version,
        item_type="recipe",
        title=f"Item {new_id('t')}",
        normalized_title="item",
        summary=None,
        body_text="body " * 20,
        source_span_ids=[],
        structured_data={"schema": "recipe.v1"},
        confidence=None,
        status=status,
    )
    session.add(item)
    await session.flush()
    return item


async def _make_asset_and_document(
    repo: DocumentRepository,
    *,
    content_hash: str,
    asset_id: str | None = None,
) -> tuple[SourceAsset, str]:
    aid = asset_id or new_id(SourceAsset.ID_PREFIX)
    asset = await repo.add_source_asset(
        id=aid,
        source_type=SourceType.PDF,
        original_filename="example.pdf",
        storage_provider="local",
        storage_key=f"source-assets/{aid}/original.pdf",
        content_hash=content_hash,
        upload_status=UploadStatus.UPLOADED,
    )
    document = await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="Example",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.QUEUED,
    )
    return asset, document.id


async def _add_span(
    session: AsyncSession,
    *,
    document_id: str,
    source_version: int,
    page_end: int,
) -> None:
    locator = {"type": "pdf_page_range", "page_start": page_end, "page_end": page_end}
    marker = f"{document_id}-{source_version}-{page_end}"
    session.add(
        SourceSpan(
            document_id=document_id,
            source_version=source_version,
            source_type=SourceType.PDF,
            locator=locator,
            locator_hash=hashlib.sha256(marker.encode()).hexdigest(),
            text=f"text-{marker}",
            text_hash=hashlib.sha256(f"text-{marker}".encode()).hexdigest(),
        )
    )
    await session.flush()


class TestGetPagesProgress:
    @pytest.mark.asyncio
    async def test_returns_zero_none_for_no_spans(
        self, db_session: AsyncSession
    ) -> None:
        repo = DocumentRepository(db_session)
        _, document_id = await _make_asset_and_document(repo, content_hash="prog-none")
        assert await repo.get_pages_progress(document_id, 1) == (0, None)

    @pytest.mark.asyncio
    async def test_returns_count_and_max_for_contiguous_spans(
        self, db_session: AsyncSession
    ) -> None:
        repo = DocumentRepository(db_session)
        _, document_id = await _make_asset_and_document(repo, content_hash="prog-3")
        for page in (1, 2, 3):
            await _add_span(
                db_session, document_id=document_id, source_version=1, page_end=page
            )
        assert await repo.get_pages_progress(document_id, 1) == (3, 3)

    @pytest.mark.asyncio
    async def test_scopes_to_source_version(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        _, document_id = await _make_asset_and_document(repo, content_hash="prog-ver")
        for page in (1, 2, 3):
            await _add_span(
                db_session, document_id=document_id, source_version=1, page_end=page
            )
        for page in (1, 2):
            await _add_span(
                db_session, document_id=document_id, source_version=2, page_end=page
            )
        assert await repo.get_pages_progress(document_id, 1) == (3, 3)
        assert await repo.get_pages_progress(document_id, 2) == (2, 2)

    @pytest.mark.asyncio
    async def test_handles_non_contiguous_pages(
        self, db_session: AsyncSession
    ) -> None:
        repo = DocumentRepository(db_session)
        _, document_id = await _make_asset_and_document(repo, content_hash="prog-gap")
        await _add_span(
            db_session, document_id=document_id, source_version=1, page_end=1
        )
        await _add_span(
            db_session, document_id=document_id, source_version=1, page_end=5
        )
        # Diagnostic signal preserved: count (2) != max_page_end (5).
        assert await repo.get_pages_progress(document_id, 1) == (2, 5)


class TestGetSourceAssetByContentHash:
    @pytest.mark.asyncio
    async def test_returns_asset_when_hash_matches(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        await _make_asset_and_document(repo, content_hash="hash-hit-1")
        found = await repo.get_source_asset_by_content_hash("hash-hit-1")
        assert found is not None
        assert found.content_hash == "hash-hit-1"
        assert found.upload_status == UploadStatus.UPLOADED

    @pytest.mark.asyncio
    async def test_returns_none_when_hash_absent(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        found = await repo.get_source_asset_by_content_hash("hash-not-stored")
        assert found is None


class TestGetDocumentByAssetId:
    @pytest.mark.asyncio
    async def test_returns_document_for_existing_asset(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        asset, document_id = await _make_asset_and_document(repo, content_hash="hash-doc-1")
        found = await repo.get_document_by_asset_id(asset.id)
        assert found is not None
        assert found.id == document_id
        assert found.asset_id == asset.id

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_asset_id(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        found = await repo.get_document_by_asset_id("asset_does_not_exist")
        assert found is None


class TestAddRoundtrip:
    @pytest.mark.asyncio
    async def test_full_insert_round_trips_with_defaults(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        asset, document_id = await _make_asset_and_document(repo, content_hash="hash-rt-1")
        document = await repo.get_document(document_id)
        assert document is not None
        await db_session.refresh(document)
        assert document.status == DocumentStatus.QUEUED
        assert document.active_source_version is None
        assert document.source_type == SourceType.PDF
        assert document.category == "recipes"
        assert document.created_at is not None
        assert document.updated_at is not None
        assert document.asset_id == asset.id


class TestDuplicateContentHash:
    @pytest.mark.asyncio
    async def test_second_insert_with_same_hash_raises_integrity_error(
        self, db_session: AsyncSession
    ) -> None:
        repo = DocumentRepository(db_session)
        await _make_asset_and_document(repo, content_hash="hash-dup-1")
        # Second insert with the same content_hash must violate the unique constraint
        # at flush time.
        with pytest.raises(IntegrityError):
            await repo.add_source_asset(
                id=new_id(SourceAsset.ID_PREFIX),
                source_type=SourceType.PDF,
                original_filename="other.pdf",
                storage_provider="local",
                storage_key="source-assets/other/original.pdf",
                content_hash="hash-dup-1",
                upload_status=UploadStatus.UPLOADED,
            )


class TestReviewStatusRoundtrip:
    """Phase 21.3 (TASK-001): the two new statuses persist through the native enum."""

    @pytest.mark.asyncio
    async def test_rejected_and_indexing_persist_and_read_back(
        self, db_session: AsyncSession
    ) -> None:
        repo = DocumentRepository(db_session)
        _, document_id = await _make_asset_and_document(repo, content_hash="rt-review-1")
        run = await _add_run(db_session, document_id=document_id)
        rejected = await _add_item(
            db_session,
            document_id=document_id,
            extraction_run_id=run.id,
            status=KnowledgeItemStatus.REJECTED,
        )
        indexing = await _add_item(
            db_session,
            document_id=document_id,
            extraction_run_id=run.id,
            status=KnowledgeItemStatus.INDEXING,
        )
        db_session.expunge_all()
        reloaded_rejected = await db_session.get(KnowledgeItem, rejected.id)
        reloaded_indexing = await db_session.get(KnowledgeItem, indexing.id)
        assert reloaded_rejected is not None
        assert reloaded_rejected.status is KnowledgeItemStatus.REJECTED
        assert reloaded_indexing is not None
        assert reloaded_indexing.status is KnowledgeItemStatus.INDEXING


class TestSupersedeTerminalGuards:
    """Phase 21.3 (D5): supersede_prior_items must not touch rejected/indexing rows."""

    @pytest.mark.asyncio
    async def test_supersede_spares_rejected_and_indexing_rows(
        self, db_session: AsyncSession
    ) -> None:
        repo = DocumentRepository(db_session)
        _, document_id = await _make_asset_and_document(repo, content_hash="sup-guard-1")
        kept_run = await _add_run(db_session, document_id=document_id)
        prior_run = await _add_run(db_session, document_id=document_id)
        ready_prior = await _add_item(
            db_session,
            document_id=document_id,
            extraction_run_id=prior_run.id,
            status=KnowledgeItemStatus.READY,
        )
        rejected_prior = await _add_item(
            db_session,
            document_id=document_id,
            extraction_run_id=prior_run.id,
            status=KnowledgeItemStatus.REJECTED,
        )
        indexing_prior = await _add_item(
            db_session,
            document_id=document_id,
            extraction_run_id=prior_run.id,
            status=KnowledgeItemStatus.INDEXING,
        )

        flipped = await repo.supersede_prior_items(
            document_id, keep_extraction_run_ids={kept_run.id}
        )

        assert flipped == 1  # only the ready sibling
        db_session.expunge_all()
        ready_reloaded = await db_session.get(KnowledgeItem, ready_prior.id)
        rejected_reloaded = await db_session.get(KnowledgeItem, rejected_prior.id)
        indexing_reloaded = await db_session.get(KnowledgeItem, indexing_prior.id)
        assert ready_reloaded is not None
        assert ready_reloaded.status is KnowledgeItemStatus.SUPERSEDED
        assert rejected_reloaded is not None
        assert rejected_reloaded.status is KnowledgeItemStatus.REJECTED
        assert indexing_reloaded is not None
        assert indexing_reloaded.status is KnowledgeItemStatus.INDEXING


class TestCountKnowledgeItemsExcludesRejected:
    """Phase 21.3 (D5): rejection drops the row out of counts.total."""

    @pytest.mark.asyncio
    async def test_total_excludes_rejected(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        _, document_id = await _make_asset_and_document(repo, content_hash="cnt-rej-1")
        run = await _add_run(db_session, document_id=document_id)
        for status in (
            KnowledgeItemStatus.READY,
            KnowledgeItemStatus.NEEDS_REVIEW,
            KnowledgeItemStatus.REJECTED,
        ):
            await _add_item(
                db_session,
                document_id=document_id,
                extraction_run_id=run.id,
                status=status,
            )

        counts = await repo.count_knowledge_items(document_id)

        assert counts.total == 2
        assert counts.ready == 1
        assert counts.needs_review == 1
