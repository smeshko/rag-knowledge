from __future__ import annotations

import hashlib

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository


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
