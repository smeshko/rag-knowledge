"""Integration tests for ingestion.pipeline.pdf_text.extract_and_persist_spans.

Real Postgres + an in-memory FakePdfTextExtractor (so the suspicion path is
deterministic) + FakeFileStorageProvider. Exercises the happy path, the
uniqueness contract, and the documented error modes.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.pdf_text import (
    EmptyPdfError,
    _sha256_json,
    _sha256_text,
    extract_and_persist_spans,
)
from rag_recipes.providers.file_storage.fake import FakeFileStorageProvider
from rag_recipes.providers.pdf_extractor.fake import FakePdfTextExtractor
from rag_recipes.providers.pdf_extractor.types import PdfPageText
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository

PDF_BYTES = b"%PDF-1.4 three page recipe fixture"
STORAGE_KEY = "source-assets/test-asset/original.pdf"

_PAGES = [
    PdfPageText(
        page_number=1,
        text="Classic Pancakes\nIngredients\n",
        extraction_method="embedded_text",
        confidence=None,
    ),
    PdfPageText(
        page_number=2,
        text="Instructions\nMix and cook.\n",
        extraction_method="embedded_text",
        confidence=None,
    ),
    PdfPageText(
        page_number=3,
        text="",
        extraction_method="embedded_text",
        confidence=0.0,
    ),
]


@pytest.fixture
def fake_extractor() -> FakePdfTextExtractor:
    return FakePdfTextExtractor(
        {FakePdfTextExtractor.content_hash(PDF_BYTES): _PAGES}
    )


@pytest.fixture
def empty_extractor() -> FakePdfTextExtractor:
    return FakePdfTextExtractor(
        {FakePdfTextExtractor.content_hash(PDF_BYTES): []}
    )


@pytest_asyncio.fixture
async def fake_storage() -> FakeFileStorageProvider:
    storage = FakeFileStorageProvider()
    await storage.put_object(STORAGE_KEY, PDF_BYTES, "application/pdf")
    return storage


async def _insert_document(session: AsyncSession) -> str:
    repo = DocumentRepository(session)
    asset = await repo.add_source_asset(
        id=new_id("asset"),
        source_type=SourceType.PDF,
        original_filename="recipe.pdf",
        storage_provider="fake",
        storage_key=STORAGE_KEY,
        content_hash=FakePdfTextExtractor.content_hash(PDF_BYTES),
        upload_status=UploadStatus.UPLOADED,
    )
    document = await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="My Recipe",
        author="Alice",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.QUEUED,
    )
    return document.id


@pytest.mark.asyncio
async def test_happy_path_three_pages_creates_three_spans(
    db_session: AsyncSession,
    fake_extractor: FakePdfTextExtractor,
    fake_storage: FakeFileStorageProvider,
) -> None:
    document_id = await _insert_document(db_session)

    count = await extract_and_persist_spans(
        db_session,
        document_id=document_id,
        source_version=1,
        extractor=fake_extractor,
        storage=fake_storage,
    )
    assert count == 3

    result = await db_session.execute(
        select(SourceSpan).where(SourceSpan.document_id == document_id)
    )
    spans = sorted(result.scalars().all(), key=lambda s: s.locator["page_start"])
    assert len(spans) == 3
    assert all(s.source_version == 1 for s in spans)
    assert all(s.source_type == SourceType.PDF for s in spans)
    assert [s.locator["page_start"] for s in spans] == [1, 2, 3]
    assert [s.locator["meta"]["suspicious"] for s in spans] == [False, False, True]

    first = spans[0]
    assert first.locator_hash == _sha256_json(
        {"type": "pdf_page_range", "page_start": 1, "page_end": 1}
    )
    assert first.text_hash == _sha256_text(_PAGES[0].text)
    # The meta key is carried in the locator but excluded from the hash input.
    assert "meta" in first.locator
    assert first.locator["meta"]["extraction_method"] == "embedded_text"


@pytest.mark.asyncio
async def test_duplicate_version_raises_integrity_error(
    db_session: AsyncSession,
    fake_extractor: FakePdfTextExtractor,
    fake_storage: FakeFileStorageProvider,
) -> None:
    document_id = await _insert_document(db_session)

    await extract_and_persist_spans(
        db_session,
        document_id=document_id,
        source_version=1,
        extractor=fake_extractor,
        storage=fake_storage,
    )
    await db_session.commit()

    with pytest.raises(IntegrityError):
        await extract_and_persist_spans(
            db_session,
            document_id=document_id,
            source_version=1,
            extractor=fake_extractor,
            storage=fake_storage,
        )


@pytest.mark.asyncio
async def test_empty_pdf_raises_empty_pdf_error(
    db_session: AsyncSession,
    empty_extractor: FakePdfTextExtractor,
    fake_storage: FakeFileStorageProvider,
) -> None:
    document_id = await _insert_document(db_session)

    with pytest.raises(EmptyPdfError):
        await extract_and_persist_spans(
            db_session,
            document_id=document_id,
            source_version=1,
            extractor=empty_extractor,
            storage=fake_storage,
        )


@pytest.mark.asyncio
async def test_missing_document_raises_lookup_error(
    db_session: AsyncSession,
    fake_extractor: FakePdfTextExtractor,
    fake_storage: FakeFileStorageProvider,
) -> None:
    with pytest.raises(LookupError):
        await extract_and_persist_spans(
            db_session,
            document_id="doc_does_not_exist",
            source_version=1,
            extractor=fake_extractor,
            storage=fake_storage,
        )
