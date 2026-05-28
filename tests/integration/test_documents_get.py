"""Integration tests for GET /api/v1/documents/{document_id}."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session
from rag_recipes.storage.enums import (
    ChunkParentType,
    ChunkType,
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan


async def _seed_document(
    session: AsyncSession, *, content_hash: str
) -> Document:
    asset_id = new_id(SourceAsset.ID_PREFIX)
    session.add(
        SourceAsset(
            id=asset_id,
            source_type=SourceType.PDF,
            original_filename="example.pdf",
            storage_provider="local",
            storage_key=f"source-assets/{asset_id}/original.pdf",
            content_hash=content_hash,
            upload_status=UploadStatus.UPLOADED,
        )
    )
    await session.flush()
    document = Document(
        asset_id=asset_id,
        category="recipes",
        subcategory=None,
        title="Example",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.QUEUED,
    )
    session.add(document)
    await session.flush()
    await session.refresh(document)
    return document


async def _seed_source_span(
    session: AsyncSession, *, document: Document, source_version: int, locator: str
) -> SourceSpan:
    span = SourceSpan(
        document_id=document.id,
        source_version=source_version,
        source_type=SourceType.PDF,
        locator={"page": 1, "marker": locator},
        locator_hash=hashlib.sha256(locator.encode()).hexdigest(),
        text=f"text-{locator}",
        text_hash=hashlib.sha256(f"text-{locator}".encode()).hexdigest(),
    )
    session.add(span)
    await session.flush()
    return span


async def _seed_extraction_run(
    session: AsyncSession, *, document: Document, source_version: int
) -> ExtractionRun:
    run = ExtractionRun(
        document_id=document.id,
        source_version=source_version,
        provider="openai",
        model="gpt-4",
        prompt_version="v1",
        schema_version="v1",
        input_source_span_ids=[],
        input_hash=hashlib.sha256(f"run-{document.id}-{source_version}".encode()).hexdigest(),
        status=ExtractionRunStatus.SUCCESS,
    )
    session.add(run)
    await session.flush()
    return run


async def _seed_knowledge_item(
    session: AsyncSession,
    *,
    document: Document,
    run: ExtractionRun,
    status: KnowledgeItemStatus,
    title: str = "item",
) -> KnowledgeItem:
    item = KnowledgeItem(
        document_id=document.id,
        extraction_run_id=run.id,
        source_version=run.source_version,
        item_type="recipe",
        title=title,
        normalized_title=title.lower(),
        summary=None,
        body_text="body",
        source_span_ids=[],
        structured_data={},
        confidence=None,
        status=status,
    )
    session.add(item)
    await session.flush()
    return item


async def _seed_chunk(
    session: AsyncSession, *, document: Document, item: KnowledgeItem, label: str
) -> Chunk:
    chunk = Chunk(
        document_id=document.id,
        parent_type=ChunkParentType.KNOWLEDGE_ITEM,
        parent_id=item.id,
        chunk_type=ChunkType.RECIPE_FULL,
        text=f"chunk-{label}",
        text_hash=hashlib.sha256(f"chunk-{label}".encode()).hexdigest(),
        source_span_ids=[],
        chunk_metadata={},
    )
    session.add(chunk)
    await session.flush()
    return chunk


@pytest.fixture
def client(db_session: AsyncSession) -> Iterator[httpx.AsyncClient]:
    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        yield httpx.AsyncClient(transport=transport, base_url="http://testserver")
    finally:
        app.dependency_overrides.pop(get_session, None)


@pytest.mark.asyncio
async def test_get_returns_doc_section_4_shape_with_zero_counts(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_document(db_session, content_hash="hash-get-1")
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"document", "counts"}
    doc = body["document"]
    # Full DocumentResponse shape (doc §4 — same as upload's `document`).
    assert set(doc.keys()) == {
        "id",
        "asset_id",
        "category",
        "subcategory",
        "title",
        "author",
        "source_type",
        "language",
        "active_source_version",
        "status",
        "created_at",
        "updated_at",
    }
    assert doc["id"] == document.id
    counts = body["counts"]
    assert counts == {
        "source_spans": 0,
        "knowledge_items": 0,
        "ready_items": 0,
        "needs_review_items": 0,
        "chunks": 0,
    }


@pytest.mark.asyncio
async def test_get_counts_match_seeded_rows(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_document(db_session, content_hash="hash-counts-1")
    # Two source spans tied to the document.
    await _seed_source_span(db_session, document=document, source_version=1, locator="a")
    await _seed_source_span(db_session, document=document, source_version=1, locator="b")
    run = await _seed_extraction_run(db_session, document=document, source_version=1)
    # 4 knowledge items: 2 ready, 1 needs_review, 1 superseded.
    ready_a = await _seed_knowledge_item(
        db_session,
        document=document,
        run=run,
        status=KnowledgeItemStatus.READY,
        title="ready-a",
    )
    await _seed_knowledge_item(
        db_session,
        document=document,
        run=run,
        status=KnowledgeItemStatus.READY,
        title="ready-b",
    )
    await _seed_knowledge_item(
        db_session,
        document=document,
        run=run,
        status=KnowledgeItemStatus.NEEDS_REVIEW,
        title="nr-a",
    )
    await _seed_knowledge_item(
        db_session,
        document=document,
        run=run,
        status=KnowledgeItemStatus.SUPERSEDED,
        title="sup-a",
    )
    # 3 chunks (all under ready_a).
    await _seed_chunk(db_session, document=document, item=ready_a, label="c1")
    await _seed_chunk(db_session, document=document, item=ready_a, label="c2")
    await _seed_chunk(db_session, document=document, item=ready_a, label="c3")

    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}")
    assert response.status_code == 200, response.text
    counts = response.json()["counts"]
    assert counts == {
        "source_spans": 2,
        "knowledge_items": 4,  # incl. superseded
        "ready_items": 2,
        "needs_review_items": 1,
        "chunks": 3,
    }


@pytest.mark.asyncio
async def test_get_counts_do_not_bleed_across_documents(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc_a = await _seed_document(db_session, content_hash="hash-bleed-a")
    doc_b = await _seed_document(db_session, content_hash="hash-bleed-b")
    run_a = await _seed_extraction_run(db_session, document=doc_a, source_version=1)
    run_b = await _seed_extraction_run(db_session, document=doc_b, source_version=1)
    await _seed_knowledge_item(
        db_session,
        document=doc_a,
        run=run_a,
        status=KnowledgeItemStatus.READY,
        title="ready-a",
    )
    await _seed_knowledge_item(
        db_session,
        document=doc_b,
        run=run_b,
        status=KnowledgeItemStatus.NEEDS_REVIEW,
        title="nr-b",
    )

    async with client:
        response = await client.get(f"/api/v1/documents/{doc_a.id}")
    assert response.status_code == 200
    counts = response.json()["counts"]
    assert counts["knowledge_items"] == 1
    assert counts["ready_items"] == 1
    assert counts["needs_review_items"] == 0


@pytest.mark.asyncio
async def test_get_unknown_id_returns_404_envelope(
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.get("/api/v1/documents/doc_does_not_exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "document_not_found"
    assert "message" in body["error"]
