"""Integration tests for DELETE /api/v1/knowledge-items/{item_id}.

The per-recipe hard delete. Before it, removing one bad recipe from a shelved
book meant ``DELETE /documents/{id}`` — throwing away the entire cookbook.

The load-bearing assertion in most of these is the *negative* one: what the
cascade must NOT touch. Everything above the item — extraction runs, source
spans, the stored PDF asset — belongs to the book and is shared with every
sibling recipe.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio

_BODY_TEXT = "Bean Stew\n\n" + ("a slow-simmered pot of beans for a cold evening. " * 6)

_STRUCTURED: dict[str, Any] = {
    "schema": "recipe.v1",
    "yield": "Serves 4",
    "ingredients": [{"position": 1, "raw_text": "1 cup dried beans"}],
    "steps": [{"position": 1, "text": "Simmer."}],
    "warnings": [],
}


@pytest.fixture
def client(
    db_session: AsyncSession,
    override_settings_with_token: None,
    auth_headers: dict[str, str],
) -> Iterator[httpx.AsyncClient]:
    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        yield httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=auth_headers
        )
    finally:
        app.dependency_overrides.pop(get_session, None)


async def _seed_document(
    session: AsyncSession, *, status: DocumentStatus = DocumentStatus.READY
) -> Document:
    repo = DocumentRepository(session)
    aid = new_id("asset")
    asset = await repo.add_source_asset(
        id=aid,
        source_type=SourceType.PDF,
        original_filename="cookbook.pdf",
        storage_provider="fake",
        storage_key=f"source-assets/{aid}/original.pdf",
        content_hash=hashlib.sha256(new_id("h").encode()).hexdigest(),
        upload_status=UploadStatus.UPLOADED,
    )
    return await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="Deletes Cookbook",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=1,
        status=status,
    )


async def _seed_run(session: AsyncSession, *, document_id: str) -> ExtractionRun:
    run = ExtractionRun(
        document_id=document_id,
        source_version=1,
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


async def _seed_span(session: AsyncSession, *, document_id: str) -> SourceSpan:
    text = "page text for the bean stew"
    locator = {"type": "pdf_page_range", "page_start": 12, "page_end": 12}
    span = SourceSpan(
        id=new_id("span"),
        document_id=document_id,
        source_version=1,
        source_type=SourceType.PDF,
        locator=locator,
        locator_hash=hashlib.sha256(str(locator).encode()).hexdigest(),
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    session.add(span)
    await session.flush()
    return span


async def _seed_item(
    session: AsyncSession,
    *,
    run: ExtractionRun,
    title: str = "Bean Stew",
    status: KnowledgeItemStatus = KnowledgeItemStatus.READY,
    span_ids: list[str] | None = None,
    indexed: bool = True,
) -> KnowledgeItem:
    item = KnowledgeItem(
        document_id=run.document_id,
        extraction_run_id=run.id,
        source_version=run.source_version,
        item_type="recipe",
        title=title,
        normalized_title=title.lower(),
        summary="A hearty stew.",
        body_text=_BODY_TEXT,
        source_span_ids=span_ids or [],
        structured_data=dict(_STRUCTURED),
        confidence={"overall": 0.9},
        status=status,
    )
    session.add(item)
    await session.flush()
    if indexed:
        for chunk_type in ("recipe_full", "recipe_title"):
            chunk = Chunk(
                document_id=run.document_id,
                parent_type="knowledge_item",
                parent_id=item.id,
                chunk_type=chunk_type,
                text=f"{chunk_type} text for {title}",
                text_hash=hashlib.sha256(
                    f"{chunk_type}-{item.id}".encode()
                ).hexdigest(),
                source_span_ids=span_ids or [],
                chunk_metadata={"category": "recipes"},
            )
            session.add(chunk)
            await session.flush()
            session.add(
                ChunkEmbedding(
                    chunk_id=chunk.id,
                    embedding_provider="fake",
                    embedding_model="fake-embedding",
                    embedding_dimensions=1536,
                    embedding_vector=[0.0] * 1536,
                )
            )
        await session.flush()
    return item


async def _count(session: AsyncSession, model: Any, *criteria: Any) -> int:
    session.expunge_all()
    total = await session.scalar(
        select(func.count()).select_from(model).where(*criteria)
    )
    return int(total or 0)


async def _index_row_counts(session: AsyncSession, item_id: str) -> tuple[int, int]:
    chunk_ids = select(Chunk.id).where(Chunk.parent_id == item_id)
    return (
        await _count(session, Chunk, Chunk.parent_id == item_id),
        await _count(session, ChunkEmbedding, ChunkEmbedding.chunk_id.in_(chunk_ids)),
    )


async def test_deleting_a_shelved_recipe_removes_it_and_its_index(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)
    assert await _index_row_counts(db_session, item.id) == (2, 2)

    async with client:
        resp = await client.delete(f"/api/v1/knowledge-items/{item.id}")

    assert resp.status_code == 204, resp.text
    assert resp.content == b""
    assert await _count(db_session, KnowledgeItem, KnowledgeItem.id == item.id) == 0
    assert await _index_row_counts(db_session, item.id) == (0, 0)


async def test_a_siblings_rows_and_the_books_own_rows_survive(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The cascade stops at the item. Everything above it is the book's."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    span = await _seed_span(db_session, document_id=doc.id)
    doomed = await _seed_item(db_session, run=run, title="Doomed", span_ids=[span.id])
    keeper = await _seed_item(db_session, run=run, title="Keeper", span_ids=[span.id])

    async with client:
        resp = await client.delete(f"/api/v1/knowledge-items/{doomed.id}")

    assert resp.status_code == 204, resp.text
    assert await _count(db_session, KnowledgeItem, KnowledgeItem.id == keeper.id) == 1
    assert await _index_row_counts(db_session, keeper.id) == (2, 2)
    assert await _count(db_session, Document, Document.id == doc.id) == 1
    assert await _count(db_session, ExtractionRun, ExtractionRun.id == run.id) == 1
    assert await _count(db_session, SourceSpan, SourceSpan.id == span.id) == 1
    assert await _count(db_session, SourceAsset, SourceAsset.id == doc.asset_id) == 1


async def test_the_books_counts_drop_by_one(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    doomed = await _seed_item(db_session, run=run, title="Doomed")
    await _seed_item(db_session, run=run, title="Keeper")

    async with client:
        before = await client.get(f"/api/v1/documents/{doc.id}")
        await client.delete(f"/api/v1/knowledge-items/{doomed.id}")
        after = await client.get(f"/api/v1/documents/{doc.id}")

    assert before.json()["counts"]["ready_items"] == 2
    assert after.json()["counts"]["ready_items"] == 1
    assert after.json()["counts"]["chunks"] == 2


@pytest.mark.parametrize(
    "status",
    [
        KnowledgeItemStatus.READY,
        KnowledgeItemStatus.NEEDS_REVIEW,
        KnowledgeItemStatus.SUPERSEDED,
        KnowledgeItemStatus.REJECTED,
        KnowledgeItemStatus.INDEXING,
    ],
)
async def test_every_item_status_is_deletable(
    client: httpx.AsyncClient, db_session: AsyncSession, status: KnowledgeItemStatus
) -> None:
    """No status guard, deliberately.

    A shelved recipe had no removal path at all before this, and ``rejected``
    — the review surface's soft delete — reaches only ``needs_review`` items.
    Deleting a mid-``indexing`` item is safe because the handler holds the
    document lock the job also takes.
    """
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(
        db_session, run=run, status=status, indexed=status is KnowledgeItemStatus.READY
    )

    async with client:
        resp = await client.delete(f"/api/v1/knowledge-items/{item.id}")

    assert resp.status_code == 204, resp.text
    assert await _count(db_session, KnowledgeItem, KnowledgeItem.id == item.id) == 0


async def test_an_unknown_item_is_404(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    async with client:
        resp = await client.delete("/api/v1/knowledge-items/item_missing")

    assert resp.status_code == 404
    body = resp.json()["error"]
    assert body["code"] == "knowledge_item_not_found"
    assert body["details"] == {"item_id": "item_missing"}


async def test_deleting_twice_is_404_the_second_time(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """204 must mean "this call removed it", not "it is absent now"."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        first = await client.delete(f"/api/v1/knowledge-items/{item.id}")
        second = await client.delete(f"/api/v1/knowledge-items/{item.id}")

    assert first.status_code == 204
    assert second.status_code == 404
    assert second.json()["error"]["code"] == "knowledge_item_not_found"


async def test_a_mid_reprocess_book_refuses_the_delete(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The pipeline is rewriting the book's items underneath us."""
    doc = await _seed_document(db_session, status=DocumentStatus.EXTRACTING_ITEMS)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.delete(f"/api/v1/knowledge-items/{item.id}")

    assert resp.status_code == 409
    body = resp.json()["error"]
    assert body["code"] == "ingestion_already_running"
    assert body["details"] == {
        "document_id": doc.id,
        "status": "extracting_items",
    }
    assert await _count(db_session, KnowledgeItem, KnowledgeItem.id == item.id) == 1


async def test_missing_token_returns_401(
    db_session: AsyncSession, override_settings_with_token: None
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            resp = await c.delete(f"/api/v1/knowledge-items/{item.id}")
    finally:
        app.dependency_overrides.pop(get_session, None)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
    assert await _count(db_session, KnowledgeItem, KnowledgeItem.id == item.id) == 1
