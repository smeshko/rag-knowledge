"""Integration tests for GET /api/v1/documents.

All endpoints under test are read-only; tests seed via the shared
``db_session`` + ``flush()`` so the route (with ``get_session`` overridden
to yield that same session) observes uncommitted rows. No savepoint
restart listener is needed because nothing commits.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
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
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset


async def _seed_document(
    session: AsyncSession,
    *,
    content_hash: str,
    category: str = "recipes",
    subcategory: str | None = None,
    title: str = "Example",
    author: str = "",
    status: DocumentStatus = DocumentStatus.QUEUED,
    language: str | None = None,
    active_source_version: int | None = None,
) -> Document:
    asset_id = new_id(SourceAsset.ID_PREFIX)
    asset = SourceAsset(
        id=asset_id,
        source_type=SourceType.PDF,
        original_filename="example.pdf",
        storage_provider="local",
        storage_key=f"source-assets/{asset_id}/original.pdf",
        content_hash=content_hash,
        upload_status=UploadStatus.UPLOADED,
    )
    session.add(asset)
    await session.flush()
    document = Document(
        asset_id=asset.id,
        category=category,
        subcategory=subcategory,
        title=title,
        author=author,
        source_type=SourceType.PDF,
        language=language,
        active_source_version=active_source_version,
        status=status,
    )
    session.add(document)
    await session.flush()
    await session.refresh(document)
    return document


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
            transport=transport,
            base_url="http://testserver",
            headers=auth_headers,
        )
    finally:
        app.dependency_overrides.pop(get_session, None)


# ---------------------------------------------------------------------------
# Shape & ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_list_returns_documents_key(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": []}


@pytest.mark.asyncio
async def test_list_returns_doc_section_3_shape(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_document(
        db_session,
        content_hash="hash-shape-1",
        title="Recipe Book",
        author="Alice",
        subcategory="dessert",
        language="en",
    )
    async with client:
        response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["documents"]
    assert len(body["documents"]) == 1
    item = body["documents"][0]
    # doc §3 list shape: smaller than doc §4 — no asset_id/language/timestamps
    assert set(item.keys()) == {
        "id",
        "category",
        "subcategory",
        "title",
        "author",
        "source_type",
        "status",
        "active_source_version",
    }
    assert item["title"] == "Recipe Book"
    assert item["author"] == "Alice"
    assert item["category"] == "recipes"
    assert item["subcategory"] == "dessert"
    assert item["source_type"] == "pdf"
    assert item["status"] == "queued"
    assert item["active_source_version"] is None


@pytest.mark.asyncio
async def test_list_orders_by_created_at_desc(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    first = await _seed_document(db_session, content_hash="hash-order-1", title="first")
    second = await _seed_document(db_session, content_hash="hash-order-2", title="second")
    third = await _seed_document(db_session, content_hash="hash-order-3", title="third")
    async with client:
        response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    ids = [d["id"] for d in response.json()["documents"]]
    # Newest first.
    assert ids[0] == third.id
    assert ids[-1] == first.id
    assert second.id in ids


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_category_filter_narrows_results(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_document(db_session, content_hash="hash-cat-1", category="recipes")
    await _seed_document(db_session, content_hash="hash-cat-2", category="history")
    async with client:
        response = await client.get("/api/v1/documents", params={"category": "recipes"})
    assert response.status_code == 200
    ids = [d["category"] for d in response.json()["documents"]]
    assert ids == ["recipes"]


@pytest.mark.asyncio
async def test_status_filter_narrows_results(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_document(
        db_session, content_hash="hash-st-1", status=DocumentStatus.QUEUED
    )
    await _seed_document(
        db_session, content_hash="hash-st-2", status=DocumentStatus.READY
    )
    async with client:
        response = await client.get("/api/v1/documents", params={"status": "ready"})
    assert response.status_code == 200
    docs = response.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["status"] == "ready"


@pytest.mark.asyncio
async def test_source_type_filter_narrows_results(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_document(db_session, content_hash="hash-srct-1")
    async with client:
        response = await client.get("/api/v1/documents", params={"source_type": "pdf"})
    assert response.status_code == 200
    docs = response.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["source_type"] == "pdf"


@pytest.mark.asyncio
async def test_combined_filters_and_together(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_document(
        db_session,
        content_hash="hash-cb-1",
        category="recipes",
        status=DocumentStatus.READY,
    )
    await _seed_document(
        db_session,
        content_hash="hash-cb-2",
        category="recipes",
        status=DocumentStatus.QUEUED,
    )
    await _seed_document(
        db_session,
        content_hash="hash-cb-3",
        category="history",
        status=DocumentStatus.READY,
    )
    async with client:
        response = await client.get(
            "/api/v1/documents",
            params={"category": "recipes", "status": "ready"},
        )
    assert response.status_code == 200
    docs = response.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["category"] == "recipes"
    assert docs[0]["status"] == "ready"


# ---------------------------------------------------------------------------
# Invalid filter envelope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_status_returns_400_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents", params={"status": "bogus"})
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert "message" in body["error"]


@pytest.mark.asyncio
async def test_invalid_source_type_returns_400_envelope(
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.get(
            "/api/v1/documents", params={"source_type": "docx"}
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_limit_caps_returned_rows(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    for i in range(5):
        await _seed_document(db_session, content_hash=f"hash-lim-{i}")
    async with client:
        response = await client.get("/api/v1/documents", params={"limit": "2"})
    assert response.status_code == 200
    assert len(response.json()["documents"]) == 2


@pytest.mark.asyncio
async def test_offset_skips_rows(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    docs = []
    for i in range(3):
        docs.append(await _seed_document(db_session, content_hash=f"hash-off-{i}"))
    # Order is newest-first: docs[2], docs[1], docs[0]
    async with client:
        response = await client.get(
            "/api/v1/documents", params={"limit": "10", "offset": "1"}
        )
    assert response.status_code == 200
    ids = [d["id"] for d in response.json()["documents"]]
    assert ids == [docs[1].id, docs[0].id]


@pytest.mark.asyncio
async def test_non_integer_limit_returns_400_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents", params={"limit": "abc"})
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert "error" in body


@pytest.mark.asyncio
async def test_non_integer_offset_returns_400_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents", params={"offset": "x"})
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_limit_zero_returns_400_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents", params={"limit": "0"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_negative_limit_returns_400_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents", params={"limit": "-1"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_limit_above_cap_returns_400_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents", params={"limit": "201"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_negative_offset_returns_400_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents", params={"offset": "-1"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# Phase 21.3 (D3): read-time review-status derivation + filter agreement
# ---------------------------------------------------------------------------


async def _seed_item_with_status(
    session: AsyncSession, *, document: Document, status: KnowledgeItemStatus
) -> KnowledgeItem:
    run = ExtractionRun(
        document_id=document.id,
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
    item = KnowledgeItem(
        document_id=document.id,
        extraction_run_id=run.id,
        source_version=1,
        item_type="recipe",
        title=f"Item {new_id('t')}",
        normalized_title="item",
        summary=None,
        body_text="body",
        source_span_ids=[],
        structured_data={"schema": "recipe.v1", "warnings": ["no_steps"]},
        confidence=None,
        status=status,
    )
    session.add(item)
    await session.flush()
    return item


@pytest.mark.asyncio
async def test_ready_doc_with_pending_item_lists_as_needs_review(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(
        db_session, content_hash="hash-derive-1", status=DocumentStatus.READY
    )
    item = await _seed_item_with_status(
        db_session, document=doc, status=KnowledgeItemStatus.NEEDS_REVIEW
    )
    async with client:
        before = await client.get("/api/v1/documents")
        assert before.json()["documents"][0]["status"] == "needs_review"

        # Decide the last pending item: the pill decays to ready — with no
        # pipeline-status write ever issued (the column stays READY throughout).
        item.status = KnowledgeItemStatus.REJECTED
        await db_session.flush()
        after = await client.get("/api/v1/documents")
        assert after.json()["documents"][0]["status"] == "ready"


@pytest.mark.asyncio
async def test_persisted_needs_review_doc_with_all_decided_lists_as_ready(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(
        db_session, content_hash="hash-derive-2", status=DocumentStatus.NEEDS_REVIEW
    )
    await _seed_item_with_status(
        db_session, document=doc, status=KnowledgeItemStatus.REJECTED
    )
    async with client:
        response = await client.get("/api/v1/documents")
    assert response.json()["documents"][0]["status"] == "ready"
    # No pipeline write: the persisted column is untouched (derivation only).
    await db_session.refresh(doc)
    assert doc.status is DocumentStatus.NEEDS_REVIEW


@pytest.mark.asyncio
async def test_status_filter_agrees_with_displayed_status(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    ready_with_pending = await _seed_document(
        db_session, content_hash="hash-fa-1", status=DocumentStatus.READY
    )
    await _seed_item_with_status(
        db_session,
        document=ready_with_pending,
        status=KnowledgeItemStatus.NEEDS_REVIEW,
    )
    plain_ready = await _seed_document(
        db_session, content_hash="hash-fa-2", status=DocumentStatus.READY
    )
    nr_all_decided = await _seed_document(
        db_session, content_hash="hash-fa-3", status=DocumentStatus.NEEDS_REVIEW
    )
    await _seed_item_with_status(
        db_session, document=nr_all_decided, status=KnowledgeItemStatus.REJECTED
    )
    nr_pending = await _seed_document(
        db_session, content_hash="hash-fa-4", status=DocumentStatus.NEEDS_REVIEW
    )
    await _seed_item_with_status(
        db_session, document=nr_pending, status=KnowledgeItemStatus.NEEDS_REVIEW
    )

    async with client:
        ready_resp = await client.get("/api/v1/documents", params={"status": "ready"})
        nr_resp = await client.get(
            "/api/v1/documents", params={"status": "needs_review"}
        )
    ready_docs = ready_resp.json()["documents"]
    nr_docs = nr_resp.json()["documents"]
    assert {d["id"] for d in ready_docs} == {plain_ready.id, nr_all_decided.id}
    assert {d["id"] for d in nr_docs} == {ready_with_pending.id, nr_pending.id}
    # The filter agrees exactly with the displayed status.
    assert all(d["status"] == "ready" for d in ready_docs)
    assert all(d["status"] == "needs_review" for d in nr_docs)


@pytest.mark.asyncio
async def test_non_terminal_and_failed_statuses_pass_through_underived(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    queued = await _seed_document(
        db_session, content_hash="hash-pt-1", status=DocumentStatus.QUEUED
    )
    await _seed_item_with_status(
        db_session, document=queued, status=KnowledgeItemStatus.NEEDS_REVIEW
    )
    failed = await _seed_document(
        db_session, content_hash="hash-pt-2", status=DocumentStatus.FAILED
    )
    await _seed_item_with_status(
        db_session, document=failed, status=KnowledgeItemStatus.NEEDS_REVIEW
    )
    async with client:
        listing = await client.get("/api/v1/documents")
        queued_resp = await client.get("/api/v1/documents", params={"status": "queued"})
        failed_resp = await client.get("/api/v1/documents", params={"status": "failed"})
    by_id = {d["id"]: d["status"] for d in listing.json()["documents"]}
    assert by_id[queued.id] == "queued"
    assert by_id[failed.id] == "failed"
    # Other status filters keep the plain column semantics.
    assert {d["id"] for d in queued_resp.json()["documents"]} == {queued.id}
    assert {d["id"] for d in failed_resp.json()["documents"]} == {failed.id}
