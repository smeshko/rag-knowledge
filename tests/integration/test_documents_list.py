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
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
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
def client(db_session: AsyncSession) -> Iterator[httpx.AsyncClient]:
    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        yield httpx.AsyncClient(transport=transport, base_url="http://testserver")
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
