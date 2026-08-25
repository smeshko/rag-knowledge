"""Integration tests for GET /api/v1/documents/{document_id}/knowledge-items.

A book's contents, at every status — the listing ``GET /review-items`` cannot
give, because it hard-filters ``needs_review`` and so goes blank exactly when a
book is fully approved.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

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
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio

_BODY_TEXT = "Bean Stew\n\n" + ("a slow-simmered pot of beans for a cold evening. " * 6)

_STRUCTURED: dict[str, Any] = {
    "schema": "recipe.v1",
    "yield": "Serves 4",
    "ingredients": [
        {"position": 1, "raw_text": "1 cup dried beans", "item_normalized": "beans"}
    ],
    "steps": [{"position": 1, "text": "Simmer."}],
    "warnings": ["no_steps"],
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
    session: AsyncSession,
    *,
    status: DocumentStatus = DocumentStatus.READY,
    title: str = "Contents Cookbook",
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
        title=title,
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
    locator = {"type": "pdf_page_range", "page_start": 12, "page_end": 14}
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
    title: str,
    status: KnowledgeItemStatus = KnowledgeItemStatus.READY,
    span_ids: list[str] | None = None,
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
        confidence={"overall": 0.82},
        status=status,
    )
    session.add(item)
    await session.flush()
    return item


def _titles(resp: httpx.Response) -> set[str]:
    return {item["title"] for item in resp.json()["knowledge_items"]}


async def test_it_lists_ready_items_which_the_review_queue_never_returns(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The whole reason this endpoint exists."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    span = await _seed_span(db_session, document_id=doc.id)
    await _seed_item(db_session, run=run, title="Shelved Stew", span_ids=[span.id])

    async with client:
        resp = await client.get(f"/api/v1/documents/{doc.id}/knowledge-items")
        queue = await client.get("/api/v1/review-items", params={"document_id": doc.id})

    assert resp.status_code == 200, resp.text
    row = resp.json()["knowledge_items"][0]
    assert row["title"] == "Shelved Stew"
    assert row["status"] == "ready"
    assert row["document"] == {"id": doc.id, "title": "Contents Cookbook"}
    assert row["source_pages"] == {"page_start": 12, "page_end": 14}
    assert row["extraction"]["yield"] == "Serves 4"
    assert row["extraction"]["top_ingredients"] == ["beans"]
    # Warnings are only *projected* for needs_review items, so a shelved recipe
    # carrying stale warning codes does not sprout flags in the listing.
    assert row["flags"] == []
    assert row["edited_at"] is None

    assert queue.json()["review_items"] == []


async def test_mixed_statuses_are_listed_except_superseded_and_rejected(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    for title, status in [
        ("Ready One", KnowledgeItemStatus.READY),
        ("Flagged One", KnowledgeItemStatus.NEEDS_REVIEW),
        ("Indexing One", KnowledgeItemStatus.INDEXING),
        ("Extracting One", KnowledgeItemStatus.EXTRACTING),
        ("Superseded One", KnowledgeItemStatus.SUPERSEDED),
        ("Rejected One", KnowledgeItemStatus.REJECTED),
    ]:
        await _seed_item(db_session, run=run, title=title, status=status)

    async with client:
        resp = await client.get(f"/api/v1/documents/{doc.id}/knowledge-items")

    assert resp.status_code == 200, resp.text
    assert _titles(resp) == {"Ready One", "Flagged One", "Indexing One", "Extracting One"}


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ready", "Ready One"),
        ("needs_review", "Flagged One"),
        ("superseded", "Superseded One"),
        ("rejected", "Rejected One"),
    ],
)
async def test_an_explicit_status_filter_reaches_even_the_hidden_ones(
    client: httpx.AsyncClient, db_session: AsyncSession, status: str, expected: str
) -> None:
    """Hiding superseded/rejected by default must not make them unreachable."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    for title, item_status in [
        ("Ready One", KnowledgeItemStatus.READY),
        ("Flagged One", KnowledgeItemStatus.NEEDS_REVIEW),
        ("Superseded One", KnowledgeItemStatus.SUPERSEDED),
        ("Rejected One", KnowledgeItemStatus.REJECTED),
    ]:
        await _seed_item(db_session, run=run, title=title, status=item_status)

    async with client:
        resp = await client.get(
            f"/api/v1/documents/{doc.id}/knowledge-items", params={"status": status}
        )

    assert resp.status_code == 200, resp.text
    assert _titles(resp) == {expected}


async def test_another_books_items_are_not_listed(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    mine = await _seed_document(db_session, title="Mine")
    theirs = await _seed_document(db_session, title="Theirs")
    await _seed_item(
        db_session, run=await _seed_run(db_session, document_id=mine.id), title="Mine One"
    )
    await _seed_item(
        db_session,
        run=await _seed_run(db_session, document_id=theirs.id),
        title="Theirs One",
    )

    async with client:
        resp = await client.get(f"/api/v1/documents/{mine.id}/knowledge-items")

    assert _titles(resp) == {"Mine One"}


async def test_limit_and_offset_page_the_listing(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    for index in range(5):
        await _seed_item(db_session, run=run, title=f"Recipe {index}")

    async with client:
        first = await client.get(
            f"/api/v1/documents/{doc.id}/knowledge-items", params={"limit": "2"}
        )
        second = await client.get(
            f"/api/v1/documents/{doc.id}/knowledge-items",
            params={"limit": "2", "offset": "2"},
        )
        tail = await client.get(
            f"/api/v1/documents/{doc.id}/knowledge-items",
            params={"limit": "2", "offset": "4"},
        )

    assert len(first.json()["knowledge_items"]) == 2
    assert len(second.json()["knowledge_items"]) == 2
    # A short page is how the client knows to stop — there is no total.
    assert len(tail.json()["knowledge_items"]) == 1
    assert _titles(first).isdisjoint(_titles(second))


async def test_an_unknown_document_is_404_not_an_empty_list(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The document is addressed, not filtered — unlike /review-items."""
    async with client:
        resp = await client.get("/api/v1/documents/doc_missing/knowledge-items")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "document_not_found"


async def test_a_book_with_no_items_is_an_empty_list(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)

    async with client:
        resp = await client.get(f"/api/v1/documents/{doc.id}/knowledge-items")

    assert resp.status_code == 200
    assert resp.json() == {"knowledge_items": []}


async def test_a_reprocessing_book_is_still_readable(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """No terminal-status guard here: reading contents mid-reprocess is harmless.

    The review queue excludes non-terminal books because *deciding* their items
    is unsafe; the write verbs keep that guard, this read does not need it.
    """
    doc = await _seed_document(db_session, status=DocumentStatus.EXTRACTING_ITEMS)
    run = await _seed_run(db_session, document_id=doc.id)
    await _seed_item(db_session, run=run, title="Mid Reprocess")

    async with client:
        resp = await client.get(f"/api/v1/documents/{doc.id}/knowledge-items")

    assert resp.status_code == 200
    assert _titles(resp) == {"Mid Reprocess"}


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"status": "nonsense"}, "status"),
        ({"limit": "0"}, "limit"),
        ({"limit": "201"}, "limit"),
        ({"limit": "abc"}, "limit"),
        ({"offset": "-1"}, "offset"),
    ],
)
async def test_bad_query_params_use_the_error_envelope_not_a_raw_422(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    params: dict[str, str],
    field: str,
) -> None:
    doc = await _seed_document(db_session)

    async with client:
        resp = await client.get(
            f"/api/v1/documents/{doc.id}/knowledge-items", params=params
        )

    assert resp.status_code == 400
    body = resp.json()["error"]
    assert body["code"] == "invalid_request"
    assert body["details"]["field"] == field


async def test_missing_token_returns_401(
    db_session: AsyncSession, override_settings_with_token: None
) -> None:
    doc = await _seed_document(db_session)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            resp = await c.get(f"/api/v1/documents/{doc.id}/knowledge-items")
    finally:
        app.dependency_overrides.pop(get_session, None)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
