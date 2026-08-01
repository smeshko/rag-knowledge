"""Integration tests for GET /api/v1/review-items (Epic 21.3, contract §1)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session
from rag_recipes.api.review_reasons import SOFT_WARNING_MESSAGES
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

_STRUCTURED: dict[str, Any] = {
    "schema": "recipe.v1",
    "yield": "24 cookies",
    "ingredients": [
        {"raw_text": "1 cup maple syrup", "item_normalized": "maple syrup"},
        {"raw_text": "2 cups flour", "item_normalized": "flour"},
        {"raw_text": "1 stick butter", "item_normalized": "butter"},
    ],
    "warnings": ["low_normalization_confidence"],
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
    status: DocumentStatus = DocumentStatus.NEEDS_REVIEW,
    title: str = "Baking with Less Sugar",
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
        active_source_version=None,
        status=status,
    )


async def _seed_run(
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


async def _seed_span(
    session: AsyncSession,
    *,
    document_id: str,
    locator: dict[str, Any],
    source_version: int = 1,
) -> SourceSpan:
    marker = new_id("marker")
    span = SourceSpan(
        id=new_id("span"),
        document_id=document_id,
        source_version=source_version,
        source_type=SourceType.PDF,
        locator=locator,
        locator_hash=hashlib.sha256(marker.encode()).hexdigest(),
        text=f"text-{marker}",
        text_hash=hashlib.sha256(f"text-{marker}".encode()).hexdigest(),
    )
    session.add(span)
    await session.flush()
    return span


async def _seed_item(
    session: AsyncSession,
    *,
    document_id: str,
    run: ExtractionRun,
    status: KnowledgeItemStatus = KnowledgeItemStatus.NEEDS_REVIEW,
    title: str = "Maple Cutout Cookies",
    span_ids: list[str] | None = None,
    structured_data: dict[str, Any] | None = None,
    confidence: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> KnowledgeItem:
    item = KnowledgeItem(
        document_id=document_id,
        extraction_run_id=run.id,
        source_version=run.source_version,
        item_type="recipe",
        title=title,
        normalized_title=title.lower(),
        summary="Crisp maple-sweetened cutout cookies.",
        body_text="body " * 30,
        source_span_ids=span_ids or [],
        structured_data=(
            structured_data if structured_data is not None else dict(_STRUCTURED)
        ),
        confidence=confidence if confidence is not None else {"overall": 0.62},
        status=status,
    )
    if created_at is not None:
        item.created_at = created_at
    session.add(item)
    await session.flush()
    return item


@pytest.mark.asyncio
async def test_listing_returns_contract_shape(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    span_a = await _seed_span(
        db_session,
        document_id=doc.id,
        locator={"type": "pdf_page_range", "page_start": 41, "page_end": 41},
    )
    span_b = await _seed_span(
        db_session,
        document_id=doc.id,
        locator={"type": "pdf_page_range", "page_start": 43, "page_end": 43},
    )
    item = await _seed_item(
        db_session, document_id=doc.id, run=run, span_ids=[span_a.id, span_b.id]
    )

    async with client:
        resp = await client.get("/api/v1/review-items")
    assert resp.status_code == 200, resp.text
    items = resp.json()["review_items"]
    assert len(items) == 1
    entry = items[0]
    assert entry["id"] == item.id
    assert entry["title"] == "Maple Cutout Cookies"
    assert entry["summary"] == "Crisp maple-sweetened cutout cookies."
    assert entry["item_type"] == "recipe"
    assert entry["document"] == {"id": doc.id, "title": "Baking with Less Sugar"}
    assert entry["source_pages"] == {"page_start": 41, "page_end": 43}
    assert entry["extraction"] == {
        "schema": "recipe.v1",
        "yield": "24 cookies",
        "top_ingredients": ["maple syrup", "flour", "butter"],
        "confidence_overall": 0.62,
    }
    # flags carry the exact 21.1 canonical copy — verbatim SOFT_WARNING_MESSAGES.
    assert entry["flags"] == [
        {
            "code": "low_normalization_confidence",
            "message": SOFT_WARNING_MESSAGES["low_normalization_confidence"],
        }
    ]


@pytest.mark.asyncio
async def test_non_pending_statuses_and_non_terminal_documents_are_excluded(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    terminal_doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=terminal_doc.id)
    for status in (
        KnowledgeItemStatus.READY,
        KnowledgeItemStatus.SUPERSEDED,
        KnowledgeItemStatus.EXTRACTING,
        KnowledgeItemStatus.REJECTED,
        KnowledgeItemStatus.INDEXING,
    ):
        await _seed_item(
            db_session, document_id=terminal_doc.id, run=run, status=status
        )
    pending = await _seed_item(db_session, document_id=terminal_doc.id, run=run)

    # Mid-reprocess (non-terminal) document: its pending item must not appear.
    queued_doc = await _seed_document(db_session, status=DocumentStatus.QUEUED)
    queued_run = await _seed_run(db_session, document_id=queued_doc.id)
    await _seed_item(db_session, document_id=queued_doc.id, run=queued_run)

    async with client:
        resp = await client.get("/api/v1/review-items")
    assert resp.status_code == 200, resp.text
    ids = [entry["id"] for entry in resp.json()["review_items"]]
    assert ids == [pending.id]


@pytest.mark.asyncio
async def test_failed_document_pending_items_appear(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """D9 recorded consequence: TERMINAL_STATUSES includes FAILED, so a failed
    document's pending items appear in the queue (its pill still reads failed)."""
    failed_doc = await _seed_document(db_session, status=DocumentStatus.FAILED)
    run = await _seed_run(db_session, document_id=failed_doc.id)
    item = await _seed_item(db_session, document_id=failed_doc.id, run=run)

    async with client:
        resp = await client.get("/api/v1/review-items")
    assert resp.status_code == 200, resp.text
    ids = [entry["id"] for entry in resp.json()["review_items"]]
    assert ids == [item.id]


@pytest.mark.asyncio
async def test_two_live_pending_generations_both_listed(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """D9: no read-time generation scoping — both generations' items are visible."""
    doc = await _seed_document(db_session)
    run_v1 = await _seed_run(db_session, document_id=doc.id, source_version=1)
    run_v2 = await _seed_run(db_session, document_id=doc.id, source_version=2)
    item_v1 = await _seed_item(db_session, document_id=doc.id, run=run_v1)
    item_v2 = await _seed_item(db_session, document_id=doc.id, run=run_v2)

    async with client:
        resp = await client.get("/api/v1/review-items")
    assert resp.status_code == 200, resp.text
    ids = {entry["id"] for entry in resp.json()["review_items"]}
    assert ids == {item_v1.id, item_v2.id}


@pytest.mark.asyncio
async def test_source_pages_tolerates_degraded_locators(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    # Locator missing page_end entirely — must not KeyError the listing.
    span = await _seed_span(
        db_session,
        document_id=doc.id,
        locator={"type": "pdf_page_range", "page_start": 7},
    )
    item = await _seed_item(db_session, document_id=doc.id, run=run, span_ids=[span.id])
    # And an item with no resolvable spans at all → both None.
    bare = await _seed_item(
        db_session, document_id=doc.id, run=run, title="Bare Item", span_ids=[]
    )

    async with client:
        resp = await client.get("/api/v1/review-items")
    assert resp.status_code == 200, resp.text
    by_id = {entry["id"]: entry for entry in resp.json()["review_items"]}
    assert by_id[item.id]["source_pages"] == {"page_start": 7, "page_end": None}
    assert by_id[bare.id]["source_pages"] == {"page_start": None, "page_end": None}


@pytest.mark.asyncio
async def test_document_id_filter_narrows_and_unknown_id_is_empty_200(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc_a = await _seed_document(db_session, title="Doc A")
    run_a = await _seed_run(db_session, document_id=doc_a.id)
    item_a = await _seed_item(db_session, document_id=doc_a.id, run=run_a)
    doc_b = await _seed_document(db_session, title="Doc B")
    run_b = await _seed_run(db_session, document_id=doc_b.id)
    await _seed_item(db_session, document_id=doc_b.id, run=run_b)

    async with client:
        narrowed = await client.get(
            "/api/v1/review-items", params={"document_id": doc_a.id}
        )
        unknown = await client.get(
            "/api/v1/review-items", params={"document_id": "doc_does_not_exist"}
        )
    assert narrowed.status_code == 200
    assert [e["id"] for e in narrowed.json()["review_items"]] == [item_a.id]
    assert unknown.status_code == 200
    assert unknown.json() == {"review_items": []}


@pytest.mark.asyncio
async def test_paging_is_newest_first_with_short_page_termination(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    items = []
    for i in range(3):
        items.append(
            await _seed_item(
                db_session,
                document_id=doc.id,
                run=run,
                title=f"Item {i}",
                created_at=datetime(2026, 8, 1, 10, i, 0, tzinfo=UTC),
            )
        )
    newest_first = [items[2].id, items[1].id, items[0].id]

    async with client:
        page_one = await client.get(
            "/api/v1/review-items", params={"limit": "2", "offset": "0"}
        )
        page_two = await client.get(
            "/api/v1/review-items", params={"limit": "2", "offset": "2"}
        )
    assert [e["id"] for e in page_one.json()["review_items"]] == newest_first[:2]
    # Short page terminates the walk (contract: no total count).
    assert [e["id"] for e in page_two.json()["review_items"]] == newest_first[2:]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "201"},
        {"limit": "abc"},
        {"offset": "-1"},
        {"offset": "abc"},
    ],
)
async def test_invalid_paging_params_yield_enveloped_400(
    client: httpx.AsyncClient, db_session: AsyncSession, params: dict[str, str]
) -> None:
    async with client:
        resp = await client.get("/api/v1/review-items", params=params)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["details"]["field"] in {"limit", "offset"}
