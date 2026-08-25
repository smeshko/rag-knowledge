"""Integration tests for the favourites surface.

``PUT``/``DELETE /api/v1/knowledge-items/{item_id}/favourite`` and
``GET /api/v1/favourites``.

Two assertions here are structural rather than behavioural and are the reason
the star lives in its own table: favouriting must not touch
``knowledge_items.updated_at`` (which ``sweep_stuck_indexing_items`` reads as
lifecycle progress), and deleting a recipe must take its star with it through
the FK cascade, with no code in the delete path that knows favourites exist.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
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
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.knowledge_item_favourite import KnowledgeItemFavourite
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio

_BODY_TEXT = "Bean Stew\n\n" + ("a slow-simmered pot of beans for a cold evening. " * 6)

_NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
_HOUR_AGO = _NOW - timedelta(hours=1)

_STRUCTURED: dict[str, Any] = {
    "schema": "recipe.v1",
    "yield": "Serves 4",
    "ingredients": [
        {"position": 1, "raw_text": "1 cup dried beans", "item_normalized": "beans"}
    ],
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
    session: AsyncSession,
    *,
    status: DocumentStatus = DocumentStatus.READY,
    title: str = "Favourites Cookbook",
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


async def _seed_favourite(
    session: AsyncSession, *, item_id: str, at: datetime
) -> KnowledgeItemFavourite:
    """A star with an explicit timestamp, for the ordering assertions."""
    favourite = KnowledgeItemFavourite(knowledge_item_id=item_id, created_at=at)
    session.add(favourite)
    await session.flush()
    return favourite


def _titles(resp: httpx.Response) -> list[str]:
    return [item["title"] for item in resp.json()["knowledge_items"]]


async def test_starring_a_recipe_puts_it_on_the_favourites_list(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    span = await _seed_span(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run, title="Bean Stew", span_ids=[span.id])

    async with client:
        before = await client.get("/api/v1/favourites")
        put = await client.put(f"/api/v1/knowledge-items/{item.id}/favourite")
        after = await client.get("/api/v1/favourites")

    assert before.status_code == 200, before.text
    assert before.json()["knowledge_items"] == []

    assert put.status_code == 200, put.text
    favourite = put.json()["favourite"]
    assert favourite["knowledge_item_id"] == item.id
    assert favourite["favourited_at"] is not None

    row = after.json()["knowledge_items"][0]
    # The favourites list is the SAME projection the shelf and the queue use,
    # so a card rendered from it needs no second request.
    assert row["title"] == "Bean Stew"
    assert row["document"] == {"id": doc.id, "title": "Favourites Cookbook"}
    assert row["source_pages"] == {"page_start": 12, "page_end": 14}
    assert row["favourited_at"] == favourite["favourited_at"]


async def test_the_star_shows_on_the_detail_and_the_book_listing(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Every surface reads the star, so no screen has to ask a second endpoint."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    starred = await _seed_item(db_session, run=run, title="Starred")
    plain = await _seed_item(db_session, run=run, title="Plain")

    async with client:
        put = await client.put(f"/api/v1/knowledge-items/{starred.id}/favourite")
        detail = await client.get(f"/api/v1/knowledge-items/{starred.id}")
        plain_detail = await client.get(f"/api/v1/knowledge-items/{plain.id}")
        listing = await client.get(f"/api/v1/documents/{doc.id}/knowledge-items")

    stamped = put.json()["favourite"]["favourited_at"]
    assert detail.json()["knowledge_item"]["favourited_at"] == stamped
    assert plain_detail.json()["knowledge_item"]["favourited_at"] is None

    by_title = {row["title"]: row for row in listing.json()["knowledge_items"]}
    assert by_title["Starred"]["favourited_at"] == stamped
    assert by_title["Plain"]["favourited_at"] is None


async def test_starring_twice_is_a_no_op_and_does_not_restamp(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The second PUT of a double-click must not reorder the list."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run, title="Bean Stew")

    async with client:
        first = await client.put(f"/api/v1/knowledge-items/{item.id}/favourite")
        second = await client.put(f"/api/v1/knowledge-items/{item.id}/favourite")
        listing = await client.get("/api/v1/favourites")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(listing.json()["knowledge_items"]) == 1

    rows = (
        await db_session.execute(
            select(func.count()).select_from(KnowledgeItemFavourite)
        )
    ).scalar_one()
    assert rows == 1


async def test_unstarring_is_idempotent_too(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """A retried DELETE must not be told its own success failed."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run, title="Bean Stew")

    async with client:
        await client.put(f"/api/v1/knowledge-items/{item.id}/favourite")
        first = await client.delete(f"/api/v1/knowledge-items/{item.id}/favourite")
        second = await client.delete(f"/api/v1/knowledge-items/{item.id}/favourite")
        listing = await client.get("/api/v1/favourites")

    assert first.status_code == 204
    assert second.status_code == 204
    assert listing.json()["knowledge_items"] == []


@pytest.mark.parametrize("method", ["put", "delete"])
async def test_an_unknown_recipe_is_a_404_on_either_write(
    client: httpx.AsyncClient, method: str
) -> None:
    async with client:
        resp = await client.request(
            method.upper(), "/api/v1/knowledge-items/item_missing/favourite"
        )

    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "knowledge_item_not_found"


async def test_the_list_is_newest_star_first_not_newest_recipe_first(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The two orders diverge the moment an OLD recipe is starred, and this
    surface answers "what did I save most recently".

    The stars are written with explicit timestamps rather than through two
    ``PUT``s: every statement in this test shares one transaction, and the
    ``created_at`` default is ``now()`` — transaction-start time — so two PUTs
    here would tie on a clock that never advances. In production each PUT is
    its own transaction. The tie-break (``knowledge_item_id DESC``) is what
    keeps even that degenerate case a stable page walk.
    """
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    older = await _seed_item(db_session, run=run, title="Older Recipe")
    newer = await _seed_item(db_session, run=run, title="Newer Recipe")
    # Starred in the opposite order to the recipes' own creation.
    await _seed_favourite(db_session, item_id=newer.id, at=_HOUR_AGO)
    await _seed_favourite(db_session, item_id=older.id, at=_NOW)

    async with client:
        listing = await client.get("/api/v1/favourites")

    assert _titles(listing) == ["Older Recipe", "Newer Recipe"]


async def test_every_status_can_be_starred_and_none_is_hidden(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The book listing hides superseded/rejected because nobody asked for them.
    Here somebody explicitly did — hiding would read as data loss."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    items = [
        await _seed_item(db_session, run=run, title=title, status=status)
        for title, status in [
            ("Ready One", KnowledgeItemStatus.READY),
            ("Flagged One", KnowledgeItemStatus.NEEDS_REVIEW),
            ("Superseded One", KnowledgeItemStatus.SUPERSEDED),
            ("Rejected One", KnowledgeItemStatus.REJECTED),
        ]
    ]

    async with client:
        for item in items:
            resp = await client.put(f"/api/v1/knowledge-items/{item.id}/favourite")
            assert resp.status_code == 200, resp.text
        listing = await client.get("/api/v1/favourites")

    assert set(_titles(listing)) == {
        "Ready One",
        "Flagged One",
        "Superseded One",
        "Rejected One",
    }


async def test_starring_leaves_the_recipe_row_untouched(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The whole reason the star is a separate table.

    ``sweep_stuck_indexing_items`` reclaims items on ``status == INDEXING AND
    updated_at < threshold``, reading ``updated_at`` as lifecycle progress. A
    ``knowledge_items.favourited_at`` column would let a reader's star — toggled
    repeatedly, even — push that timestamp forward and delay the sweep that
    exists to un-stick the item.
    """
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(
        db_session, run=run, title="Mid Flight", status=KnowledgeItemStatus.INDEXING
    )
    await db_session.commit()
    before = (
        await db_session.execute(
            select(KnowledgeItem.updated_at).where(KnowledgeItem.id == item.id)
        )
    ).scalar_one()

    async with client:
        await client.put(f"/api/v1/knowledge-items/{item.id}/favourite")
        await client.delete(f"/api/v1/knowledge-items/{item.id}/favourite")

    after = (
        await db_session.execute(
            select(KnowledgeItem.updated_at).where(KnowledgeItem.id == item.id)
        )
    ).scalar_one()
    assert after == before


async def test_deleting_the_recipe_takes_its_star_with_it(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The FK cascade, so neither delete path needs to learn favourites exist."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run, title="Bean Stew")
    await db_session.commit()

    async with client:
        await client.put(f"/api/v1/knowledge-items/{item.id}/favourite")
        deleted = await client.delete(f"/api/v1/knowledge-items/{item.id}")
        listing = await client.get("/api/v1/favourites")

    assert deleted.status_code == 204, deleted.text
    assert listing.json()["knowledge_items"] == []

    orphans = (
        await db_session.execute(
            select(func.count()).select_from(KnowledgeItemFavourite)
        )
    ).scalar_one()
    assert orphans == 0


async def test_the_page_walk_uses_the_same_limit_offset_contract(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    items = [
        await _seed_item(db_session, run=run, title=f"Recipe {index}")
        for index in range(3)
    ]

    async with client:
        for item in items:
            await client.put(f"/api/v1/knowledge-items/{item.id}/favourite")
        first = await client.get("/api/v1/favourites", params={"limit": "2"})
        second = await client.get(
            "/api/v1/favourites", params={"limit": "2", "offset": "2"}
        )

    assert len(first.json()["knowledge_items"]) == 2
    assert len(second.json()["knowledge_items"]) == 1
    assert set(_titles(first)) | set(_titles(second)) == {
        "Recipe 0",
        "Recipe 1",
        "Recipe 2",
    }


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "201"},
        {"limit": "many"},
        {"offset": "-1"},
    ],
)
async def test_bad_paging_is_the_shared_error_envelope(
    client: httpx.AsyncClient, params: dict[str, str]
) -> None:
    async with client:
        resp = await client.get("/api/v1/favourites", params=params)

    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_request"
