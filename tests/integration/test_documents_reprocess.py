"""Integration tests for POST /api/v1/documents/{document_id}/reprocess (Phase 6.3).

The route commits, so this module brings the standard
``after_transaction_end`` savepoint-restart listener; auth fails closed,
so each request sends a bearer token through the shared conftest
fixtures.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset


@pytest_asyncio.fixture
async def savepoint_session(db_session: AsyncSession) -> AsyncIterator[AsyncSession]:
    sync_session = db_session.sync_session

    @event.listens_for(sync_session, "after_transaction_end")
    def _restart_savepoint(sess: Any, trans: Any) -> None:
        if trans.nested and not trans._parent.nested:
            sess.begin_nested()

    try:
        yield db_session
    finally:
        event.remove(sync_session, "after_transaction_end", _restart_savepoint)


@pytest.fixture
def client(
    savepoint_session: AsyncSession,
    override_settings_with_token: None,
    auth_headers: dict[str, str],
) -> Iterator[httpx.AsyncClient]:
    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

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


async def _seed_document(
    session: AsyncSession,
    *,
    content_hash: str,
    status: DocumentStatus,
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
    doc = Document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="Example",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=active_source_version,
        status=status,
    )
    session.add(doc)
    await session.flush()
    await session.refresh(doc)
    return doc


# ---------------------------------------------------------------------------
# Happy path — all three modes from every terminal source status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_status",
    [DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW, DocumentStatus.FAILED],
)
@pytest.mark.parametrize(
    "mode,reason",
    [
        ("auto", None),
        ("reuse_source_spans", "investigating cost"),
        ("new_source_version", "new pdf"),
    ],
)
async def test_reprocess_happy_path_for_each_mode_and_terminal_status(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    source_status: DocumentStatus,
    mode: str,
    reason: str | None,
) -> None:
    doc = await _seed_document(
        savepoint_session,
        content_hash=f"hash-{source_status.value}-{mode}",
        status=source_status,
        active_source_version=1 if source_status is DocumentStatus.READY else None,
    )
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": mode, "reason": reason},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
        "document_id": doc.id,
        "status": "queued",
        "previous_active_source_version": doc.active_source_version,
        "current_source_version": doc.active_source_version,
    }

    # Verify the row was actually mutated in the DB.
    refreshed = (
        await savepoint_session.execute(
            select(Document).where(Document.id == doc.id)
        )
    ).scalar_one()
    await savepoint_session.refresh(refreshed)
    assert refreshed.status is DocumentStatus.QUEUED
    assert refreshed.last_reprocess_mode == mode
    assert refreshed.last_reprocess_reason == reason
    assert refreshed.active_source_version == doc.active_source_version


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_id_returns_404_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.post(
            "/api/v1/documents/doc_does_not_exist/reprocess",
            json={"mode": "auto"},
        )
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "document_not_found"


@pytest.mark.asyncio
async def test_invalid_mode_returns_400_envelope(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
) -> None:
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-invalid-mode",
        status=DocumentStatus.READY,
        active_source_version=1,
    )
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": "bogus"},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["details"]["field"] == "mode"
    assert body["error"]["details"]["value"] == "bogus"


@pytest.mark.asyncio
async def test_non_terminal_status_returns_409_envelope(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
) -> None:
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-already-queued",
        status=DocumentStatus.QUEUED,
    )
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": "auto"},
        )
    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "ingestion_already_running"


@pytest.mark.asyncio
async def test_concurrency_guard_zero_row_update_returns_409(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
) -> None:
    """The atomic guarded UPDATE closes the check-then-write race: when the
    row's status is no longer in the terminal set, the UPDATE matches 0 rows
    and the route 409s — the same path a losing concurrent POST would take.
    """
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-concurrency",
        status=DocumentStatus.EXTRACTING_TEXT,
    )
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": "auto"},
        )
    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "ingestion_already_running"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reprocess_401_without_token(
    savepoint_session: AsyncSession,
    override_settings_with_token: None,
) -> None:
    """A request without a bearer header is rejected by the auth gate even
    when a token is configured — proves the documents route stays gated.
    """
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-no-auth",
        status=DocumentStatus.READY,
        active_source_version=1,
    )

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            response = await c.post(
                f"/api/v1/documents/{doc.id}/reprocess",
                json={"mode": "auto"},
            )
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
