"""Integration tests for POST /api/v1/documents/{document_id}/reprocess (Phase 6.3).

The route commits, so this module brings the standard
``after_transaction_end`` savepoint-restart listener; auth fails closed,
so each request sends a bearer token through the shared conftest
fixtures.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan


async def _seed_span(
    session: AsyncSession, *, document_id: str, source_version: int, page: int = 1
) -> SourceSpan:
    """Seed one SourceSpan so max_source_version can resolve a version."""
    locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
    text = f"span page {page}"
    span = SourceSpan(
        id=new_id("span"),
        document_id=document_id,
        source_version=source_version,
        source_type=SourceType.PDF,
        locator=locator,
        locator_hash=hashlib.sha256(str(locator).encode()).hexdigest(),
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    session.add(span)
    await session.flush()
    return span


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
    fake_arq_redis: AsyncMock,
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
    # new_source_version targets max(spans)+1 = 1 here (these docs have no spans);
    # reuse/auto keep the active version. current_source_version reflects the target.
    expected_current = 1 if mode == "new_source_version" else doc.active_source_version
    assert body == {
        "document_id": doc.id,
        "status": "queued",
        "previous_active_source_version": doc.active_source_version,
        "current_source_version": expected_current,
    }

    # Verify the row was actually mutated in the DB. active_source_version is NOT
    # touched by the endpoint for any mode (the new-version flip is the worker's job).
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

    # Dispatch: new_source_version always enqueues (v=max+1); reuse enqueues only
    # when a version resolves (READY has active=1; NEEDS_REVIEW/FAILED have no
    # spans → nothing to reuse); auto is not dispatched until TASK-004.
    dispatched = mode == "new_source_version" or (
        mode == "reuse_source_spans" and doc.active_source_version is not None
    )
    if dispatched:
        fake_arq_redis.enqueue_job.assert_awaited_once()
    else:
        fake_arq_redis.enqueue_job.assert_not_awaited()


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
# Enqueue (Epic 11.1) — reuse_source_spans only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reuse_enqueues_process_document_with_resolved_active_version(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    fake_arq_redis: AsyncMock,
) -> None:
    """A reuse POST on a READY doc enqueues process_document once with the active
    version, reuse_source_spans=True, and the session_id passthrough."""
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-reuse-active",
        status=DocumentStatus.READY,
        active_source_version=1,
    )
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": "reuse_source_spans", "reason": "rerun"},
        )
    assert response.status_code == 200, response.text
    fake_arq_redis.enqueue_job.assert_awaited_once_with(
        "process_document",
        doc.id,
        _job_id=None,
        _queue_name=None,
        _session_id=doc.id,
        source_version=1,
        reuse_source_spans=True,
    )


@pytest.mark.asyncio
async def test_reuse_resolves_version_from_max_span_when_active_null(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    fake_arq_redis: AsyncMock,
) -> None:
    """A FAILED doc with active_source_version=None but existing v1 spans resolves
    source_version=1 via max_source_version."""
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-reuse-maxspan",
        status=DocumentStatus.FAILED,
        active_source_version=None,
    )
    await _seed_span(savepoint_session, document_id=doc.id, source_version=1)
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": "reuse_source_spans"},
        )
    assert response.status_code == 200, response.text
    fake_arq_redis.enqueue_job.assert_awaited_once_with(
        "process_document",
        doc.id,
        _job_id=None,
        _queue_name=None,
        _session_id=doc.id,
        source_version=1,
        reuse_source_spans=True,
    )


@pytest.mark.asyncio
async def test_reuse_with_no_spans_does_not_enqueue(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    fake_arq_redis: AsyncMock,
) -> None:
    """A reuse POST on a terminal doc with no active version and no spans enqueues
    no job (no fabricated source_version=1); the row still flips to QUEUED."""
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-reuse-nospans",
        status=DocumentStatus.FAILED,
        active_source_version=None,
    )
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": "reuse_source_spans"},
        )
    assert response.status_code == 200, response.text
    fake_arq_redis.enqueue_job.assert_not_awaited()
    refreshed = (
        await savepoint_session.execute(select(Document).where(Document.id == doc.id))
    ).scalar_one()
    await savepoint_session.refresh(refreshed)
    assert refreshed.status is DocumentStatus.QUEUED


@pytest.mark.asyncio
async def test_new_source_version_enqueues_fresh_version_run(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    fake_arq_redis: AsyncMock,
) -> None:
    """new_source_version enqueues a non-reuse run at max(spans)+1, leaves the
    active version untouched, and reports current_source_version = the new version.
    """
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-new-version",
        status=DocumentStatus.READY,
        active_source_version=1,
    )
    await _seed_span(savepoint_session, document_id=doc.id, source_version=1)
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": "new_source_version", "reason": "new pdf"},
        )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "document_id": doc.id,
        "status": "queued",
        "previous_active_source_version": 1,
        "current_source_version": 2,
    }
    fake_arq_redis.enqueue_job.assert_awaited_once_with(
        "process_document",
        doc.id,
        _job_id=None,
        _queue_name=None,
        _session_id=doc.id,
        source_version=2,
        reuse_source_spans=False,
    )
    # The endpoint does NOT flip active_source_version — the worker does on success.
    refreshed = (
        await savepoint_session.execute(select(Document).where(Document.id == doc.id))
    ).scalar_one()
    await savepoint_session.refresh(refreshed)
    assert refreshed.active_source_version == 1


@pytest.mark.asyncio
async def test_reprocess_resets_stale_last_progress_at(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
) -> None:
    """Reprocess clears the stale extraction heartbeat (review #3).

    A terminal doc carries a last_progress_at from its original run. If reprocess
    left it intact, sweep_stuck_jobs — which reaps on
    coalesce(last_progress_at, updated_at) — could mark the freshly-requeued job
    FAILED before the worker starts. The UPDATE must reset it to NULL so the
    just-bumped updated_at governs the pre-progress stage."""
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-stale-heartbeat",
        status=DocumentStatus.READY,
        active_source_version=1,
    )
    # Backdate the heartbeat far past any stuck-job timeout.
    await savepoint_session.execute(
        text(
            "UPDATE documents "
            "SET last_progress_at = now() - make_interval(mins => 10000) "
            "WHERE id = :id"
        ),
        {"id": doc.id},
    )
    await savepoint_session.flush()
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": "reuse_source_spans"},
        )
    assert response.status_code == 200, response.text
    refreshed = (
        await savepoint_session.execute(select(Document).where(Document.id == doc.id))
    ).scalar_one()
    await savepoint_session.refresh(refreshed)
    assert refreshed.status is DocumentStatus.QUEUED
    # The stale heartbeat is cleared, so the cron falls back to the fresh updated_at.
    assert refreshed.last_progress_at is None


@pytest.mark.asyncio
async def test_reuse_enqueue_failure_still_returns_200(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    fake_arq_redis: AsyncMock,
) -> None:
    """A Redis failure on enqueue is logged, not fatal: the row stays QUEUED and
    the response is still 200 (the stuck-job cron is the backstop)."""
    fake_arq_redis.enqueue_job.side_effect = RuntimeError("redis down")
    doc = await _seed_document(
        savepoint_session,
        content_hash="hash-reuse-enqueue-fail",
        status=DocumentStatus.READY,
        active_source_version=1,
    )
    async with client:
        response = await client.post(
            f"/api/v1/documents/{doc.id}/reprocess",
            json={"mode": "reuse_source_spans"},
        )
    assert response.status_code == 200, response.text
    refreshed = (
        await savepoint_session.execute(select(Document).where(Document.id == doc.id))
    ).scalar_one()
    await savepoint_session.refresh(refreshed)
    assert refreshed.status is DocumentStatus.QUEUED


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
