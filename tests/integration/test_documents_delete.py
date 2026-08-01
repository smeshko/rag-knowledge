"""Integration tests for DELETE /api/v1/documents/{document_id} (Phase 21.2).

Modeled on ``test_documents_upload.py``, not ``test_documents_get.py``: the
route calls ``session.commit()``, which releases ``db_session``'s nested
savepoint and breaks the conftest rollback isolation — so ``get_session`` is
overridden with the local savepoint-restart ``savepoint_session`` fixture
(duplicated per repo convention; it is not in conftest).

Seeding and the gone-assertions are imported from
``test_document_delete_repository`` (settled in TASK-001 — import, don't
duplicate); the storage doubles come from ``test_documents_upload``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_file_storage, get_session
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.storage.enums import DocumentStatus
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from tests.integration.test_document_delete_repository import (
    SeededChain,
    _seed_chain,
    assert_document_rows_gone,
)
from tests.integration.test_documents_upload import (
    PDF_BYTES,
    FlakyDeleteStorage,
    SpyLocalFileStorage,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def savepoint_session(db_session: AsyncSession) -> AsyncIterator[AsyncSession]:
    """``db_session`` with a savepoint-restart listener.

    The route calls ``session.commit()``. The integration ``db_session``
    fixture binds a session to a connection inside an outer transaction
    and a nested savepoint that is rolled back at teardown. Without a
    savepoint-restart listener, the route's commit releases that
    savepoint and breaks test isolation. The listener re-opens a
    nested savepoint each time the previous one ends.
    """
    sync_session = db_session.sync_session

    @event.listens_for(sync_session, "after_transaction_end")
    def _restart_savepoint(sess: Any, trans: Any) -> None:
        if trans.nested and not trans._parent.nested:  # outer SAVEPOINT released
            sess.begin_nested()

    try:
        yield db_session
    finally:
        event.remove(sync_session, "after_transaction_end", _restart_savepoint)


@pytest.fixture
def app_storage(tmp_path: Path) -> SpyLocalFileStorage:
    return SpyLocalFileStorage(tmp_path)


@pytest.fixture
def client(
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
    override_settings_with_token: None,
    auth_headers: dict[str, str],
) -> Iterator[httpx.AsyncClient]:
    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    def _override_storage() -> FileStorageProvider:
        return app_storage

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = _override_storage
    transport = httpx.ASGITransport(app=app)
    try:
        yield httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=auth_headers,
        )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)


async def _seed_with_stored_object(
    session: AsyncSession,
    storage: SpyLocalFileStorage,
    suffix: str,
    *,
    status: DocumentStatus = DocumentStatus.READY,
) -> SeededChain:
    """Seed a full chain and store the actual PDF object behind its key —
    the seed helper only writes DB rows, so without this the "stored file
    removed" assertion would test an object that was never there."""
    chain = await _seed_chain(session, suffix, status=status)
    await storage.put_object(chain.storage_key, PDF_BYTES, "application/pdf")
    return chain


async def _count_items(session: AsyncSession, document_id: str) -> int:
    result = await session.execute(
        select(func.count())
        .select_from(KnowledgeItem)
        .where(KnowledgeItem.document_id == document_id)
    )
    return result.scalar_one()


async def test_delete_ready_document_cascades_and_removes_stored_file(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
) -> None:
    chain = await _seed_with_stored_object(savepoint_session, app_storage, "route_ok")
    assert await app_storage.exists(chain.storage_key) is True

    async with client:
        response = await client.delete(f"/api/v1/documents/{chain.document_id}")

    assert response.status_code == 204, response.text
    assert response.content == b""
    await assert_document_rows_gone(savepoint_session, chain)
    assert await app_storage.exists(chain.storage_key) is False
    assert chain.storage_key in app_storage.delete_calls


async def test_delete_unknown_id_returns_404_envelope(
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.delete("/api/v1/documents/doc_missing")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "document_not_found"
    assert body["error"]["details"]["document_id"] == "doc_missing"


async def test_second_delete_of_same_id_returns_404(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
) -> None:
    chain = await _seed_with_stored_object(savepoint_session, app_storage, "route_2x")

    async with client:
        first = await client.delete(f"/api/v1/documents/{chain.document_id}")
        second = await client.delete(f"/api/v1/documents/{chain.document_id}")

    assert first.status_code == 204
    assert second.status_code == 404
    assert second.json()["error"]["code"] == "document_not_found"


@pytest.mark.parametrize(
    "status",
    [DocumentStatus.QUEUED, DocumentStatus.EXTRACTING_ITEMS],
    ids=lambda status: status.value,
)
async def test_delete_non_terminal_document_returns_409_and_deletes_nothing(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
    status: DocumentStatus,
) -> None:
    chain = await _seed_with_stored_object(
        savepoint_session, app_storage, f"route_{status.value}", status=status
    )

    async with client:
        response = await client.delete(f"/api/v1/documents/{chain.document_id}")

    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "ingestion_already_running"
    assert body["error"]["details"]["document_id"] == chain.document_id
    assert body["error"]["details"]["status"] == status.value

    # Nothing was deleted: the document, its items, and the stored file remain.
    savepoint_session.expire_all()
    assert await savepoint_session.get(Document, chain.document_id) is not None
    assert await _count_items(savepoint_session, chain.document_id) == len(
        chain.item_ids
    )
    assert await app_storage.exists(chain.storage_key) is True
    assert app_storage.delete_calls == []


@pytest.mark.parametrize(
    "status",
    [DocumentStatus.NEEDS_REVIEW, DocumentStatus.FAILED],
    ids=lambda status: status.value,
)
async def test_delete_succeeds_for_all_terminal_statuses(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
    status: DocumentStatus,
) -> None:
    chain = await _seed_with_stored_object(
        savepoint_session, app_storage, f"route_{status.value}", status=status
    )

    async with client:
        response = await client.delete(f"/api/v1/documents/{chain.document_id}")

    assert response.status_code == 204
    await assert_document_rows_gone(savepoint_session, chain)


async def test_file_storage_failure_still_returns_204_and_logs_error(
    savepoint_session: AsyncSession,
    tmp_path: Path,
    override_settings_with_token: None,
    auth_headers: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D6: the DB cascade is committed first; a FileStorageError from
    ``delete_object`` is swallowed by ``_best_effort_delete`` (which logs via
    ``logger.exception`` — ERROR level, naming the key) and the route still
    returns 204. The orphaned file leaks; the rows stay gone."""
    flaky_storage = FlakyDeleteStorage(tmp_path)
    chain = await _seed_with_stored_object(
        savepoint_session, flaky_storage, "route_flaky"
    )
    # The Alembic migration that builds the test DB runs fileConfig with
    # disable_existing_loggers=True, which disables this route logger.
    # Re-enable it so _best_effort_delete's ERROR record is observable here.
    logging.getLogger("rag_recipes.api.routes.documents").disabled = False

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    def _override_storage() -> FileStorageProvider:
        return flaky_storage

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = _override_storage
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=auth_headers,
        ) as client:
            with caplog.at_level(logging.ERROR, logger="rag_recipes.api.routes.documents"):
                response = await client.delete(
                    f"/api/v1/documents/{chain.document_id}"
                )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)

    assert response.status_code == 204
    assert chain.storage_key in flaky_storage.delete_calls
    error_records = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR and chain.storage_key in record.getMessage()
    ]
    assert error_records, "expected an ERROR record naming the storage key"
    await assert_document_rows_gone(savepoint_session, chain)
