from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_file_storage, get_session
from rag_recipes.providers.errors import FileStorageError
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.providers.file_storage.local import LocalFileStorage
from rag_recipes.providers.file_storage.types import StoredObject
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.repositories.documents import DocumentRepository
from tests.integration.conftest import AUTH_HEADERS

PDF_BYTES = b"%PDF-1.4\n%minimal test pdf payload\n%%EOF\n"
NON_PDF_BYTES = b"not a pdf at all"


@pytest.fixture(autouse=True)
def _auto_token_override(override_settings_with_token: None) -> None:
    """Apply the fail-closed auth override to every documents-upload test."""


class SpyLocalFileStorage(LocalFileStorage):
    def __init__(self, root_path: Path) -> None:
        super().__init__(root_path)
        self.put_calls: list[str] = []
        self.delete_calls: list[str] = []

    async def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        self.put_calls.append(key)
        return await super().put_object(key, data, content_type)

    async def delete_object(self, key: str) -> None:
        self.delete_calls.append(key)
        await super().delete_object(key)


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


# ---------------------------------------------------------------------------
# Happy path & sequential duplicate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_happy_path_returns_doc_6_shape(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
    tmp_path: Path,
) -> None:
    async with client:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
            data={"category": "recipes", "title": "My Recipe", "author": "Alice"},
        )
    assert response.status_code == 201, response.text
    body = response.json()
    assert set(body.keys()) == {"document", "ingestion"}
    doc = body["document"]
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
    assert doc["title"] == "My Recipe"
    assert doc["author"] == "Alice"
    assert doc["category"] == "recipes"
    assert doc["status"] == "queued"
    assert doc["active_source_version"] is None
    assert doc["source_type"] == "pdf"
    assert doc["created_at"] is not None
    assert doc["updated_at"] is not None
    assert body["ingestion"] == {"status": "queued"}

    # File landed on disk under <tmp>/source-assets/{asset_id}/original.pdf.
    asset_id = doc["asset_id"]
    stored_path = tmp_path / "source-assets" / asset_id / "original.pdf"
    assert stored_path.is_file()
    assert stored_path.read_bytes() == PDF_BYTES
    assert app_storage.put_calls == [f"source-assets/{asset_id}/original.pdf"]

    # DB rows: source_asset has upload_status="uploaded", document has status=queued.
    asset = await savepoint_session.get(SourceAsset, asset_id)
    assert asset is not None
    assert asset.upload_status == UploadStatus.UPLOADED
    document = await savepoint_session.get(Document, doc["id"])
    assert document is not None
    assert document.status == DocumentStatus.QUEUED
    assert document.active_source_version is None


@pytest.mark.asyncio
async def test_sequential_duplicate_returns_existing_no_new_rows_or_store(
    client: httpx.AsyncClient,
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
) -> None:
    async with client:
        first = await client.post(
            "/api/v1/documents",
            files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
        )
        assert first.status_code == 201, first.text
        first_doc_id = first.json()["document"]["id"]
        first_asset_id = first.json()["document"]["asset_id"]

        second = await client.post(
            "/api/v1/documents",
            files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
        )
    assert second.status_code == 201
    second_doc = second.json()["document"]
    assert second_doc["id"] == first_doc_id
    assert second_doc["asset_id"] == first_asset_id

    # Storage was hit exactly once.
    assert len(app_storage.put_calls) == 1

    # Counts: exactly one row in each table.
    asset_count = await savepoint_session.execute(
        select(func.count()).select_from(SourceAsset)
    )
    document_count = await savepoint_session.execute(
        select(func.count()).select_from(Document)
    )
    assert asset_count.scalar_one() == 1
    assert document_count.scalar_one() == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_pdf_returns_415_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("not-a-pdf.txt", NON_PDF_BYTES, "text/plain")},
        )
    assert response.status_code == 415
    body = response.json()
    assert body["error"]["code"] == "unsupported_file_type"
    assert "message" in body["error"]


@pytest.mark.asyncio
async def test_missing_file_returns_400_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.post(
            "/api/v1/documents",
            data={"category": "recipes"},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_empty_file_bytes_returns_400_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# Defaults & passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_title_defaults_to_filename_stem_and_author_to_empty(
    client: httpx.AsyncClient, savepoint_session: AsyncSession
) -> None:
    async with client:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("my-cookbook.pdf", PDF_BYTES, "application/pdf")},
        )
    assert response.status_code == 201, response.text
    doc = response.json()["document"]
    assert doc["title"] == "my-cookbook"
    assert doc["author"] == ""
    assert doc["category"] == "recipes"


@pytest.mark.asyncio
async def test_provided_fields_are_persisted_verbatim(
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("ignored.pdf", PDF_BYTES, "application/pdf")},
            data={
                "category": "history",
                "subcategory": "medieval",
                "title": "Custom Title",
                "author": "Author Name",
                "language": "en",
            },
        )
    assert response.status_code == 201
    doc = response.json()["document"]
    assert doc["category"] == "history"
    assert doc["subcategory"] == "medieval"
    assert doc["title"] == "Custom Title"
    assert doc["author"] == "Author Name"
    assert doc["language"] == "en"


# ---------------------------------------------------------------------------
# Failure-injection tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def winner_seeded(savepoint_session: AsyncSession) -> dict[str, str]:
    """Pre-seed a winning SourceAsset+Document with PDF_BYTES' content_hash."""
    import hashlib

    content_hash = hashlib.sha256(PDF_BYTES).hexdigest()
    repo = DocumentRepository(savepoint_session)
    asset_id = new_id(SourceAsset.ID_PREFIX)
    await repo.add_source_asset(
        id=asset_id,
        source_type=SourceType.PDF,
        original_filename="winner.pdf",
        storage_provider="local",
        storage_key=f"source-assets/{asset_id}/original.pdf",
        content_hash=content_hash,
        upload_status=UploadStatus.UPLOADED,
    )
    document = await repo.add_document(
        asset_id=asset_id,
        category="recipes",
        subcategory=None,
        title="winner",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.QUEUED,
    )
    await savepoint_session.commit()
    return {"asset_id": asset_id, "document_id": document.id, "content_hash": content_hash}


class _IntegrityErrorRepo(DocumentRepository):
    """Override add_source_asset to raise IntegrityError after rollback-safe state."""

    async def add_source_asset(self, **kwargs: Any) -> SourceAsset:  # type: ignore[override]
        raise IntegrityError("statement", {}, Exception("duplicate key"))


@pytest.mark.asyncio
async def test_duplicate_race_recovery_deletes_orphan_and_returns_winner(
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
    winner_seeded: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject an IntegrityError on flush. The route must:

    * delete the orphaned key it just stored,
    * re-fetch the winner via content_hash + asset_id,
    * return the winner's document with 201, not 500.
    """
    # We need to bypass the upfront ``get_source_asset_by_content_hash`` hit
    # (which would short-circuit to the existing doc). So patch the
    # ``DocumentRepository`` constructor to return a repo whose
    # ``get_source_asset_by_content_hash`` first returns ``None`` (forcing the
    # insert path) then returns the actual winner on the recovery call.
    real_repo_cls = DocumentRepository

    class FlakyDuplicateLookupRepo(_IntegrityErrorRepo):
        def __init__(self, session: AsyncSession) -> None:
            super().__init__(session)
            self._calls = 0

        async def get_source_asset_by_content_hash(self, content_hash: str) -> SourceAsset | None:
            self._calls += 1
            if self._calls == 1:
                return None  # bypass pre-insert short-circuit
            # Recovery call → real DB lookup of the winner.
            return await real_repo_cls.get_source_asset_by_content_hash(self, content_hash)

    import rag_recipes.api.routes.documents as docs_route

    monkeypatch.setattr(docs_route, "DocumentRepository", FlakyDuplicateLookupRepo)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    def _override_storage() -> FileStorageProvider:
        return app_storage

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = _override_storage
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=AUTH_HEADERS,
        ) as c:
            response = await c.post(
                "/api/v1/documents",
                files={"file": ("loser.pdf", PDF_BYTES, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["document"]["id"] == winner_seeded["document_id"]
    assert body["document"]["asset_id"] == winner_seeded["asset_id"]
    # The orphan key (under the *loser's* asset id) was deleted.
    assert len(app_storage.put_calls) == 1
    assert len(app_storage.delete_calls) == 1
    assert app_storage.put_calls[0] == app_storage.delete_calls[0]
    # Winner's storage key (under the seeded asset id) was never touched
    # via storage (it was inserted via DB only in the fixture).


class _NonIntegrityErrorRepo(DocumentRepository):
    """Override add_document to raise a non-IntegrityError during flush."""

    async def add_document(self, **kwargs: Any) -> Document:  # type: ignore[override]
        raise RuntimeError("flush boom")


@pytest.mark.asyncio
async def test_post_storage_precommit_failure_deletes_orphan_and_returns_500(
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_recipes.api.routes.documents as docs_route

    monkeypatch.setattr(docs_route, "DocumentRepository", _NonIntegrityErrorRepo)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    def _override_storage() -> FileStorageProvider:
        return app_storage

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = _override_storage
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=AUTH_HEADERS,
        ) as c:
            response = await c.post(
                "/api/v1/documents",
                files={"file": ("loser.pdf", PDF_BYTES, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)

    assert response.status_code == 500, response.text
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    # Orphan was deleted.
    assert len(app_storage.put_calls) == 1
    assert app_storage.delete_calls == app_storage.put_calls


@pytest.mark.asyncio
async def test_commit_ambiguous_keeps_file_and_returns_500(
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If session.commit() raises, do NOT delete the stored file."""
    original_commit = savepoint_session.commit

    async def _explode() -> None:
        raise RuntimeError("commit boom")

    monkeypatch.setattr(savepoint_session, "commit", _explode)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    def _override_storage() -> FileStorageProvider:
        return app_storage

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = _override_storage
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=AUTH_HEADERS,
        ) as c:
            response = await c.post(
                "/api/v1/documents",
                files={"file": ("loser.pdf", PDF_BYTES, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)
        monkeypatch.setattr(savepoint_session, "commit", original_commit)

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    # File stayed on disk: no delete_object call.
    assert app_storage.delete_calls == []
    assert len(app_storage.put_calls) == 1


class _DuplicateLookupErrorRepo(DocumentRepository):
    async def get_source_asset_by_content_hash(self, content_hash: str) -> SourceAsset | None:
        raise RuntimeError("db is down")


@pytest.mark.asyncio
async def test_pre_storage_duplicate_lookup_failure_returns_500(
    savepoint_session: AsyncSession,
    app_storage: SpyLocalFileStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_recipes.api.routes.documents as docs_route

    monkeypatch.setattr(docs_route, "DocumentRepository", _DuplicateLookupErrorRepo)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    def _override_storage() -> FileStorageProvider:
        return app_storage

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = _override_storage
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=AUTH_HEADERS,
        ) as c:
            response = await c.post(
                "/api/v1/documents",
                files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    # Nothing was stored, nothing was deleted.
    assert app_storage.put_calls == []
    assert app_storage.delete_calls == []


class FailingStorage(LocalFileStorage):
    def __init__(self, root_path: Path) -> None:
        super().__init__(root_path)
        self.delete_calls: list[str] = []

    async def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        raise FileStorageError("storage down")

    async def delete_object(self, key: str) -> None:
        self.delete_calls.append(key)


@pytest.mark.asyncio
async def test_pre_storage_put_object_failure_returns_500(
    savepoint_session: AsyncSession, tmp_path: Path
) -> None:
    failing_storage = FailingStorage(tmp_path)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    def _override_storage() -> FileStorageProvider:
        return failing_storage

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = _override_storage
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=AUTH_HEADERS,
        ) as c:
            response = await c.post(
                "/api/v1/documents",
                files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert failing_storage.delete_calls == []
    # Filesystem stayed empty.
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Cleanup-failure resilience (review round-2 #3)
# ---------------------------------------------------------------------------


class FlakyDeleteStorage(SpyLocalFileStorage):
    """put_object stores normally; delete_object always raises."""

    async def delete_object(self, key: str) -> None:
        self.delete_calls.append(key)
        raise FileStorageError("delete failed")


@pytest.mark.asyncio
async def test_duplicate_race_recovery_survives_delete_failure(
    savepoint_session: AsyncSession,
    tmp_path: Path,
    winner_seeded: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delete_object failure during duplicate-race cleanup must not abort
    winner re-fetch. The orphan leaks (logged), but the user still gets the
    winning document with 201.
    """
    flaky_storage = FlakyDeleteStorage(tmp_path)
    real_repo_cls = DocumentRepository

    class FlakyDuplicateLookupRepo(_IntegrityErrorRepo):
        def __init__(self, session: AsyncSession) -> None:
            super().__init__(session)
            self._calls = 0

        async def get_source_asset_by_content_hash(self, content_hash: str) -> SourceAsset | None:
            self._calls += 1
            if self._calls == 1:
                return None
            return await real_repo_cls.get_source_asset_by_content_hash(self, content_hash)

    import rag_recipes.api.routes.documents as docs_route

    monkeypatch.setattr(docs_route, "DocumentRepository", FlakyDuplicateLookupRepo)

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
            headers=AUTH_HEADERS,
        ) as c:
            response = await c.post(
                "/api/v1/documents",
                files={"file": ("loser.pdf", PDF_BYTES, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["document"]["id"] == winner_seeded["document_id"]
    assert body["document"]["asset_id"] == winner_seeded["asset_id"]
    # Cleanup was attempted exactly once and failed silently.
    assert len(flaky_storage.put_calls) == 1
    assert len(flaky_storage.delete_calls) == 1
    assert flaky_storage.put_calls[0] == flaky_storage.delete_calls[0]


@pytest.mark.asyncio
async def test_post_storage_precommit_cleanup_failure_still_returns_500(
    savepoint_session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a non-IntegrityError fires during flush AND delete_object then
    fails, the route must still return the doc-6 500 envelope (not crash
    into the outer catch-all without rollback semantics).
    """
    flaky_storage = FlakyDeleteStorage(tmp_path)

    import rag_recipes.api.routes.documents as docs_route

    monkeypatch.setattr(docs_route, "DocumentRepository", _NonIntegrityErrorRepo)

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
            headers=AUTH_HEADERS,
        ) as c:
            response = await c.post(
                "/api/v1/documents",
                files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    # Cleanup was attempted once and failed silently.
    assert len(flaky_storage.put_calls) == 1
    assert len(flaky_storage.delete_calls) == 1


# ---------------------------------------------------------------------------
# Enqueue on fresh insert (Phase 8.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_enqueues_process_document_on_fresh_insert(
    client: httpx.AsyncClient,
    fake_arq_redis: AsyncMock,
) -> None:
    async with client:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
        )
    assert response.status_code == 201, response.text
    doc_id = response.json()["document"]["id"]
    fake_arq_redis.enqueue_job.assert_awaited_once_with(
        "process_document",
        doc_id,
        _job_id=None,
        _queue_name=None,
        _session_id=doc_id,
    )


@pytest.mark.asyncio
async def test_upload_does_not_enqueue_on_duplicate_recovery(
    client: httpx.AsyncClient,
    fake_arq_redis: AsyncMock,
) -> None:
    async with client:
        first = await client.post(
            "/api/v1/documents",
            files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
        )
        assert first.status_code == 201, first.text
        second = await client.post(
            "/api/v1/documents",
            files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
        )
    assert second.status_code == 201, second.text
    # Only the fresh insert enqueued; the duplicate-recovery path does not.
    assert fake_arq_redis.enqueue_job.await_count == 1


@pytest.mark.asyncio
async def test_upload_returns_201_even_when_enqueue_fails(
    client: httpx.AsyncClient,
    fake_arq_redis: AsyncMock,
    savepoint_session: AsyncSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_arq_redis.enqueue_job.side_effect = RuntimeError("redis down")
    # The Alembic migration that builds the test DB runs fileConfig with
    # disable_existing_loggers=True, which disables this route logger. Re-enable
    # it so the route's enqueue-failure warning is observable here.
    route_logger = logging.getLogger("rag_recipes.api.routes.documents")
    route_logger.disabled = False
    with caplog.at_level(logging.WARNING, logger="rag_recipes.api.routes.documents"):
        async with client:
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
            )
    assert response.status_code == 201, response.text
    doc_id = response.json()["document"]["id"]
    # The document row is committed despite the enqueue failure.
    document = await savepoint_session.get(Document, doc_id)
    assert document is not None
    assert document.status == DocumentStatus.QUEUED
    # The failure was logged for operator recovery.
    assert any(
        "Failed to enqueue process_document" in record.getMessage()
        for record in caplog.records
    )
