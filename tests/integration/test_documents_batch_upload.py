"""Integration tests for POST /api/v1/documents/batch (Epic 19.2)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_file_storage, get_session, get_settings
from rag_recipes.providers.file_storage.local import LocalFileStorage
from rag_recipes.storage.models.document import Document
from tests.integration.conftest import AUTH_HEADERS, TEST_API_TOKEN

PDF_BYTES = b"%PDF-1.4\n%batch cohort payload\n%%EOF\n"
PDF_BYTES_2 = b"%PDF-1.4\n%batch cohort payload two\n%%EOF\n"
NON_PDF_BYTES = b"not a pdf at all"


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


def _enable_anthropic_settings() -> None:
    real = get_settings()
    overridden = real.model_copy(
        update={
            "personal_api_token": TEST_API_TOKEN,
            "llm_provider": "anthropic",
            "anthropic_api_key": "sk-ant-test",
        }
    )
    app.dependency_overrides[get_settings] = lambda: overridden


def _disable_batch_settings() -> None:
    # Token set (auth passes) but llm_provider forced to openai → batch path
    # disabled. Force it explicitly rather than inheriting the ambient default, so a
    # local .env with LLM_PROVIDER=anthropic doesn't leave the Anthropic batch path
    # enabled (_anthropic_batch_enabled checks llm_provider == "anthropic").
    real = get_settings()
    overridden = real.model_copy(
        update={"personal_api_token": TEST_API_TOKEN, "llm_provider": "openai"}
    )
    app.dependency_overrides[get_settings] = lambda: overridden


@pytest.fixture
def app_storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(tmp_path)


@pytest.fixture
def client(
    savepoint_session: AsyncSession,
    app_storage: LocalFileStorage,
) -> Iterator[httpx.AsyncClient]:
    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = lambda: app_storage
    transport = httpx.ASGITransport(app=app)
    try:
        yield httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=AUTH_HEADERS
        )
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)
        app.dependency_overrides.pop(get_settings, None)


def _files(*payloads: tuple[str, bytes]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (name, data, "application/pdf")) for name, data in payloads]


@pytest.mark.asyncio
async def test_batch_upload_creates_and_enqueues_batch_mode(
    client: httpx.AsyncClient,
    fake_arq_redis: AsyncMock,
    savepoint_session: AsyncSession,
) -> None:
    _enable_anthropic_settings()
    async with client:
        response = await client.post(
            "/api/v1/documents/batch",
            files=_files(("a.pdf", PDF_BYTES), ("b.pdf", PDF_BYTES_2)),
            data={"category": "recipes"},
        )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["total"] == 2
    assert body["created"] == 2
    assert body["duplicates"] == 0
    assert body["errors"] == 0
    doc_ids = [item["document_id"] for item in body["items"]]
    assert all(doc_ids)

    # Every enqueue used batch_mode=True (and never the synchronous form).
    assert fake_arq_redis.enqueue_job.await_count == 2
    for call in fake_arq_redis.enqueue_job.await_args_list:
        assert call.args[0] == "process_document"
        assert call.kwargs["batch_mode"] is True

    # Documents were actually created at QUEUED.
    count = (
        await savepoint_session.execute(
            select(func.count()).select_from(Document).where(Document.id.in_(doc_ids))
        )
    ).scalar_one()
    assert count == 2


@pytest.mark.asyncio
async def test_batch_upload_mixed_cohort(
    client: httpx.AsyncClient,
    fake_arq_redis: AsyncMock,
) -> None:
    _enable_anthropic_settings()
    async with client:
        response = await client.post(
            "/api/v1/documents/batch",
            files=[
                ("files", ("ok.pdf", PDF_BYTES, "application/pdf")),
                ("files", ("bad.txt", NON_PDF_BYTES, "text/plain")),
                ("files", ("dup.pdf", PDF_BYTES, "application/pdf")),
            ],
            data={"category": "recipes"},
        )
    assert response.status_code == 201, response.text
    body = response.json()
    by_name = {item["filename"]: item for item in body["items"]}
    assert by_name["ok.pdf"]["status"] == "created"
    assert by_name["bad.txt"]["status"] == "error"
    # Same content hash as ok.pdf → duplicate pointing at the existing doc.
    assert by_name["dup.pdf"]["status"] == "duplicate"
    assert by_name["dup.pdf"]["document_id"] == by_name["ok.pdf"]["document_id"]
    assert body["created"] == 1
    assert body["errors"] == 1
    assert body["duplicates"] == 1
    # Only the one fresh PDF was enqueued (batch mode); the bad/dup ones were not.
    assert fake_arq_redis.enqueue_job.await_count == 1


@pytest.mark.asyncio
async def test_batch_upload_rejected_when_provider_disabled(
    client: httpx.AsyncClient,
    fake_arq_redis: AsyncMock,
    savepoint_session: AsyncSession,
) -> None:
    _disable_batch_settings()
    async with client:
        response = await client.post(
            "/api/v1/documents/batch",
            files=_files(("a.pdf", PDF_BYTES)),
            data={"category": "recipes"},
        )
    assert response.status_code == 409, response.text
    # Nothing created, nothing enqueued.
    assert fake_arq_redis.enqueue_job.await_count == 0
    count = (
        await savepoint_session.execute(select(func.count()).select_from(Document))
    ).scalar_one()
    assert count == 0


@pytest.mark.asyncio
async def test_batch_upload_requires_token(
    savepoint_session: AsyncSession,
    app_storage: LocalFileStorage,
) -> None:
    _enable_anthropic_settings()

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield savepoint_session

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = lambda: app_storage
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:  # no auth header
            response = await c.post(
                "/api/v1/documents/batch",
                files=_files(("a.pdf", PDF_BYTES)),
                data={"category": "recipes"},
            )
        assert response.status_code == 401
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)
        app.dependency_overrides.pop(get_settings, None)


@pytest.mark.asyncio
async def test_single_upload_still_enqueues_synchronously(
    client: httpx.AsyncClient,
    fake_arq_redis: AsyncMock,
) -> None:
    # Regression: the helper refactor must keep POST /documents enqueuing the
    # synchronous job (no batch_mode kwarg).
    _enable_anthropic_settings()
    async with client:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("recipe.pdf", PDF_BYTES, "application/pdf")},
        )
    assert response.status_code == 201, response.text
    fake_arq_redis.enqueue_job.assert_awaited_once()
    call = fake_arq_redis.enqueue_job.await_args
    assert call.args[0] == "process_document"
    assert "batch_mode" not in call.kwargs
