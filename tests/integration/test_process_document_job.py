"""End-to-end PDF ingestion: upload route → arq enqueue → burst worker → spans.

Unlike the savepoint-isolated route tests, this exercises a real
``arq.worker.Worker`` reading from Redis, so the document must be committed to
the test database for the worker's separate connection to see it. The worker's
``on_startup`` is wired to the session-scoped test engine and a tmp storage
root; committed rows are cleaned up explicitly in teardown.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import Worker, func
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_arq_redis, get_file_storage, get_session
from rag_recipes.config import get_settings
from rag_recipes.ingestion.jobs import process_document
from rag_recipes.providers.file_storage.local import LocalFileStorage
from rag_recipes.storage.enums import DocumentStatus
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.ingestion_failure import IngestionFailure
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.failures import FailuresRepository
from rag_recipes.storage.session import build_session_factory
from tests.integration.conftest import AUTH_HEADERS

pytestmark = pytest.mark.asyncio

_FIXTURE = Path("data/fixtures/pdfs/sample_recipe.pdf")


class _EmptyExtractor:
    """Stand-in for PyMuPdfExtractor that returns no pages (drives pdf_empty)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def extract_pages(self, file: bytes) -> list[Any]:
        return []


async def _upload_and_run_worker(
    *,
    test_engine: AsyncEngine,
    redis_arq_settings: RedisSettings,
    queue_name: str,
    tmp_path: Path,
) -> dict[str, str]:
    """Upload the fixture PDF through the route, then run a burst worker.

    Returns the created ``{"document_id", "asset_id"}``. The route commits to
    the test engine; the caller is responsible for cleanup.
    """
    settings = get_settings().model_copy(update={"local_storage_root": str(tmp_path)})
    storage = LocalFileStorage(tmp_path)
    session_factory = build_session_factory(test_engine)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    # A pool whose default queue is this test's isolated queue, so the route's
    # enqueue (which passes no _queue_name) lands where the worker reads.
    pool = await create_pool(redis_arq_settings, default_queue_name=queue_name)
    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = lambda: storage
    app.dependency_overrides[get_arq_redis] = lambda: pool

    async def _startup(ctx: dict[str, Any]) -> None:
        ctx["settings"] = settings
        ctx["session_factory"] = session_factory

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=AUTH_HEADERS
        ) as client:
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("sample_recipe.pdf", _FIXTURE.read_bytes(), "application/pdf")},
            )
        assert response.status_code == 201, response.text
        doc = response.json()["document"]

        worker = Worker(
            functions=[func(process_document, name="process_document", max_tries=1)],
            redis_settings=redis_arq_settings,
            burst=True,
            max_jobs=1,
            queue_name=queue_name,
            on_startup=_startup,
            poll_delay=0.1,
        )
        try:
            await asyncio.wait_for(worker.async_run(), timeout=30)
        finally:
            await worker.close()
        return {"document_id": doc["id"], "asset_id": doc["asset_id"]}
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)
        app.dependency_overrides.pop(get_arq_redis, None)
        await pool.aclose()


async def _cleanup(test_engine: AsyncEngine, ids: dict[str, str]) -> None:
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        await session.execute(
            delete(SourceSpan).where(SourceSpan.document_id == ids["document_id"])
        )
        await session.execute(
            delete(IngestionFailure).where(
                IngestionFailure.document_id == ids["document_id"]
            )
        )
        await session.execute(delete(Document).where(Document.id == ids["document_id"]))
        await session.execute(delete(SourceAsset).where(SourceAsset.id == ids["asset_id"]))
        await session.commit()


async def test_process_document_writes_three_spans_and_transitions(
    test_engine: AsyncEngine,
    redis_arq_settings: RedisSettings,
    arq_queue_cleanup: str,
    tmp_path: Path,
    override_settings_with_token: None,
) -> None:
    ids = await _upload_and_run_worker(
        test_engine=test_engine,
        redis_arq_settings=redis_arq_settings,
        queue_name=arq_queue_cleanup,
        tmp_path=tmp_path,
    )
    try:
        session_factory = build_session_factory(test_engine)
        async with session_factory() as session:
            status = await session.scalar(
                select(Document.status).where(Document.id == ids["document_id"])
            )
            assert status == DocumentStatus.CREATING_SOURCE_SPANS

            spans = (
                await session.execute(
                    select(SourceSpan).where(
                        SourceSpan.document_id == ids["document_id"]
                    )
                )
            ).scalars().all()
            spans = sorted(spans, key=lambda s: s.locator["page_start"])
            assert [s.locator["page_start"] for s in spans] == [1, 2, 3]
            assert [s.locator["page_end"] for s in spans] == [1, 2, 3]
            assert [s.locator["meta"]["suspicious"] for s in spans] == [
                False,
                False,
                True,
            ]

            failures = await FailuresRepository(session).list_failures(
                ids["document_id"]
            )
            assert failures == []
    finally:
        await _cleanup(test_engine, ids)


async def test_process_document_marks_empty_pdf_failed(
    test_engine: AsyncEngine,
    redis_arq_settings: RedisSettings,
    arq_queue_cleanup: str,
    tmp_path: Path,
    override_settings_with_token: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch the import site (jobs.py), not the definition site, so the job's
    # `extractor = PyMuPdfExtractor(...)` resolves to the empty stand-in.
    monkeypatch.setattr(
        "rag_recipes.ingestion.jobs.PyMuPdfExtractor", _EmptyExtractor
    )

    ids = await _upload_and_run_worker(
        test_engine=test_engine,
        redis_arq_settings=redis_arq_settings,
        queue_name=arq_queue_cleanup,
        tmp_path=tmp_path,
    )
    try:
        session_factory = build_session_factory(test_engine)
        async with session_factory() as session:
            status = await session.scalar(
                select(Document.status).where(Document.id == ids["document_id"])
            )
            assert status == DocumentStatus.FAILED

            failures = await FailuresRepository(session).list_failures(
                ids["document_id"]
            )
            assert len(failures) == 1
            assert failures[0].reason == "pdf_empty"
    finally:
        await _cleanup(test_engine, ids)
