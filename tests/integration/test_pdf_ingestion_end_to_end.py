"""Full pathway: upload PDF → poll status → burst worker → poll status again.

Distinct from ``test_process_document_job.py`` (which asserts the worker's
behaviour): this exercises the **status endpoint's polling shape** before and
after ingestion, and proves the Langfuse session_id seam end-to-end via a
recorder injected at the worker's ``on_startup``.

Like the 8.1 job test, this runs a real ``arq.worker.Worker`` reading from
Redis, so the document is committed to the test database (the worker's separate
connection must see it). Committed rows are cleaned up explicitly in teardown.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager
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
from rag_recipes.providers._observability import ProviderObservability
from rag_recipes.providers.file_storage.local import LocalFileStorage
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.ingestion_failure import IngestionFailure
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.failures import FailuresRepository
from rag_recipes.storage.session import build_session_factory
from tests.integration.conftest import AUTH_HEADERS

pytestmark = pytest.mark.asyncio

_FIXTURE = Path("data/fixtures/pdfs/sample_recipe.pdf")


class _LangfuseSessionRecorder:
    """Captures session_ids passed through the Langfuse session-scope seam.

    Satisfies the ``SessionScope`` protocol (``__call__(*, session_id) -> CM``)
    that ``ProviderObservability._session_scope`` holds and the job's
    ``langfuse_session_scope`` invokes. Recording on call (rather than on enter)
    is sufficient — the job always enters the scope it opens.
    """

    def __init__(self) -> None:
        self.session_ids: list[str] = []

    def __call__(self, *, session_id: str) -> AbstractContextManager[None]:
        self.session_ids.append(session_id)
        return contextlib.nullcontext()


async def _get_status(client: httpx.AsyncClient, document_id: str) -> dict[str, Any]:
    response = await client.get(f"/api/v1/documents/{document_id}/status")
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


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


async def test_full_upload_to_creating_source_spans_pathway(
    test_engine: AsyncEngine,
    redis_arq_settings: RedisSettings,
    arq_queue_cleanup: str,
    tmp_path: Path,
    override_settings_with_token: None,
) -> None:
    settings = get_settings().model_copy(update={"local_storage_root": str(tmp_path)})
    storage = LocalFileStorage(tmp_path)
    session_factory = build_session_factory(test_engine)
    recorder = _LangfuseSessionRecorder()

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    # A pool whose default queue is this test's isolated queue, so the route's
    # enqueue (which passes no _queue_name) lands where the worker reads.
    pool = await create_pool(redis_arq_settings, default_queue_name=arq_queue_cleanup)
    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = lambda: storage
    app.dependency_overrides[get_arq_redis] = lambda: pool

    async def _startup(ctx: dict[str, Any]) -> None:
        ctx["settings"] = settings
        ctx["session_factory"] = session_factory
        # Inject a disabled observability whose only wired collaborator is the
        # recorder session-scope. langfuse_session_scope reads _session_scope
        # directly, so this captures the session_id without a real Langfuse SDK.
        ctx["observability"] = ProviderObservability(
            None, enabled=False, session_scope=recorder
        )

    ids: dict[str, str] | None = None
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=AUTH_HEADERS
        ) as client:
            # Step 1: upload the synthetic PDF.
            response = await client.post(
                "/api/v1/documents",
                files={
                    "file": (
                        "sample_recipe.pdf",
                        _FIXTURE.read_bytes(),
                        "application/pdf",
                    )
                },
            )
            assert response.status_code == 201, response.text
            doc = response.json()["document"]
            ids = {"document_id": doc["id"], "asset_id": doc["asset_id"]}
            document_id = doc["id"]

            # Step 2: poll status before the worker runs — queued shape.
            before = await _get_status(client, document_id)
            assert before["status"] == "queued"
            assert before["current_source_version"] == 1
            assert before["active_source_version"] is None
            assert before["progress"]["pages_processed"] == 0
            assert before["progress"]["pages_total"] is None
            assert before["terminal"] is False

            # Step 3: run the in-process burst worker to completion.
            worker = Worker(
                functions=[func(process_document, name="process_document", max_tries=1)],
                redis_settings=redis_arq_settings,
                burst=True,
                max_jobs=1,
                queue_name=arq_queue_cleanup,
                on_startup=_startup,
                poll_delay=0.1,
            )
            try:
                await asyncio.wait_for(worker.async_run(), timeout=30)
            finally:
                await worker.close()

            # Step 4: poll status after the worker — creating_source_spans shape.
            after = await _get_status(client, document_id)
            assert after["status"] == "creating_source_spans"
            assert after["current_source_version"] == 1
            assert after["active_source_version"] is None
            assert after["progress"]["pages_processed"] == 3
            assert after["progress"]["pages_total"] == 3
            assert after["terminal"] is False

        # Step 5: the Langfuse session_id seam was driven with document.id.
        assert document_id in recorder.session_ids

        # Step 6 (defensive): 3 spans for v=1, and no failure row.
        async with session_factory() as session:
            spans = (
                await session.execute(
                    select(SourceSpan.id).where(
                        SourceSpan.document_id == document_id
                    )
                )
            ).scalars().all()
            assert len(spans) == 3
            failures = await FailuresRepository(session).list_failures(document_id)
            assert failures == []
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)
        app.dependency_overrides.pop(get_arq_redis, None)
        await pool.aclose()
        if ids is not None:
            await _cleanup(test_engine, ids)
