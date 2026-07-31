"""End-to-end ingestion to READY (Phase 10.3 / Epic 10 capstone).

Uploads a document through ``POST /documents``, runs a real burst arq worker, and
asserts the document reaches the terminal ``ready`` status with chunks +
embeddings persisted and the four search indexes present. To make ``ready``
deterministic the source spans are seeded with explicit ids (``extract_and_persist_spans``
is no-oped) and a ``FakeLLMProvider`` is keyed by the rendered window prompt to
return one valid recipe (so extraction yields a ready ``KnowledgeItem``); a
``FakeEmbeddingProvider`` produces the 1536-dim vectors — mirroring the Risks/
DECISIONS provider-injection approach.
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
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_arq_redis, get_file_storage, get_session
from rag_recipes.config import get_settings
from rag_recipes.ingestion.jobs import process_document
from rag_recipes.ingestion.pipeline.windows import build_windows
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.file_storage.local import LocalFileStorage
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.storage.enums import DocumentStatus
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.ingestion_failure import IngestionFailure
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.session import build_session_factory
from tests.integration.conftest import AUTH_HEADERS
from tests.integration.test_process_document_dedup import (
    _build_spans,
    _noop_extract,
    _recipe_output,
    _request_hash_for,
)

pytestmark = pytest.mark.asyncio

_FIXTURE = Path("data/fixtures/pdfs/sample_recipe.pdf")
_WINDOW_SIZE = 3
_OVERLAP = 1


async def _cleanup(
    session_factory: async_sessionmaker[AsyncSession], document_id: str, asset_id: str
) -> None:
    async with session_factory() as session:
        await session.execute(
            delete(ChunkEmbedding).where(
                ChunkEmbedding.chunk_id.in_(
                    select(Chunk.id).where(Chunk.document_id == document_id)
                )
            )
        )
        for model in (Chunk, KnowledgeItem, ExtractionRun, SourceSpan, IngestionFailure):
            await session.execute(delete(model).where(model.document_id == document_id))
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.execute(delete(SourceAsset).where(SourceAsset.id == asset_id))
        await session.commit()


async def test_e2e_post_documents_to_ready(
    test_engine: AsyncEngine,
    redis_arq_settings: RedisSettings,
    arq_queue_cleanup: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_settings_with_token: None,
) -> None:
    settings = get_settings().model_copy(
        update={
            "local_storage_root": str(tmp_path),
            "pdf_window_size_pages": _WINDOW_SIZE,
            "pdf_overlap_pages": _OVERLAP,
        }
    )
    storage = LocalFileStorage(tmp_path)
    session_factory = build_session_factory(test_engine)
    queue_name = arq_queue_cleanup

    # Seeded spans drive a deterministic recipe, so the worker must not run real
    # PDF extraction over them.
    monkeypatch.setattr(
        "rag_recipes.ingestion.jobs.extract_and_persist_spans", _noop_extract
    )

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    pool = await create_pool(redis_arq_settings, default_queue_name=queue_name)
    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_file_storage] = lambda: storage
    app.dependency_overrides[get_arq_redis] = lambda: pool

    document_id: str | None = None
    asset_id: str | None = None
    try:
        # Step 1: upload the document through the real route.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=AUTH_HEADERS
        ) as client:
            response = await client.post(
                "/api/v1/documents",
                files={
                    "file": ("sample_recipe.pdf", _FIXTURE.read_bytes(), "application/pdf")
                },
            )
        assert response.status_code == 201, response.text
        doc = response.json()["document"]
        document_id, asset_id = str(doc["id"]), str(doc["asset_id"])

        # Step 2: seed the source spans (page 3 carries the recipe) so extraction is
        # deterministic; the keyed fake returns one valid recipe per window.
        spans = _build_spans(document_id)
        async with session_factory() as session:
            session.add_all(spans)
            await session.commit()
        window_low, window_high = build_windows(spans, _WINDOW_SIZE, _OVERLAP)
        llm = FakeLLMProvider(
            responses_by_hash={
                _request_hash_for(window_low): _recipe_output(
                    cited_span_id="span_p3", overall=0.9
                ),
                _request_hash_for(window_high): _recipe_output(
                    cited_span_id="span_p3", overall=0.9
                ),
            }
        )

        async def _startup(ctx: dict[str, Any]) -> None:
            ctx["settings"] = settings
            ctx["session_factory"] = session_factory
            ctx["llm_provider"] = llm
            ctx["embedding_provider"] = FakeEmbeddingProvider()

        # Step 3: run the burst worker over the enqueued job.
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

        # Step 4: the document reached READY with chunks + embeddings, all visible
        # in a fresh session, and the four search indexes are present.
        async with session_factory() as session:
            document = await session.get(Document, document_id)
            assert document is not None
            assert document.status == DocumentStatus.READY
            assert document.active_source_version == 1

            chunk_ids = list(
                (
                    await session.execute(
                        select(Chunk.id).where(Chunk.document_id == document_id)
                    )
                )
                .scalars()
                .all()
            )
            assert len(chunk_ids) == 5

            embeddings = list(
                (
                    await session.execute(
                        select(ChunkEmbedding).where(
                            ChunkEmbedding.chunk_id.in_(chunk_ids)
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(embeddings) == 5  # one per chunk
            assert {e.chunk_id for e in embeddings} == set(chunk_ids)
            assert all(e.embedding_dimensions == 1536 for e in embeddings)
            assert all(len(e.embedding_vector) == 1536 for e in embeddings)

            indexes = {
                row[0]
                for row in (
                    await session.execute(
                        text(
                            "SELECT indexname FROM pg_indexes "
                            "WHERE tablename IN ('chunks', 'chunk_embeddings')"
                        )
                    )
                ).all()
            }
            assert {
                "ix_chunks_ts_vector",
                "ix_chunk_embeddings_hnsw",
                "ix_chunk_embeddings_provider_model",
            } <= indexes
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_file_storage, None)
        app.dependency_overrides.pop(get_arq_redis, None)
        await pool.aclose()
        if document_id and asset_id:
            await _cleanup(session_factory, document_id, asset_id)
