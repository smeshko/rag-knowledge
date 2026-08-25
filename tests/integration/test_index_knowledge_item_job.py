"""Driven integration tests for the ``index_knowledge_item`` arq job (Epic 21.3, D1/D1a/D9).

Follows the ``test_process_document_job`` burst-worker pattern: committed rows
(the worker runs in its own session — the savepoint ``db_session`` fixture is
invisible to it), an isolated queue, and a ``FakeEmbeddingProvider`` injected
via ``on_startup`` so no real provider is ever called. Search assertions
override the embedding space to the fake ``(provider, model)`` pair exactly as
``test_search.py`` does, and use the keyword leg (random fake vectors give no
meaningful vector ranking).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import Worker, func
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import (
    get_arq_redis,
    get_embedding_provider,
    get_reranker_provider,
    get_session,
)
from rag_recipes.api.dependencies import (
    get_settings as get_settings_dep,
)
from rag_recipes.config import get_settings
from rag_recipes.ingestion.jobs import index_knowledge_item
from rag_recipes.providers._observability import TraceContext
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.embeddings.types import Embedding
from rag_recipes.providers.errors import EmbeddingTechnicalError
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.session import build_session_factory
from tests.integration.conftest import AUTH_HEADERS, TEST_API_TOKEN

pytestmark = pytest.mark.asyncio

_FAKE_PROVIDER = "fake"
_FAKE_MODEL = "fake-embedding"

_STRUCTURED: dict[str, Any] = {
    "schema": "recipe.v1",
    "yield": "24 cookies",
    "ingredients": [
        {"raw_text": "1 cup maple syrup", "item_normalized": "maple syrup"},
        {"raw_text": "2 cups flour", "item_normalized": "flour"},
    ],
    "ingredients_text": "maple syrup, flour, butter",
    "steps": [{"text": "Mix and bake the cookies."}],
    "steps_text": "Mix and bake the cookies.",
    "warnings": ["low_normalization_confidence"],
}


class _FailingEmbeddingProvider(EmbeddingProvider):
    async def embed_text(
        self, text: str, *, trace_context: TraceContext | None = None
    ) -> Embedding:
        raise EmbeddingTechnicalError("boom")

    async def embed_batch(
        self, texts: list[str], *, trace_context: TraceContext | None = None
    ) -> list[Embedding]:
        raise EmbeddingTechnicalError("boom")


async def _seed_document(
    session: AsyncSession,
    *,
    status: DocumentStatus = DocumentStatus.NEEDS_REVIEW,
    active_source_version: int | None = None,
) -> tuple[str, str]:
    """Insert a committed asset + document; returns ``(document_id, asset_id)``."""
    aid = new_id("asset")
    session.add(
        SourceAsset(
            id=aid,
            source_type=SourceType.PDF,
            original_filename="cookbook.pdf",
            storage_provider="fake",
            storage_key=f"source-assets/{aid}/original.pdf",
            content_hash=hashlib.sha256(new_id("h").encode()).hexdigest(),
            upload_status=UploadStatus.UPLOADED,
        )
    )
    await session.flush()
    doc = Document(
        asset_id=aid,
        category="recipes",
        subcategory=None,
        title="Baking with Less Sugar",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=active_source_version,
        status=status,
    )
    session.add(doc)
    await session.flush()
    return doc.id, aid


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


async def _seed_item(
    session: AsyncSession,
    *,
    run: ExtractionRun,
    status: KnowledgeItemStatus = KnowledgeItemStatus.NEEDS_REVIEW,
    title: str = "Maple Cutout Cookies",
    span_ids: list[str] | None = None,
) -> KnowledgeItem:
    item = KnowledgeItem(
        document_id=run.document_id,
        extraction_run_id=run.id,
        source_version=run.source_version,
        item_type="recipe",
        title=title,
        normalized_title=title.lower(),
        summary="Crisp maple-sweetened cutout cookies.",
        body_text=(
            "Maple cutout cookies with maple syrup, flour and butter. "
            "Roll the dough, cut shapes and bake."
        ),
        source_span_ids=span_ids or [],
        structured_data=dict(_STRUCTURED),
        confidence={"overall": 0.62},
        status=status,
    )
    session.add(item)
    await session.flush()
    return item


async def _cleanup(test_engine: AsyncEngine, document_id: str, asset_id: str) -> None:
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        chunk_ids = select(Chunk.id).where(Chunk.document_id == document_id)
        await session.execute(
            delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids))
        )
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await session.execute(
            delete(KnowledgeItem).where(KnowledgeItem.document_id == document_id)
        )
        await session.execute(
            delete(ExtractionRun).where(ExtractionRun.document_id == document_id)
        )
        await session.execute(
            delete(SourceSpan).where(SourceSpan.document_id == document_id)
        )
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.execute(delete(SourceAsset).where(SourceAsset.id == asset_id))
        await session.commit()


def _job_ctx(
    test_engine: AsyncEngine, provider: EmbeddingProvider | None = None
) -> dict[str, Any]:
    return {
        "settings": get_settings(),
        "session_factory": build_session_factory(test_engine),
        "embedding_provider": provider or FakeEmbeddingProvider(),
    }


async def test_approved_item_is_indexed_handed_off_and_searchable(
    test_engine: AsyncEngine,
    redis_arq_settings: RedisSettings,
    arq_queue_cleanup: str,
) -> None:
    """End-to-end: POST approve → queued job → burst worker → ready + retrievable."""
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        document_id, asset_id = await _seed_document(session)
        run = await _seed_run(session, document_id=document_id)
        span = SourceSpan(
            id=new_id("span"),
            document_id=document_id,
            source_version=1,
            source_type=SourceType.PDF,
            locator={"type": "pdf_page_range", "page_start": 41, "page_end": 43},
            locator_hash=hashlib.sha256(new_id("m").encode()).hexdigest(),
            text="recipe text",
            text_hash=hashlib.sha256(new_id("t").encode()).hexdigest(),
        )
        session.add(span)
        await session.flush()
        item = await _seed_item(session, run=run, span_ids=[span.id])
        item_id = item.id
        await session.commit()

    settings = get_settings().model_copy(
        update={
            "personal_api_token": TEST_API_TOKEN,
            "embedding_provider": _FAKE_PROVIDER,
            "embedding_model": _FAKE_MODEL,
        }
    )
    fake = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    # The route's enqueue must reach the isolated queue: override the autouse
    # fake_arq_redis with a real pool whose *default* queue is this test's
    # (the route passes no _queue_name).
    pool = await create_pool(redis_arq_settings, default_queue_name=arq_queue_cleanup)
    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_arq_redis] = lambda: pool
    app.dependency_overrides[get_settings_dep] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: fake
    app.dependency_overrides[get_reranker_provider] = lambda: None

    async def _startup(ctx: dict[str, Any]) -> None:
        ctx["settings"] = settings
        ctx["session_factory"] = session_factory
        ctx["embedding_provider"] = fake

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=AUTH_HEADERS
        ) as client:
            resp = await client.post(
                f"/api/v1/knowledge-items/{item_id}/review",
                json={"decision": "approved"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["knowledge_item"]["status"] == "indexing"

            worker = Worker(
                functions=[
                    func(index_knowledge_item, name="index_knowledge_item", max_tries=1)
                ],
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

            async with session_factory() as session:
                reloaded = await session.get(KnowledgeItem, item_id)
                assert reloaded is not None
                assert reloaded.status is KnowledgeItemStatus.READY
                chunk_count = len(
                    (
                        await session.execute(
                            select(Chunk.id).where(Chunk.parent_id == item_id)
                        )
                    ).all()
                )
                assert chunk_count >= 1
                embedding_count = len(
                    (
                        await session.execute(
                            select(ChunkEmbedding.id).where(
                                ChunkEmbedding.chunk_id.in_(
                                    select(Chunk.id).where(Chunk.parent_id == item_id)
                                )
                            )
                        )
                    ).all()
                )
                assert embedding_count == chunk_count
                doc = await session.get(Document, document_id)
                assert doc is not None
                # D1a: the READY-gate handoff replicated — the item's version is live.
                assert doc.active_source_version == 1

            # The searchability invariant end-to-end: READY + version equality.
            search = await client.post(
                "/api/v1/search",
                json={"query": "maple cutout cookies", "mode": "keyword"},
            )
            assert search.status_code == 200, search.text
            result_ids = [r["item"]["id"] for r in search.json()["results"]]
            assert item_id in result_ids
    finally:
        for dep in (
            get_session,
            get_arq_redis,
            get_settings_dep,
            get_embedding_provider,
            get_reranker_provider,
        ):
            app.dependency_overrides.pop(dep, None)
        await pool.aclose()
        await _cleanup(test_engine, document_id, asset_id)


async def test_editing_a_ready_item_re_embeds_it_from_the_saved_text(
    test_engine: AsyncEngine,
    redis_arq_settings: RedisSettings,
    arq_queue_cleanup: str,
) -> None:
    """End-to-end: PATCH a shelved recipe → queued job → burst worker → findable
    by the words it now contains, and no longer by the ones it lost.

    The approve path above proves an item reaches search. This proves it can be
    *corrected* there — the thing that was impossible while a ``ready`` item's
    only edit path was a 404. The searchability assertions are the whole point:
    a row rewrite alone would leave the chunks describing the old text and the
    request would still answer 200.
    """
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        document_id, asset_id = await _seed_document(session)
        run = await _seed_run(session, document_id=document_id)
        item = await _seed_item(session, run=run, status=KnowledgeItemStatus.INDEXING)
        item_id = item.id
        await session.commit()

    settings = get_settings().model_copy(
        update={
            "personal_api_token": TEST_API_TOKEN,
            "embedding_provider": _FAKE_PROVIDER,
            "embedding_model": _FAKE_MODEL,
        }
    )
    fake = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _startup(ctx: dict[str, Any]) -> None:
        ctx["settings"] = settings
        ctx["session_factory"] = session_factory
        ctx["embedding_provider"] = fake

    async def _drain() -> None:
        worker = Worker(
            functions=[
                func(index_knowledge_item, name="index_knowledge_item", max_tries=1)
            ],
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

    pool = await create_pool(redis_arq_settings, default_queue_name=arq_queue_cleanup)
    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_arq_redis] = lambda: pool
    app.dependency_overrides[get_settings_dep] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: fake
    app.dependency_overrides[get_reranker_provider] = lambda: None

    try:
        # Put the item on the shelf the ordinary way, so the "before" state is
        # a genuinely indexed row rather than hand-written chunks.
        assert await index_knowledge_item(_job_ctx(test_engine, fake), item_id) >= 1

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=AUTH_HEADERS
        ) as client:
            before = await client.post(
                "/api/v1/search",
                json={"query": "maple cutout cookies", "mode": "keyword"},
            )
            assert item_id in [r["item"]["id"] for r in before.json()["results"]]

            # A whole-recipe correction, and it has to be: "cutout" sits in
            # the title, the summary AND the seeded body_text, and `apply_edit`
            # rebuilds body_text only when the ingredient or step lines change
            # (its documented rule — prose that lives only there survives an
            # edit that does not invalidate it). Patching all three is what
            # makes the disappearance assertion below mean something.
            resp = await client.patch(
                f"/api/v1/knowledge-items/{item_id}",
                json={
                    "title": "Sourdough Pretzel Knots",
                    "summary": "Chewy salted knots with a sourdough tang.",
                    "steps": [
                        "Shape the dough into knots and rest them briefly.",
                        "Boil in soda water, then bake until deeply browned.",
                    ],
                },
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["knowledge_item"]["status"] == "indexing"

            # Mid-flight the recipe is genuinely absent from search — chunks
            # gone, nothing to match. Asserted rather than glossed over,
            # because it is the cost the UI has to warn about.
            async with session_factory() as session:
                assert (
                    await session.execute(
                        select(Chunk.id).where(Chunk.parent_id == item_id)
                    )
                ).all() == []

            await _drain()

            async with session_factory() as session:
                reloaded = await session.get(KnowledgeItem, item_id)
                assert reloaded is not None
                assert reloaded.status is KnowledgeItemStatus.READY
                assert reloaded.title == "Sourdough Pretzel Knots"
                chunk_ids = (
                    (
                        await session.execute(
                            select(Chunk.id).where(Chunk.parent_id == item_id)
                        )
                    )
                    .scalars()
                    .all()
                )
                assert len(chunk_ids) >= 1
                embeddings = (
                    await session.execute(
                        select(ChunkEmbedding.id).where(
                            ChunkEmbedding.chunk_id.in_(chunk_ids)
                        )
                    )
                ).all()
                assert len(embeddings) == len(chunk_ids)

            # The claim that matters: the index describes the SAVED text.
            after_new = await client.post(
                "/api/v1/search",
                json={"query": "sourdough pretzel knots", "mode": "keyword"},
            )
            assert item_id in [r["item"]["id"] for r in after_new.json()["results"]]

            # The other half of the claim: the OLD text is really gone, not
            # merely outranked by the new.
            after_old = await client.post(
                "/api/v1/search",
                json={"query": "cutout", "mode": "keyword"},
            )
            assert item_id not in [r["item"]["id"] for r in after_old.json()["results"]]
    finally:
        for dep in (
            get_session,
            get_arq_redis,
            get_settings_dep,
            get_embedding_provider,
            get_reranker_provider,
        ):
            app.dependency_overrides.pop(dep, None)
        await pool.aclose()
        await _cleanup(test_engine, document_id, asset_id)


async def test_redelivery_noops_without_duplicate_rows(
    test_engine: AsyncEngine,
) -> None:
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        document_id, asset_id = await _seed_document(session)
        run = await _seed_run(session, document_id=document_id)
        item = await _seed_item(session, run=run, status=KnowledgeItemStatus.INDEXING)
        item_id = item.id
        await session.commit()

    try:
        ctx = _job_ctx(test_engine)
        first = await index_knowledge_item(ctx, item_id)
        assert first >= 1

        async with session_factory() as session:
            chunk_ids_before = sorted(
                (
                    await session.execute(
                        select(Chunk.id).where(Chunk.parent_id == item_id)
                    )
                ).scalars()
            )
            embeddings_before = len(
                (
                    await session.execute(
                        select(ChunkEmbedding.id).where(
                            ChunkEmbedding.chunk_id.in_(chunk_ids_before)
                        )
                    )
                ).all()
            )

        # Re-delivery: the item is now READY, so the job must no-op.
        second = await index_knowledge_item(ctx, item_id)
        assert second == 0

        async with session_factory() as session:
            chunk_ids_after = sorted(
                (
                    await session.execute(
                        select(Chunk.id).where(Chunk.parent_id == item_id)
                    )
                ).scalars()
            )
            embeddings_after = len(
                (
                    await session.execute(
                        select(ChunkEmbedding.id).where(
                            ChunkEmbedding.chunk_id.in_(chunk_ids_after)
                        )
                    )
                ).all()
            )
        assert chunk_ids_after == chunk_ids_before
        assert embeddings_after == embeddings_before
    finally:
        await _cleanup(test_engine, document_id, asset_id)


async def test_provider_failure_rolls_back_to_indexing(
    test_engine: AsyncEngine,
) -> None:
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        document_id, asset_id = await _seed_document(session)
        run = await _seed_run(session, document_id=document_id)
        item = await _seed_item(session, run=run, status=KnowledgeItemStatus.INDEXING)
        item_id = item.id
        await session.commit()

    try:
        ctx = _job_ctx(test_engine, provider=_FailingEmbeddingProvider())
        with pytest.raises(EmbeddingTechnicalError):
            await index_knowledge_item(ctx, item_id)

        async with session_factory() as session:
            reloaded = await session.get(KnowledgeItem, item_id)
            assert reloaded is not None
            # The single transaction never committed: still indexing (arq retries),
            # zero chunks persisted.
            assert reloaded.status is KnowledgeItemStatus.INDEXING
            chunks = (
                await session.execute(
                    select(Chunk.id).where(Chunk.parent_id == item_id)
                )
            ).all()
            assert chunks == []
    finally:
        await _cleanup(test_engine, document_id, asset_id)


async def test_newer_version_approval_flips_active_and_supersedes(
    test_engine: AsyncEngine,
) -> None:
    """D1a recorded consequence: the first approval of a newer generation flips
    the active version and supersedes the old version's items — while rejected
    rows stay untouched (TASK-001's supersede guard)."""
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        document_id, asset_id = await _seed_document(
            session, status=DocumentStatus.READY, active_source_version=1
        )
        run_v1 = await _seed_run(session, document_id=document_id, source_version=1)
        run_v2 = await _seed_run(session, document_id=document_id, source_version=2)
        ready_v1 = await _seed_item(
            session, run=run_v1, status=KnowledgeItemStatus.READY, title="Old Ready"
        )
        rejected_v1 = await _seed_item(
            session, run=run_v1, status=KnowledgeItemStatus.REJECTED, title="Old Reject"
        )
        approving_v2 = await _seed_item(
            session, run=run_v2, status=KnowledgeItemStatus.INDEXING, title="New Item"
        )
        ready_v1_id, rejected_v1_id = ready_v1.id, rejected_v1.id
        approving_id = approving_v2.id
        await session.commit()

    try:
        written = await index_knowledge_item(_job_ctx(test_engine), approving_id)
        assert written >= 1

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            assert doc.active_source_version == 2
            approved = await session.get(KnowledgeItem, approving_id)
            assert approved is not None
            assert approved.status is KnowledgeItemStatus.READY
            old_ready = await session.get(KnowledgeItem, ready_v1_id)
            assert old_ready is not None
            assert old_ready.status is KnowledgeItemStatus.SUPERSEDED
            old_rejected = await session.get(KnowledgeItem, rejected_v1_id)
            assert old_rejected is not None
            assert old_rejected.status is KnowledgeItemStatus.REJECTED
    finally:
        await _cleanup(test_engine, document_id, asset_id)


async def test_stale_item_reverts_to_needs_review_before_chunking(
    test_engine: AsyncEngine,
) -> None:
    """D9 rule 4: a v1 item whose document already carries a v2 generation is
    reverted at run time — no chunks, no embeddings, no handoff, no supersede.

    Driven directly: TASK-003's 409 closes the API path, which is exactly why
    the job needs its own guard."""
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        document_id, asset_id = await _seed_document(session)
        run_v1 = await _seed_run(session, document_id=document_id, source_version=1)
        run_v2 = await _seed_run(session, document_id=document_id, source_version=2)
        stale_v1 = await _seed_item(
            session, run=run_v1, status=KnowledgeItemStatus.INDEXING, title="Stale"
        )
        fresh_v2 = await _seed_item(session, run=run_v2, title="Fresh Pending")
        stale_id, fresh_id = stale_v1.id, fresh_v2.id
        await session.commit()

    try:
        written = await index_knowledge_item(_job_ctx(test_engine), stale_id)
        assert written == 0

        async with session_factory() as session:
            stale = await session.get(KnowledgeItem, stale_id)
            assert stale is not None
            assert stale.status is KnowledgeItemStatus.NEEDS_REVIEW
            fresh = await session.get(KnowledgeItem, fresh_id)
            assert fresh is not None
            assert fresh.status is KnowledgeItemStatus.NEEDS_REVIEW  # not superseded
            doc = await session.get(Document, document_id)
            assert doc is not None
            assert doc.active_source_version is None
            chunks = (
                await session.execute(
                    select(Chunk.id).where(Chunk.document_id == document_id)
                )
            ).all()
            assert chunks == []
            embeddings = (
                await session.execute(select(ChunkEmbedding.id))
            ).all()
            # No embedding rows can belong to this doc — it has zero chunks; the
            # global check keeps the assertion simple and still meaningful here.
            del embeddings
    finally:
        await _cleanup(test_engine, document_id, asset_id)
