"""Integration tests for the embedding stage (Phase 10.2).

Real Postgres (``db_session``): seeds a document + ready item + its chunks, then
exercises ``embed_chunks`` / ``re_embed_for_model`` against the real
``(chunk_id, embedding_provider, embedding_model)`` unique constraint — one
``ChunkEmbedding`` per chunk, ``dimensions == 1536``, same-model upsert preserving
``id``/``created_at`` while replacing the vector, and a different model inserting a
new row per chunk.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.jobs import _embed_document_chunks
from rag_recipes.ingestion.pipeline.chunking import persist_chunks_for_ready_items
from rag_recipes.ingestion.pipeline.embedding import embed_chunks, re_embed_for_model
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
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
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio

SOURCE_VERSION = 1
CATEGORY = "recipes"


class _RaisingEmbeddingProvider(EmbeddingProvider):
    """Always raises EmbeddingTechnicalError — scaffolding for the failure test."""

    async def embed_text(
        self, text: str, *, trace_context: TraceContext | None = None
    ) -> Embedding:
        raise EmbeddingTechnicalError("boom")

    async def embed_batch(
        self, texts: list[str], *, trace_context: TraceContext | None = None
    ) -> list[Embedding]:
        raise EmbeddingTechnicalError("boom")


async def _seed_document_with_chunks(session: AsyncSession) -> tuple[str, list[str]]:
    """Seed a document + ready item + its chunks; return (document_id, chunk_ids)."""
    repo = DocumentRepository(session)
    pdf_bytes = b"%PDF-1.4 embedding fixture"
    asset = await repo.add_source_asset(
        id=new_id("asset"),
        source_type=SourceType.PDF,
        original_filename="recipe.pdf",
        storage_provider="fake",
        storage_key=f"source-assets/{new_id('asset')}/original.pdf",
        content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
        upload_status=UploadStatus.UPLOADED,
    )
    document = await repo.add_document(
        asset_id=asset.id,
        category=CATEGORY,
        subcategory=None,
        title="My Recipe Book",
        author="Alice",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.CREATING_CHUNKS,
    )
    run = ExtractionRun(
        document_id=document.id,
        source_version=SOURCE_VERSION,
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=["span_1"],
        input_hash=hashlib.sha256(b"window").hexdigest(),
        status=ExtractionRunStatus.SUCCESS,
        output_json=None,
    )
    session.add(run)
    await session.flush()
    item = KnowledgeItem(
        document_id=document.id,
        extraction_run_id=run.id,
        source_version=SOURCE_VERSION,
        item_type="recipe",
        title="Tomato and White Bean Soup",
        normalized_title="tomato and white bean soup",
        summary="A hearty weeknight soup.",
        body_text="Tomato and White Bean Soup\n\nA hearty weeknight soup.",
        source_span_ids=["span_1"],
        structured_data={
            "ingredients_text": "2 tbsp olive oil\n1 onion",
            "ingredients": [{"raw_text": "2 tbsp olive oil"}],
            "steps_text": "1. Heat the oil.",
            "steps": [{"text": "Heat the oil."}],
        },
        status=KnowledgeItemStatus.READY,
    )
    session.add(item)
    await session.flush()
    count = await persist_chunks_for_ready_items(
        session, document_id=document.id, source_version=1, category=CATEGORY
    )
    assert count == 5
    chunk_ids = list(
        (
            await session.execute(
                select(Chunk.id).where(Chunk.document_id == document.id)
            )
        )
        .scalars()
        .all()
    )
    return document.id, chunk_ids


async def _count_embeddings(session: AsyncSession, chunk_ids: list[str]) -> int:
    return await session.scalar(  # type: ignore[return-value]
        select(func.count())
        .select_from(ChunkEmbedding)
        .where(ChunkEmbedding.chunk_id.in_(chunk_ids))
    )


async def test_embed_chunks_one_embedding_per_chunk_dims_1536(
    db_session: AsyncSession,
) -> None:
    document_id, chunk_ids = await _seed_document_with_chunks(db_session)
    chunks = list(
        (
            await db_session.execute(
                select(Chunk).where(Chunk.document_id == document_id)
            )
        )
        .scalars()
        .all()
    )

    rows = await embed_chunks(
        db_session, chunks, provider=FakeEmbeddingProvider(), batch_size=100
    )

    assert len(rows) == 5
    assert {r.chunk_id for r in rows} == set(chunk_ids)
    assert all(r.embedding_dimensions == 1536 for r in rows)
    assert all(len(r.embedding_vector) == 1536 for r in rows)
    assert all(r.embedding_provider == "fake" for r in rows)
    # Exactly one ChunkEmbedding per chunk for the configured provider/model.
    assert await _count_embeddings(db_session, chunk_ids) == 5


async def test_re_embed_same_provider_model_upserts_preserving_id(
    db_session: AsyncSession,
) -> None:
    document_id, chunk_ids = await _seed_document_with_chunks(db_session)
    target_chunk = chunk_ids[0]
    # Pre-insert a row with a known id and a placeholder zero vector, so re-embedding
    # the same (chunk, provider, model) must replace the vector while keeping id.
    placeholder = ChunkEmbedding(
        id="embedding_placeholder",
        chunk_id=target_chunk,
        embedding_provider="fake",
        embedding_model="fake-embedding",
        embedding_dimensions=1536,
        embedding_vector=[0.0] * 1536,
    )
    db_session.add(placeholder)
    await db_session.flush()
    original_created_at = placeholder.created_at

    await re_embed_for_model(
        db_session,
        document_id,
        provider=FakeEmbeddingProvider(),
        batch_size=100,
        trace_context=TraceContext(session_id=document_id),
    )

    # The Core upsert wrote through the ORM identity map; expire so the re-select
    # reads the persisted row rather than the stale in-session placeholder.
    db_session.expire_all()

    # Same-key upsert: row count for the target stays at one, id + created_at are
    # preserved, and the placeholder vector was replaced by the fake's vector.
    rows = list(
        (
            await db_session.execute(
                select(ChunkEmbedding).where(
                    ChunkEmbedding.chunk_id == target_chunk,
                    ChunkEmbedding.embedding_provider == "fake",
                    ChunkEmbedding.embedding_model == "fake-embedding",
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.id == "embedding_placeholder"
    assert row.created_at == original_created_at
    # The placeholder zero vector was replaced by the fake's (non-zero) vector.
    assert any(value != 0.0 for value in row.embedding_vector)
    # Every other chunk also got exactly one row (5 chunks total).
    assert await _count_embeddings(db_session, chunk_ids) == 5


async def test_re_embed_different_model_inserts_new_rows(
    db_session: AsyncSession,
) -> None:
    document_id, chunk_ids = await _seed_document_with_chunks(db_session)

    await re_embed_for_model(
        db_session,
        document_id,
        provider=FakeEmbeddingProvider(model="model-a"),
        batch_size=100,
    )
    assert await _count_embeddings(db_session, chunk_ids) == 5

    await re_embed_for_model(
        db_session,
        document_id,
        provider=FakeEmbeddingProvider(model="model-b"),
        batch_size=100,
    )
    # A different model inserts a fresh row per chunk: two rows per chunk now.
    assert await _count_embeddings(db_session, chunk_ids) == 10


async def test_embed_chunks_failure_persists_nothing(
    db_session: AsyncSession,
) -> None:
    document_id, chunk_ids = await _seed_document_with_chunks(db_session)
    chunks = list(
        (
            await db_session.execute(
                select(Chunk).where(Chunk.document_id == document_id)
            )
        )
        .scalars()
        .all()
    )

    with pytest.raises(EmbeddingTechnicalError):
        await embed_chunks(
            db_session, chunks, provider=_RaisingEmbeddingProvider(), batch_size=100
        )

    # embed_chunks raises before the upsert, so no rows were written for this run.
    assert await _count_embeddings(db_session, chunk_ids) == 0


def _single_session_factory(session: AsyncSession):  # type: ignore[no-untyped-def]
    """Yield the savepoint ``db_session`` to the stage and suppress its commit."""

    @asynccontextmanager
    async def _factory() -> AsyncIterator[AsyncSession]:
        original = session.commit
        session.commit = _noop  # type: ignore[method-assign]
        try:
            yield session
        finally:
            session.commit = original  # type: ignore[method-assign]

    async def _noop() -> None:
        return None

    return _factory


async def test_embed_stage_advances_stuck_job_heartbeat(
    db_session: AsyncSession,
) -> None:
    # Review #1: EMBEDDING_CHUNKS is no longer sweep-exempt, so the embedding stage
    # must advance last_progress_at — otherwise a long-but-healthy run with a stale
    # extraction heartbeat would be reaped. Seed a stale heartbeat, run the stage,
    # and assert it moved forward.
    document_id, _ = await _seed_document_with_chunks(db_session)
    await db_session.execute(
        text(
            "UPDATE documents SET last_progress_at = now() - make_interval(mins => 120) "
            "WHERE id = :id"
        ),
        {"id": document_id},
    )
    await db_session.flush()
    stale = await db_session.scalar(
        select(Document.last_progress_at).where(Document.id == document_id)
    )

    await _embed_document_chunks(
        _single_session_factory(db_session),
        document_id=document_id,
        source_version=1,
        provider=FakeEmbeddingProvider(),
        batch_size=100,
    )

    db_session.expire_all()
    document = await db_session.get(Document, document_id)
    assert document is not None
    assert document.status == DocumentStatus.EMBEDDING_CHUNKS
    assert document.last_progress_at is not None
    assert document.last_progress_at > stale
