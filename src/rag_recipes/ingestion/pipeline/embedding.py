"""Embedding stage: turn persisted ``Chunk`` rows into ``ChunkEmbedding`` rows (doc 2 § 6).

Pipeline-pure: no arq/job concerns, no status transitions, no ``Settings`` — the
job layer (``ingestion.jobs``) owns transitions and provider construction.
``embed_chunks`` embeds chunk texts in ``batch_size`` groups via the configured
``EmbeddingProvider`` and upserts one ``ChunkEmbedding`` per chunk on the
``(chunk_id, embedding_provider, embedding_model)`` unique key, so re-embedding
the same provider/model replaces the row while a different model appends one.

The stored ``embedding_provider`` / ``embedding_model`` / ``embedding_dimensions``
/ ``embedding_vector`` are read straight off the returned ``Embedding`` (never
re-derived from ``Settings``), so a persisted row always matches the vector that
was actually produced. The caller owns the transaction: ``embed_chunks`` flushes
but never commits (mirroring ``extract_and_persist_spans`` / ``persist_chunks``),
so a mid-run failure persists nothing for that run.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.providers._observability import TraceContext
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.embeddings.types import Embedding
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding

__all__ = ["embed_chunks", "re_embed_for_model"]


def _batched(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    """Yield ``items`` in contiguous slices of at most ``size``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _embedding_values(chunk: Chunk, embedding: Embedding) -> dict[str, Any]:
    """Build the ``chunk_embeddings`` row values for one ``(chunk, Embedding)`` pair.

    Provider/model/dimensions/vector come from the ``Embedding`` so the row always
    matches the vector actually produced.
    """
    return {
        "id": new_id(ChunkEmbedding.ID_PREFIX),
        "chunk_id": chunk.id,
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
        "embedding_dimensions": embedding.dimensions,
        "embedding_vector": embedding.vector,
    }


async def embed_chunks(
    session: AsyncSession,
    chunks: list[Chunk],
    *,
    provider: EmbeddingProvider,
    batch_size: int,
    trace_context: TraceContext | None = None,
) -> list[ChunkEmbedding]:
    """Embed ``chunks`` and upsert one ``ChunkEmbedding`` per chunk.

    Calls ``provider.embed_batch`` once per ``batch_size``-group (so the call
    count is observable and the stage is provider-agnostic), then upserts the rows
    in a single ``INSERT ... ON CONFLICT (chunk_id, embedding_provider,
    embedding_model) DO UPDATE`` that replaces ``embedding_dimensions`` /
    ``embedding_vector`` while preserving the existing row's ``id`` / ``created_at``
    (``ChunkEmbedding`` has no ``updated_at``). Returns the persisted rows via
    ``RETURNING`` (the live row on both insert and conflict). Empty input returns
    ``[]`` with no provider call. Flushes; never commits.
    """
    if not chunks:
        return []

    values: list[dict[str, Any]] = []
    for group in _batched(chunks, batch_size):
        embeddings = await provider.embed_batch(
            [chunk.text for chunk in group], trace_context=trace_context
        )
        for chunk, embedding in zip(group, embeddings, strict=True):
            values.append(_embedding_values(chunk, embedding))

    insert_stmt = pg_insert(ChunkEmbedding).values(values)
    upsert = insert_stmt.on_conflict_do_update(
        index_elements=["chunk_id", "embedding_provider", "embedding_model"],
        set_={
            "embedding_dimensions": insert_stmt.excluded.embedding_dimensions,
            "embedding_vector": insert_stmt.excluded.embedding_vector,
        },
    ).returning(ChunkEmbedding)
    result = await session.execute(upsert)
    rows = list(result.scalars().all())
    await session.flush()
    return rows


async def re_embed_for_model(
    session: AsyncSession,
    document_id: str,
    *,
    provider: EmbeddingProvider,
    batch_size: int,
    trace_context: TraceContext | None = None,
) -> list[ChunkEmbedding]:
    """Ad-hoc/admin re-embed of every chunk for ``document_id`` (no API path).

    Loads the document's ``Chunk`` rows and re-runs ``embed_chunks``. Because
    ``embed_chunks`` upserts by ``(chunk_id, provider, model)``, re-running with
    the configured provider/model replaces those rows in place, while a newly
    configured model inserts a fresh row per chunk — the doc-2 § 6 / doc-5 § 7
    re-embedding rule. For manual/admin use only; not exposed via the API.
    """
    result = await session.execute(
        select(Chunk).where(Chunk.document_id == document_id)
    )
    chunks = list(result.scalars().all())
    return await embed_chunks(
        session,
        chunks,
        provider=provider,
        batch_size=batch_size,
        trace_context=trace_context,
    )
