"""Unit tests for ingestion.pipeline.embedding.embed_chunks (Phase 10.2).

No real DB: a minimal fake session records ``execute``/``flush``/``commit`` so the
batching, trace-forwarding, and empty-input behaviour is observable, and the pure
``_embedding_values`` mapping is asserted directly. The real ON CONFLICT upsert
(one row per chunk, re-embed semantics, ``dimensions == 1536`` in the DB) is
covered by the integration tests, which need the real unique constraint.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.ingestion.pipeline.embedding import _embedding_values, embed_chunks
from rag_recipes.providers._observability import TraceContext
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.embeddings.types import Embedding
from rag_recipes.storage.models.chunk import Chunk


class _RecordingEmbeddingProvider(EmbeddingProvider):
    """Wraps FakeEmbeddingProvider and records every embed_batch call.

    Scaffolding for the embed_chunks tests (subject = embed_chunks): the Fake has
    no call log, so the "respects batch size" assertion reads this spy's `.calls`.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._fake = FakeEmbeddingProvider(**kwargs)
        self.calls: list[tuple[list[str], TraceContext | None]] = []

    async def embed_text(
        self, text: str, *, trace_context: TraceContext | None = None
    ) -> Embedding:
        return await self._fake.embed_text(text, trace_context=trace_context)

    async def embed_batch(
        self, texts: list[str], *, trace_context: TraceContext | None = None
    ) -> list[Embedding]:
        self.calls.append((list(texts), trace_context))
        return await self._fake.embed_batch(texts, trace_context=trace_context)


class _FakeResult:
    def scalars(self) -> _FakeResult:
        return self

    def all(self) -> list[Any]:
        return []


class _FakeSession:
    """Records execute/flush/commit; the upsert itself is exercised in integration."""

    def __init__(self) -> None:
        self.execute_calls = 0
        self.flush_calls = 0
        self.commit_calls = 0

    async def execute(self, statement: Any) -> _FakeResult:
        self.execute_calls += 1
        return _FakeResult()

    async def flush(self) -> None:
        self.flush_calls += 1

    async def commit(self) -> None:
        self.commit_calls += 1


def _chunk(idx: int) -> Chunk:
    return Chunk(id=f"chunk_{idx}", text=f"chunk text {idx}")


@pytest.mark.asyncio
async def test_embed_chunks_batches_by_batch_size() -> None:
    provider = _RecordingEmbeddingProvider()
    session = _FakeSession()
    chunks = [_chunk(i) for i in range(5)]

    await embed_chunks(session, chunks, provider=provider, batch_size=2)  # type: ignore[arg-type]

    # 5 chunks at batch_size=2 → embed_batch called with texts of lengths 2, 2, 1.
    assert [len(texts) for texts, _ in provider.calls] == [2, 2, 1]
    assert provider.calls[0][0] == ["chunk text 0", "chunk text 1"]
    # A single upsert statement and one flush; never a commit (caller owns the txn).
    assert session.execute_calls == 1
    assert session.flush_calls == 1
    assert session.commit_calls == 0


@pytest.mark.asyncio
async def test_embed_chunks_forwards_trace_context_to_every_call() -> None:
    provider = _RecordingEmbeddingProvider()
    session = _FakeSession()
    trace = TraceContext(session_id="doc_123")
    chunks = [_chunk(i) for i in range(3)]

    await embed_chunks(
        session, chunks, provider=provider, batch_size=2, trace_context=trace  # type: ignore[arg-type]
    )

    assert provider.calls  # at least one call
    assert all(ctx is trace for _, ctx in provider.calls)


@pytest.mark.asyncio
async def test_embed_chunks_empty_input_returns_empty_and_calls_nothing() -> None:
    provider = _RecordingEmbeddingProvider()
    session = _FakeSession()

    result = await embed_chunks(session, [], provider=provider, batch_size=2)  # type: ignore[arg-type]

    assert result == []
    assert provider.calls == []
    assert session.execute_calls == 0
    assert session.flush_calls == 0


def test_embedding_values_maps_fields_from_embedding() -> None:
    chunk = _chunk(7)
    embedding = Embedding(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        vector=[0.1] * 1536,
    )

    values = _embedding_values(chunk, embedding)

    assert values["chunk_id"] == "chunk_7"
    assert values["embedding_provider"] == "openai"
    assert values["embedding_model"] == "text-embedding-3-small"
    assert values["embedding_dimensions"] == 1536
    assert values["embedding_vector"] == [0.1] * 1536
    assert values["id"].startswith("embedding_")


def test_embedding_values_dimensions_match_fake_default() -> None:
    chunk = _chunk(0)
    [embedding] = [
        Embedding(provider="fake", model="fake-embedding", dimensions=1536, vector=[0.0] * 1536)
    ]
    values = _embedding_values(chunk, embedding)
    assert values["embedding_dimensions"] == 1536
    assert len(values["embedding_vector"]) == 1536


def test_reason_for_maps_embedding_technical_error() -> None:
    # An EmbeddingTechnicalError raised by the embedding stage routes through
    # process_document's handler with the structured "embedding_failed" reason.
    from rag_recipes.ingestion.jobs import _reason_for
    from rag_recipes.providers.errors import EmbeddingTechnicalError

    assert _reason_for(EmbeddingTechnicalError("boom")) == "embedding_failed"
