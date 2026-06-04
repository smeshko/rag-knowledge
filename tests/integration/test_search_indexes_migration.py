"""Integration test for the Phase 10.3 search-index migration.

The ``test_engine`` fixture runs ``alembic upgrade head``, so by the time these
tests run the migration has applied. They assert the four search objects exist:
the generated ``chunks.ts_vector`` column, its GIN index, the HNSW vector index,
and the ``(embedding_provider, embedding_model)`` filter index.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.asyncio


async def test_search_indexes_exist(db_session: AsyncSession) -> None:
    result = await db_session.execute(
        text(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename IN ('chunks', 'chunk_embeddings')"
        )
    )
    indexes = {row[0] for row in result.all()}
    assert "ix_chunks_ts_vector" in indexes
    assert "ix_chunk_embeddings_hnsw" in indexes
    assert "ix_chunk_embeddings_provider_model" in indexes
    # Pre-existing indexes are untouched.
    assert "ix_chunks_document_id" in indexes


async def test_chunks_ts_vector_is_generated_column(db_session: AsyncSession) -> None:
    result = await db_session.execute(
        text(
            "SELECT data_type, is_generated FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'ts_vector'"
        )
    )
    row = result.first()
    assert row is not None, "chunks.ts_vector column is missing"
    data_type, is_generated = row
    assert data_type == "tsvector"
    assert is_generated == "ALWAYS"


async def test_hnsw_index_uses_cosine_ops(db_session: AsyncSession) -> None:
    # The HNSW index must use vector_cosine_ops so Epic 12's cosine KNN can use it.
    result = await db_session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunk_embeddings_hnsw'")
    )
    indexdef = result.scalar()
    assert indexdef is not None
    assert "hnsw" in indexdef.lower()
    assert "vector_cosine_ops" in indexdef
