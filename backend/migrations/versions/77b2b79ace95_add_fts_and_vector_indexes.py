"""add fts and vector indexes

Revision ID: 77b2b79ace95
Revises: 403b4234ea81
Create Date: 2026-06-04 16:39:39.717791

Phase 10.3 — the search indexes retrieval (Epic 12) depends on. The bodies are
hand-written because Alembic autogenerate cannot see a generated column or the
HNSW / GIN-with-ops index syntax (it would emit an empty upgrade()):

- ``chunks.ts_vector`` — a DB-managed ``GENERATED ALWAYS AS
  (to_tsvector('english', text)) STORED`` column. It is intentionally NOT mapped
  on the ``Chunk`` ORM model (DECISIONS #1): a generated column is read-only to
  the ORM and nothing in Epic 10 reads it; Epic 12 queries it via raw SQL /
  ``column("ts_vector")``. The two-arg ``to_tsvector('english', text)`` is
  IMMUTABLE, so it is legal in a generated column (the single-arg form is not).
- ``ix_chunks_ts_vector`` — GIN index over the generated tsvector (keyword search).
- ``ix_chunk_embeddings_hnsw`` — HNSW index with ``vector_cosine_ops`` (vector
  search; needs pgvector >= 0.5.0, default m=16 / ef_construction=64).
- ``ix_chunk_embeddings_provider_model`` — btree filter so vector queries scope to
  one (provider, model) embedding space.

``downgrade()`` drops all four in reverse and leaves the ``vector`` extension
installed (it is a cluster object created by the initial migration).
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '77b2b79ace95'
down_revision: Union[str, Sequence[str], None] = '403b4234ea81'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "ALTER TABLE chunks ADD COLUMN ts_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_ts_vector ON chunks USING gin (ts_vector)")
    op.execute(
        "CREATE INDEX ix_chunk_embeddings_hnsw ON chunk_embeddings "
        "USING hnsw (embedding_vector vector_cosine_ops)"
    )
    op.create_index(
        "ix_chunk_embeddings_provider_model",
        "chunk_embeddings",
        ["embedding_provider", "embedding_model"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_chunk_embeddings_provider_model", table_name="chunk_embeddings"
    )
    op.execute("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_chunks_ts_vector")
    op.execute("ALTER TABLE chunks DROP COLUMN ts_vector")
