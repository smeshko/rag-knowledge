"""Unit tests for rag_recipes.storage.models.chunk_embedding."""

from __future__ import annotations

from rag_recipes.storage.base import Vector
from rag_recipes.storage.models import ChunkEmbedding


def test_tablename() -> None:
    assert ChunkEmbedding.__tablename__ == "chunk_embeddings"


def test_table_columns_match_doc_2() -> None:
    expected = [
        "id",
        "chunk_id",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "embedding_vector",
        "created_at",
    ]
    assert list(ChunkEmbedding.__table__.columns.keys()) == expected


def test_no_updated_at_column() -> None:
    assert "updated_at" not in ChunkEmbedding.__table__.columns


def test_foreign_key_targets_chunks() -> None:
    col = ChunkEmbedding.__table__.columns["chunk_id"]
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "chunks.id"


def test_default_id_uses_embedding_prefix() -> None:
    default = ChunkEmbedding.__table__.columns["id"].default
    assert default is not None
    value = default.arg(None)
    assert value.startswith("embedding_")


def test_embedding_vector_is_vector_1536() -> None:
    col = ChunkEmbedding.__table__.columns["embedding_vector"]
    assert isinstance(col.type, Vector)
    assert col.type.dim == 1536


def test_construct_with_vector_list() -> None:
    instance = ChunkEmbedding(
        chunk_id="chunk_x",
        embedding_provider="p",
        embedding_model="m",
        embedding_dimensions=1536,
        embedding_vector=[0.0] * 1536,
    )
    assert instance.embedding_vector[0] == 0.0
