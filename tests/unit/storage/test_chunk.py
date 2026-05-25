"""Unit tests for rag_recipes.storage.models.chunk."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from rag_recipes.storage.models import Chunk, KnowledgeItem


def test_tablename() -> None:
    assert Chunk.__tablename__ == "chunks"


def test_table_columns_match_doc_2() -> None:
    expected = [
        "id",
        "document_id",
        "parent_type",
        "parent_id",
        "chunk_type",
        "text",
        "text_hash",
        "source_span_ids",
        "metadata",
        "created_at",
        "updated_at",
    ]
    assert list(Chunk.__table__.columns.keys()) == expected


def test_chunk_metadata_python_attribute_maps_to_metadata_column() -> None:
    assert Chunk.__mapper__.attrs["chunk_metadata"].columns[0].name == "metadata"
    assert Chunk.__mapper__.attrs.get("metadata") is None


def test_foreign_keys_target_documents_and_knowledge_items() -> None:
    doc_fks = list(Chunk.__table__.columns["document_id"].foreign_keys)
    parent_fks = list(Chunk.__table__.columns["parent_id"].foreign_keys)
    assert len(doc_fks) == 1
    assert doc_fks[0].target_fullname == "documents.id"
    assert len(parent_fks) == 1
    assert parent_fks[0].target_fullname == "knowledge_items.id"


def test_default_id_uses_chunk_prefix() -> None:
    default = Chunk.__table__.columns["id"].default
    assert default is not None
    value = default.arg(None)
    assert value.startswith("chunk_")


def test_parent_type_enum_uses_chunk_parent_type_enum_type() -> None:
    col_type = Chunk.__table__.columns["parent_type"].type
    assert isinstance(col_type, sa.Enum)
    assert col_type.name == "chunk_parent_type_enum"
    assert col_type.enums == ["knowledge_item"]


def test_chunk_type_enum_uses_chunk_type_enum_type() -> None:
    col_type = Chunk.__table__.columns["chunk_type"].type
    assert isinstance(col_type, sa.Enum)
    assert col_type.name == "chunk_type_enum"
    assert col_type.enums == [
        "recipe_full",
        "recipe_title",
        "recipe_summary",
        "recipe_ingredients",
        "recipe_steps",
    ]


def test_invalid_chunk_type_raises() -> None:
    with pytest.raises(ValueError):
        Chunk(chunk_type="bogus")


def test_invalid_parent_type_raises() -> None:
    with pytest.raises(ValueError):
        Chunk(parent_type="bogus")


def test_no_document_relationship() -> None:
    assert Chunk.__mapper__.relationships.get("document") is None


def test_jsonb_column_types() -> None:
    cols = Chunk.__table__.columns
    assert isinstance(cols["source_span_ids"].type, JSONB)
    assert isinstance(cols["metadata"].type, JSONB)


def test_all_columns_not_null() -> None:
    for col in Chunk.__table__.columns:
        assert col.nullable is False, col.name


def test_document_id_validator_raises_after_knowledge_item_attached_with_mismatched_doc() -> None:
    item = KnowledgeItem(document_id="doc_A")
    chunk = Chunk(document_id="doc_B")
    with pytest.raises(ValueError, match="document_id"):
        chunk.knowledge_item = item


def test_document_id_validator_raises_when_document_id_changed_after_attach() -> None:
    item = KnowledgeItem(document_id="doc_A")
    chunk = Chunk(document_id="doc_A")
    chunk.knowledge_item = item
    with pytest.raises(ValueError, match="document_id"):
        chunk.document_id = "doc_B"


def test_document_id_validator_matching_passes() -> None:
    item = KnowledgeItem(document_id="doc_A")
    chunk = Chunk(document_id="doc_A")
    chunk.knowledge_item = item
    assert chunk.knowledge_item is item
