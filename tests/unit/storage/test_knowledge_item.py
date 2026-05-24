"""Unit tests for rag_recipes.storage.models.knowledge_item."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from rag_recipes.storage.models import KnowledgeItem


def test_tablename() -> None:
    assert KnowledgeItem.__tablename__ == "knowledge_items"


def test_table_columns_match_doc_2() -> None:
    expected = [
        "id",
        "document_id",
        "extraction_run_id",
        "source_version",
        "item_type",
        "title",
        "normalized_title",
        "summary",
        "body_text",
        "source_span_ids",
        "structured_data",
        "confidence",
        "status",
        "created_at",
        "updated_at",
    ]
    assert list(KnowledgeItem.__table__.columns.keys()) == expected


def test_foreign_keys_target_documents_and_extraction_runs() -> None:
    doc_fks = list(KnowledgeItem.__table__.columns["document_id"].foreign_keys)
    run_fks = list(KnowledgeItem.__table__.columns["extraction_run_id"].foreign_keys)
    assert len(doc_fks) == 1
    assert doc_fks[0].target_fullname == "documents.id"
    assert len(run_fks) == 1
    assert run_fks[0].target_fullname == "extraction_runs.id"


def test_default_id_uses_item_prefix() -> None:
    default = KnowledgeItem.__table__.columns["id"].default
    assert default is not None
    value = default.arg(None)
    assert value.startswith("item_")


def test_nullable_columns() -> None:
    cols = KnowledgeItem.__table__.columns
    assert cols["summary"].nullable is True
    assert cols["confidence"].nullable is True
    for name in [
        "id",
        "document_id",
        "extraction_run_id",
        "source_version",
        "item_type",
        "title",
        "normalized_title",
        "body_text",
        "source_span_ids",
        "structured_data",
        "status",
        "created_at",
        "updated_at",
    ]:
        assert cols[name].nullable is False, name


def test_status_enum_uses_knowledge_item_status_enum_type() -> None:
    col_type = KnowledgeItem.__table__.columns["status"].type
    assert isinstance(col_type, sa.Enum)
    assert col_type.name == "knowledge_item_status_enum"
    assert col_type.enums == ["ready", "needs_review", "superseded"]


def test_invalid_status_raises() -> None:
    with pytest.raises(ValueError):
        KnowledgeItem(status="bogus")


def test_jsonb_column_types() -> None:
    cols = KnowledgeItem.__table__.columns
    assert isinstance(cols["source_span_ids"].type, JSONB)
    assert isinstance(cols["structured_data"].type, JSONB)
    assert isinstance(cols["confidence"].type, JSONB)


def test_structured_data_default_callable_returns_empty_dict() -> None:
    default = KnowledgeItem.__table__.columns["structured_data"].default
    assert default is not None
    assert default.arg(None) == {}


def test_source_span_ids_accepts_list_of_strings() -> None:
    item = KnowledgeItem(source_span_ids=["span_1", "span_2"])
    assert item.source_span_ids == ["span_1", "span_2"]
