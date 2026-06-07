"""Unit tests for the ExtractionBatch / ExtractionBatchItem models (Epic 19.2)."""

from __future__ import annotations

import re

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from rag_recipes.storage.enums import (
    ExtractionBatchItemStatus,
    ExtractionBatchStatus,
)
from rag_recipes.storage.models import ExtractionBatch, ExtractionBatchItem

_CUSTOM_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def test_tablenames() -> None:
    assert ExtractionBatch.__tablename__ == "extraction_batches"
    assert ExtractionBatchItem.__tablename__ == "extraction_batch_items"


def test_batch_columns() -> None:
    assert list(ExtractionBatch.__table__.columns.keys()) == [
        "id",
        "provider",
        "provider_batch_id",
        "model",
        "processing_status",
        "request_count",
        "created_at",
        "completed_at",
    ]


def test_item_columns_include_request_input_and_schema() -> None:
    cols = ExtractionBatchItem.__table__.columns
    assert "request_input" in cols
    assert "request_schema" in cols
    assert isinstance(cols["request_schema"].type, JSONB)
    assert isinstance(cols["input_source_span_ids"].type, JSONB)


def test_item_batch_id_is_nullable_fk() -> None:
    col = ExtractionBatchItem.__table__.columns["batch_id"]
    assert col.nullable is True
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "extraction_batches.id"


def test_item_document_id_is_fk() -> None:
    col = ExtractionBatchItem.__table__.columns["document_id"]
    fks = list(col.foreign_keys)
    assert fks[0].target_fullname == "documents.id"


def test_submitting_present_in_both_enums() -> None:
    assert ExtractionBatchStatus.SUBMITTING.value == "submitting"
    assert ExtractionBatchItemStatus.SUBMITTING.value == "submitting"
    # Reserved terminal item statuses for 19.3 — declared now to avoid a second
    # enum migration.
    for terminal in ("succeeded", "rejected", "errored", "expired", "canceled"):
        assert terminal in {m.value for m in ExtractionBatchItemStatus}


def test_item_id_is_valid_anthropic_custom_id() -> None:
    default = ExtractionBatchItem.__table__.columns["id"].default
    value = default.arg(None)
    assert value.startswith("ebitem_")
    # Anthropic custom_id constraint: 1-64 chars of [A-Za-z0-9_-].
    assert _CUSTOM_ID_RE.fullmatch(value), value


def test_partial_unique_index_present() -> None:
    index = next(
        ix
        for ix in ExtractionBatchItem.__table__.indexes
        if ix.name == "uq_extraction_batch_items_doc_version_hash_active"
    )
    assert index.unique is True
    assert {c.name for c in index.columns} == {
        "document_id",
        "source_version",
        "input_hash",
    }
    # Scoped to non-terminal statuses only.
    where_sql = str(index.dialect_options["postgresql"]["where"]).lower()
    assert "pending" in where_sql and "submitting" in where_sql and "submitted" in where_sql


def test_status_enum_types() -> None:
    batch_type = ExtractionBatch.__table__.columns["processing_status"].type
    assert isinstance(batch_type, sa.Enum)
    assert batch_type.enum_class is ExtractionBatchStatus
    item_type = ExtractionBatchItem.__table__.columns["status"].type
    assert isinstance(item_type, sa.Enum)
    assert item_type.enum_class is ExtractionBatchItemStatus


def test_invalid_status_raises() -> None:
    with pytest.raises(ValueError):
        ExtractionBatchItem(
            document_id="doc_x",
            source_version=1,
            input_hash="h",
            input_source_span_ids=["span_x"],
            request_input="prompt",
            request_schema={"type": "object"},
            prompt_version="v1",
            schema_version="recipe.v1",
            status="bogus",  # type: ignore[arg-type]
        )
