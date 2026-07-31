"""Unit tests for rag_recipes.storage.models.extraction_run."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from rag_recipes.storage.enums import ExtractionRunStatus
from rag_recipes.storage.models import ExtractionRun


def test_tablename() -> None:
    assert ExtractionRun.__tablename__ == "extraction_runs"


def test_table_columns_match_doc_2() -> None:
    expected = [
        "id",
        "document_id",
        "source_version",
        "provider",
        "model",
        "prompt_version",
        "schema_version",
        "input_source_span_ids",
        "input_hash",
        "status",
        "output_json",
        "error_message",
        "created_at",
        "completed_at",
    ]
    assert list(ExtractionRun.__table__.columns.keys()) == expected


def test_no_updated_at_column() -> None:
    assert "updated_at" not in ExtractionRun.__table__.columns


def test_document_id_is_foreign_key() -> None:
    col = ExtractionRun.__table__.columns["document_id"]
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "documents.id"


def test_jsonb_columns() -> None:
    assert isinstance(ExtractionRun.__table__.columns["input_source_span_ids"].type, JSONB)
    assert isinstance(ExtractionRun.__table__.columns["output_json"].type, JSONB)


def test_input_source_span_ids_has_no_default() -> None:
    """Provenance must be supplied explicitly, not silently defaulted to [].

    ExtractionRun is the canonical audit record of which source spans the
    model saw. A default=list would let a construction bug persist a NOT NULL
    empty array, turning missing provenance into unrecoverable data loss for
    debugging, validation, and replay.
    """
    assert ExtractionRun.__table__.columns["input_source_span_ids"].default is None


def test_nullable_columns() -> None:
    cols = ExtractionRun.__table__.columns
    assert cols["output_json"].nullable is True
    assert cols["error_message"].nullable is True
    assert cols["completed_at"].nullable is True


def test_default_id_uses_run_prefix() -> None:
    default = ExtractionRun.__table__.columns["id"].default
    assert default is not None
    value = default.arg(None)
    assert value.startswith("run_")


def test_status_enum_uses_extraction_run_status_enum_type() -> None:
    col_type = ExtractionRun.__table__.columns["status"].type
    assert isinstance(col_type, sa.Enum)
    assert col_type.name == "extraction_run_status_enum"
    assert col_type.native_enum is True
    assert col_type.enum_class is ExtractionRunStatus


def test_invalid_status_raises() -> None:
    with pytest.raises(ValueError):
        ExtractionRun(
            source_version=1,
            provider="openai",
            model="gpt-4",
            prompt_version="v1",
            schema_version="v1",
            input_source_span_ids=["span_x"],
            input_hash="h",
            status="invalid_value",  # type: ignore[arg-type]
        )
