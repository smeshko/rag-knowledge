"""Unit tests for rag_recipes.storage.enums."""

from __future__ import annotations

from enum import StrEnum

import pytest
import sqlalchemy as sa

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import (
    ChunkParentType,
    ChunkType,
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    ReprocessMode,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.models import (  # noqa: F401 — register tables on Base.metadata
    Document,
    ExtractionRun,
    SourceAsset,
    SourceSpan,
)


def test_source_type_members() -> None:
    assert issubclass(SourceType, StrEnum)
    assert {member.value for member in SourceType} == {"pdf"}


def test_upload_status_members() -> None:
    assert issubclass(UploadStatus, StrEnum)
    assert {member.value for member in UploadStatus} == {
        "uploading",
        "uploaded",
        "upload_failed",
        "deleted",
    }


def test_document_status_members() -> None:
    assert issubclass(DocumentStatus, StrEnum)
    assert {member.value for member in DocumentStatus} == {
        "queued",
        "extracting_text",
        "creating_source_spans",
        "extracting_items",
        "validating_items",
        "creating_chunks",
        "embedding_chunks",
        "indexing",
        "ready",
        "needs_review",
        "failed",
    }


def test_extraction_run_status_members() -> None:
    assert issubclass(ExtractionRunStatus, StrEnum)
    assert {member.value for member in ExtractionRunStatus} == {
        "running",
        "success",
        "failed",
        "rejected",
    }


def test_knowledge_item_status_members() -> None:
    assert issubclass(KnowledgeItemStatus, StrEnum)
    assert {member.value for member in KnowledgeItemStatus} == {
        "ready",
        "needs_review",
        "superseded",
    }


def test_chunk_type_members() -> None:
    assert issubclass(ChunkType, StrEnum)
    assert {member.value for member in ChunkType} == {
        "recipe_full",
        "recipe_title",
        "recipe_summary",
        "recipe_ingredients",
        "recipe_steps",
    }


def test_chunk_parent_type_members() -> None:
    assert issubclass(ChunkParentType, StrEnum)
    assert {member.value for member in ChunkParentType} == {"knowledge_item"}


def test_reprocess_mode_members() -> None:
    assert issubclass(ReprocessMode, StrEnum)
    assert {member.value for member in ReprocessMode} == {
        "auto",
        "reuse_source_spans",
        "new_source_version",
    }
    assert ReprocessMode("auto") is ReprocessMode.AUTO
    with pytest.raises(ValueError):
        ReprocessMode("bogus")


def _column_enum(table: str, column: str) -> sa.Enum:
    coltype = Base.metadata.tables[table].columns[column].type
    assert isinstance(coltype, sa.Enum)
    return coltype


def test_enum_column_labels_use_lowercase_values() -> None:
    """Postgres ENUM labels must be the lowercase enum values, not member names.

    Doc 2 §1–§3 documents lowercase values in JSON examples (e.g. "uploaded",
    "ready", "queued"). Without ``values_callable``, SQLAlchemy persists the
    uppercase member names ("UPLOADED", "READY", ...), which would diverge
    from the public contract and from any raw SQL written against the schema.
    """
    cases = [
        ("source_assets", "source_type", SourceType),
        ("source_assets", "upload_status", UploadStatus),
        ("documents", "source_type", SourceType),
        ("documents", "status", DocumentStatus),
        ("source_spans", "source_type", SourceType),
        ("extraction_runs", "status", ExtractionRunStatus),
    ]
    for table, column, enum_cls in cases:
        labels = _column_enum(table, column).enums
        expected = [member.value for member in enum_cls]
        assert labels == expected, (
            f"{table}.{column} enum labels {labels!r} do not match "
            f"{enum_cls.__name__}.value list {expected!r}"
        )
