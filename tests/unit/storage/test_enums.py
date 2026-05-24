"""Unit tests for rag_recipes.storage.enums."""

from __future__ import annotations

from enum import StrEnum

from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    SourceType,
    UploadStatus,
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
