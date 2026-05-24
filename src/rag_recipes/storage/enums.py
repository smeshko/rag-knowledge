"""Status enums shared by ORM models and Postgres ENUM type definitions."""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    PDF = "pdf"


class UploadStatus(StrEnum):
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    UPLOAD_FAILED = "upload_failed"
    DELETED = "deleted"


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    EXTRACTING_TEXT = "extracting_text"
    CREATING_SOURCE_SPANS = "creating_source_spans"
    EXTRACTING_ITEMS = "extracting_items"
    VALIDATING_ITEMS = "validating_items"
    CREATING_CHUNKS = "creating_chunks"
    EMBEDDING_CHUNKS = "embedding_chunks"
    INDEXING = "indexing"
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ExtractionRunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"
