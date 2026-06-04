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


class KnowledgeItemStatus(StrEnum):
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    SUPERSEDED = "superseded"
    # Phase 9.5: staging status for candidates committed mid-extraction. Keeps
    # in-flight items invisible to readers querying ready/needs_review until
    # finalize promotes the winners in one atomic transaction.
    EXTRACTING = "extracting"


class ChunkType(StrEnum):
    RECIPE_FULL = "recipe_full"
    RECIPE_TITLE = "recipe_title"
    RECIPE_SUMMARY = "recipe_summary"
    RECIPE_INGREDIENTS = "recipe_ingredients"
    RECIPE_STEPS = "recipe_steps"


class ChunkParentType(StrEnum):
    KNOWLEDGE_ITEM = "knowledge_item"


class ReprocessMode(StrEnum):
    AUTO = "auto"
    REUSE_SOURCE_SPANS = "reuse_source_spans"
    NEW_SOURCE_VERSION = "new_source_version"
