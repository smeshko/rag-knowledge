"""Status enums shared by ORM models and Postgres ENUM type definitions."""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    PDF = "pdf"
    # A source with no file behind it: the handwritten shelf, whose one
    # SourceAsset row exists only to satisfy documents.asset_id and whose
    # knowledge items are typed by a human rather than extracted. Load-bearing
    # beyond provenance — ``POST /documents/{id}/reprocess`` refuses a MANUAL
    # document, because re-extracting a book with no PDF would supersede every
    # hand-typed recipe in it with nothing.
    MANUAL = "manual"


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


class ExtractionBatchStatus(StrEnum):
    # Local mirror of the Anthropic batch lifecycle (Epic 19.2). SUBMITTING is the
    # durable-claim state committed *before* the provider call so a crash can never
    # double-submit (DECISIONS #7); 19.3 advances IN_PROGRESS/ENDED on poll.
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    FAILED = "failed"


class ExtractionBatchItemStatus(StrEnum):
    # Per-window registration lifecycle (Epic 19.2). 19.2 sets PENDING/SUBMITTING/
    # SUBMITTED; the terminal values are declared now so 19.3 result ingestion needs
    # no enum migration.
    PENDING = "pending"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    ERRORED = "errored"
    EXPIRED = "expired"
    CANCELED = "canceled"


class KnowledgeItemStatus(StrEnum):
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    SUPERSEDED = "superseded"
    # Phase 9.5: staging status for candidates committed mid-extraction. Keeps
    # in-flight items invisible to readers querying ready/needs_review until
    # finalize promotes the winners in one atomic transaction.
    EXTRACTING = "extracting"
    # Phase 21.3 (review decisions): transitional approve state — the review
    # POST flips needs_review → indexing and the index_knowledge_item job
    # completes indexing → ready (or the item is reverted to needs_review).
    INDEXING = "indexing"
    # Phase 21.3 (review decisions): terminal reject state. Never superseded,
    # never resurrected, excluded from counts (D5/D7).
    REJECTED = "rejected"


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
