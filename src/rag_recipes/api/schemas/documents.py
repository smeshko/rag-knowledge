"""Response schemas for the documents API (doc 6 § 2)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    asset_id: str
    category: str
    subcategory: str | None
    title: str
    author: str
    source_type: str
    language: str | None
    active_source_version: int | None
    status: str
    created_at: datetime
    updated_at: datetime


class UploadIngestion(BaseModel):
    status: str


class UploadResponse(BaseModel):
    document: DocumentResponse
    ingestion: UploadIngestion


class BatchUploadItemStatus(StrEnum):
    """Per-file outcome of a batch cohort upload — closed set (Epic 21.1)."""

    CREATED = "created"
    DUPLICATE = "duplicate"
    ERROR = "error"


class BatchUploadErrorCode(StrEnum):
    """Machine-readable code on ``error`` items (Epic 21.1, D2).

    Mirrors the ``ErrorCode`` values actually raisable by the per-file upload
    helper; anything unexpected maps to ``internal_error``.
    """

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    INTERNAL_ERROR = "internal_error"


class BatchUploadItemResult(BaseModel):
    """Per-file outcome in a batch cohort upload (Epic 19.2)."""

    filename: str
    status: BatchUploadItemStatus
    document_id: str | None = None
    error: str | None = None
    error_code: BatchUploadErrorCode | None = None


class BatchUploadResponse(BaseModel):
    items: list[BatchUploadItemResult]
    total: int
    created: int
    duplicates: int
    errors: int


class DocumentListItem(BaseModel):
    """The doc §3 list item — smaller than ``DocumentResponse`` (no
    ``asset_id``/``language``/timestamps)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    category: str
    subcategory: str | None
    title: str
    author: str
    source_type: str
    status: str
    active_source_version: int | None


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]


class DocumentCounts(BaseModel):
    source_spans: int
    knowledge_items: int
    ready_items: int
    needs_review_items: int
    chunks: int


class DocumentDetailResponse(BaseModel):
    document: DocumentResponse
    counts: DocumentCounts


class IngestionProgress(BaseModel):
    stage: str
    message: str | None
    pages_total: int | None
    pages_processed: int | None


class IngestionFailureInfo(BaseModel):
    """Latest ingestion failure for a FAILED document (doc 6 § 5; Epic 21 D1).

    ``stage`` is the status the document failed *from*
    (``IngestionFailure.last_status``), not its current status.
    ``error_message`` is deliberately excluded — ``reason`` is the stable,
    FE-presentable code.
    """

    reason: str
    stage: str
    failed_at: datetime


class IngestionStatusResponse(BaseModel):
    document_id: str
    status: str
    active_source_version: int | None
    current_source_version: int | None
    progress: IngestionProgress
    terminal: bool
    failure: IngestionFailureInfo | None = None


class ReprocessRequest(BaseModel):
    mode: str
    reason: str | None = None


class ReprocessResponse(BaseModel):
    document_id: str
    status: str
    previous_active_source_version: int | None
    current_source_version: int | None
