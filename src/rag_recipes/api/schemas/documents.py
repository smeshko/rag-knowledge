"""Response schemas for the documents API (doc 6 § 2)."""

from __future__ import annotations

from datetime import datetime

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


class IngestionStatusResponse(BaseModel):
    document_id: str
    status: str
    active_source_version: int | None
    current_source_version: int | None
    progress: IngestionProgress
    terminal: bool


class ReprocessRequest(BaseModel):
    mode: str
    reason: str | None = None


class ReprocessResponse(BaseModel):
    document_id: str
    status: str
    previous_active_source_version: int | None
    current_source_version: int | None
