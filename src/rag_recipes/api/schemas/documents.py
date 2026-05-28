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
