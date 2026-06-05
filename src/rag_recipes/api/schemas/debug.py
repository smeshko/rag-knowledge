"""Response schemas for the dev-only debug endpoints (doc 6 § 9)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ExtractionRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    source_version: int
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    status: str
    created_at: datetime
    completed_at: datetime | None


class ExtractionRunListResponse(BaseModel):
    extraction_runs: list[ExtractionRunSummary]


class ExtractionRunDetail(ExtractionRunSummary):
    input_source_span_ids: list[str]
    input_hash: str
    output_json: dict[str, Any] | None
    error_message: str | None


class SourceSpanSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    source_version: int
    source_type: str
    locator: dict[str, Any]
    locator_hash: str
    # Full span text — the copyright-sensitive field (doc 6 § 9). The endpoint is
    # dev-only and 404s in production.
    text: str
    text_hash: str
    created_at: datetime


class SourceSpanListResponse(BaseModel):
    source_spans: list[SourceSpanSummary]
