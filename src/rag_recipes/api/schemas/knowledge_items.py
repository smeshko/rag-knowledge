"""Response schemas for GET /api/v1/knowledge-items/{id} (doc 6 § 8)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class KnowledgeItemDetail(BaseModel):
    id: str
    document_id: str
    item_type: str
    title: str
    summary: str | None
    status: str
    source_span_ids: list[str]
    confidence: dict[str, Any] | None
    # The FULL recipe.v1 structured payload, passed through verbatim (no per-field
    # model) so any structured_data.schema version renders unchanged and nothing is
    # truncated.
    structured_data: dict[str, Any]


class KnowledgeItemDisplay(BaseModel):
    title: str
    subtitle: str | None


class KnowledgeItemSourceCitation(BaseModel):
    source_span_id: str
    label: str
    locator: dict[str, Any] | None


class KnowledgeItemResponse(BaseModel):
    knowledge_item: KnowledgeItemDetail
    display: KnowledgeItemDisplay
    source_citations: list[KnowledgeItemSourceCitation]
