"""Response schemas for GET /api/v1/knowledge-items/{id} (doc 6 § 8)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ReviewReason(BaseModel):
    """One reason an item needs review (Epic 21.1, D3).

    ``code`` is a stable machine code (a ``validate_soft`` warning code, or
    the ``llm_warning`` envelope for non-canonical strings); ``message`` is a
    presentation-layer human label.
    """

    code: str
    message: str


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
    # Mapped from structured_data["warnings"] for needs_review items; [] otherwise
    # (Epic 21.1, D3).
    review_reasons: list[ReviewReason] = []


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
