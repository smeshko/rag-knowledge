"""Schemas for the review-queue surface (Epic 21.3, contract §1).

``schema`` / ``yield`` are reserved words JSON-side, carried by ``schema_`` /
``yield_`` with aliases — the idiom from ``api/schemas/search.py``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_recipes.api.schemas.knowledge_items import ReviewReason


class ReviewItemDocument(BaseModel):
    id: str
    title: str


class ReviewItemSourcePages(BaseModel):
    """Min/max page bounds resolved from the item's own source spans.

    ``needs_review`` items have no chunks, so these come straight from
    ``KnowledgeItem.source_span_ids`` locators; both ``None`` when nothing
    resolves.
    """

    page_start: int | None
    page_end: int | None


class ReviewItemExtraction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(alias="schema")
    yield_: str | None = Field(alias="yield")
    top_ingredients: list[str]
    confidence_overall: float | None


class ReviewItem(BaseModel):
    id: str
    title: str
    summary: str | None
    item_type: str
    document: ReviewItemDocument
    source_pages: ReviewItemSourcePages
    extraction: ReviewItemExtraction
    # 21.1's canonical {code, message} projection, reused verbatim.
    flags: list[ReviewReason]


class ReviewItemListResponse(BaseModel):
    review_items: list[ReviewItem]


class ReviewDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewRequest(BaseModel):
    # Closed enum: a bad value 422s through the app's enveloped
    # RequestValidationError handler (contract §2; the doc's "FastAPI default
    # HTTPValidationError" line is a known doc defect — see plan D6).
    decision: ReviewDecision


class ReviewedKnowledgeItem(BaseModel):
    id: str
    document_id: str
    status: str


class ReviewResponse(BaseModel):
    knowledge_item: ReviewedKnowledgeItem
    decision: str
