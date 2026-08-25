"""Schemas for the review-queue surface (Epic 21.3, contract §1).

``schema`` / ``yield`` are reserved words JSON-side, carried by ``schema_`` /
``yield_`` with aliases — the idiom from ``api/schemas/search.py``.

``ReviewItem`` outgrew the review queue: the per-book listing
(``GET /documents/{id}/knowledge-items``) needs exactly the same row, so it is
re-exported here as ``KnowledgeItemSummary`` and wrapped by
``KnowledgeItemListResponse``. Both stay in *this* module rather than moving to
``schemas/knowledge_items.py`` because this module already imports
``ReviewReason`` from there — the reverse import would be a cycle.
"""

from __future__ import annotations

from datetime import datetime
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
    # Always ``needs_review`` on /review-items; the per-book listing is what
    # makes this field carry information (additive, so the queue's contract is
    # unchanged).
    status: str
    document: ReviewItemDocument
    source_pages: ReviewItemSourcePages
    extraction: ReviewItemExtraction
    # 21.1's canonical {code, message} projection, reused verbatim.
    flags: list[ReviewReason]
    # Null until a reviewer corrects the item in place (Epic 22.2), so the queue
    # can mark a row as already corrected.
    edited_at: datetime | None = None
    # Null unless the reader starred this recipe. Carried on every listing row,
    # not just the favourites one, so a shelf or queue card can render its own
    # star without a second request.
    favourited_at: datetime | None = None


class ReviewItemListResponse(BaseModel):
    review_items: list[ReviewItem]


#: The same row, named for the surface that is not a review queue. One model,
#: so a card rendered from either listing cannot drift.
KnowledgeItemSummary = ReviewItem


class KnowledgeItemListResponse(BaseModel):
    """``GET /documents/{document_id}/knowledge-items``.

    No total and no cursor, matching ``ReviewItemListResponse`` — clients walk
    ``limit``/``offset`` until a page comes back short.
    """

    knowledge_items: list[KnowledgeItemSummary]


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
