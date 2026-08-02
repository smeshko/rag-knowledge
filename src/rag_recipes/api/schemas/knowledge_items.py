"""Schemas for the knowledge-item detail (doc 6 § 8) and its edit request."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    # When a reviewer last corrected this item in place; null means never edited
    # (Epic 22.2). Lets the queue mark an item as already corrected.
    edited_at: datetime | None = None


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


class KnowledgeItemUpdateRequest(BaseModel):
    """A reviewer's in-place correction of a ``needs_review`` item (Epic 22.2).

    Every field is optional and read with ``exclude_unset`` semantics: absent
    means "leave it alone", explicit ``null`` means "clear it". Clearing a
    summary or a yield is a real operation, so the two cannot be conflated.

    The two lists are **whole-array replacement** — what a form submits — which
    is what makes add, remove and reorder fall out for free.

    Only the fields a human can meaningfully author are here. ``confidence``,
    ``source_span_ids``, ``schema``, ``item_type`` and ``warnings`` are
    machine-owned provenance and are not client-writable; the per-ingredient
    parse (``quantity_value``, ``unit_normalized``, …) is nulled on an edited
    line rather than maintained by hand.
    """

    # Unknown keys are rejected rather than ignored: a client trying to write
    # `confidence` or `warnings` should hear "no", not have it silently dropped.
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    summary: str | None = None
    yield_: str | None = Field(default=None, alias="yield")
    prep_time: str | None = None
    cook_time: str | None = None
    total_time: str | None = None
    ingredients: list[str] | None = None
    steps: list[str] | None = None

    @model_validator(mode="after")
    def _non_nullable_fields_are_not_nulled(self) -> KnowledgeItemUpdateRequest:
        """Reject an explicit null on the fields that have no "cleared" meaning.

        ``title`` is NOT NULL on the row and hard-validated non-empty at ingest,
        so neither an explicit null nor whitespace-only can be honoured. The two
        lists are whole-array replacement: an empty section is ``[]``, and a
        client that serialises it as ``null`` instead is making a mistake worth
        hearing about rather than a 500 three layers down.

        Absent is always fine — it means "leave this alone".
        """
        supplied = self.model_fields_set
        if "title" in supplied and not (self.title or "").strip():
            raise ValueError("title must be a non-empty string")
        for name in ("ingredients", "steps"):
            if name in supplied and getattr(self, name) is None:
                raise ValueError(f"{name} must be a list, not null (send [] to empty it)")
        return self

    @field_validator("ingredients", "steps")
    @classmethod
    def _lines_must_not_be_blank(cls, value: list[str] | None) -> list[str] | None:
        """A blank line would violate the ingest-time non-empty ``raw_text`` rule.

        Rejected rather than silently dropped: dropping a line the reviewer can
        still see in the form is the kind of surprise that costs trust.
        """
        if value is not None and any(not line.strip() for line in value):
            raise ValueError("lines must not be blank")
        return value
