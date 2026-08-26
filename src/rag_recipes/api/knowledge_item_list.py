"""The canonical listing-row projection for knowledge items.

Extracted from ``routes/review.py`` so the review queue
(``GET /review-items``) and the per-book contents listing
(``GET /documents/{document_id}/knowledge-items``) build the *same* row from
the same fields — the ``knowledge_item_view.build_knowledge_item_response``
precedent applied to the list shape: written once means two endpoints cannot
disagree about what a recipe row says.

Source pages are resolved from ``KnowledgeItem.source_span_ids`` rather than
from chunks (plan D4). That was originally a necessity — ``needs_review`` items
have no chunks — and is what makes the projection status-agnostic, so the
per-book listing gets page provenance for ``ready`` rows for free.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.review_reasons import build_review_reasons, thresholds_for_item
from rag_recipes.api.schemas.review import (
    KnowledgeItemSummary,
    ReviewItemDocument,
    ReviewItemExtraction,
    ReviewItemSourcePages,
)
from rag_recipes.api.search_projection import top_ingredients
from rag_recipes.ingestion.pipeline.persist import thresholds_from_settings
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.knowledge_item_favourite import KnowledgeItemFavourite
from rag_recipes.storage.models.source_span import SourceSpan

__all__ = ["build_summaries", "source_pages"]


def source_pages(
    span_ids: list[str], locators_by_id: dict[str, dict[str, Any]]
) -> ReviewItemSourcePages:
    """Min/max page bounds over the item's resolved span locators.

    Keys are read with ``.get`` and skipped when absent (the
    ``knowledge_item_view.pdf_page_label`` precedent) — a degraded locator must
    never 500 the whole listing. Both bounds are ``None`` when nothing resolves.
    """
    starts: list[int] = []
    ends: list[int] = []
    for span_id in span_ids:
        locator = locators_by_id.get(span_id)
        if not isinstance(locator, dict):
            continue
        start = locator.get("page_start")
        end = locator.get("page_end")
        if isinstance(start, int):
            starts.append(start)
        if isinstance(end, int):
            ends.append(end)
    return ReviewItemSourcePages(
        page_start=min(starts) if starts else None,
        page_end=max(ends) if ends else None,
    )


async def build_summaries(
    session: AsyncSession, rows: list[tuple[KnowledgeItem, str, str]]
) -> list[KnowledgeItemSummary]:
    """Project ``(item, document_id, document_title)`` rows into listing rows.

    One batched span fetch for the whole page, never one per item — the N+1 the
    21.3 listing was written to avoid.
    """
    # Resolved once per page, not per row: the bounds are settings-derived and
    # identical for every item, and `build_review_reasons` needs them to attach
    # the observed value/threshold aids to the three confidence codes.
    current_thresholds = thresholds_from_settings()
    page_span_ids = {
        span_id for item, _, _ in rows for span_id in (item.source_span_ids or [])
    }
    locators_by_id: dict[str, dict[str, Any]] = {}
    if page_span_ids:
        span_rows = (
            await session.execute(
                select(SourceSpan.id, SourceSpan.locator).where(
                    SourceSpan.id.in_(page_span_ids)
                )
            )
        ).all()
        locators_by_id = {row.id: row.locator for row in span_rows}

    # One batched read for the whole page, the span fetch's rule: a per-row
    # lookup here would reintroduce the N+1 this function exists to avoid.
    # `GET /favourites` joins the table itself, so for that caller this is a
    # second read of rows it already has — cheap (a PK-keyed IN over one page)
    # and worth it to keep every listing on the same projection.
    favourited_at_by_id: dict[str, datetime] = {}
    page_item_ids = [item.id for item, _, _ in rows]
    if page_item_ids:
        favourite_rows = (
            await session.execute(
                select(
                    KnowledgeItemFavourite.knowledge_item_id,
                    KnowledgeItemFavourite.created_at,
                ).where(KnowledgeItemFavourite.knowledge_item_id.in_(page_item_ids))
            )
        ).all()
        favourited_at_by_id = {row[0]: row[1] for row in favourite_rows}

    summaries: list[KnowledgeItemSummary] = []
    for item, doc_id, doc_title in rows:
        structured = item.structured_data or {}
        summaries.append(
            KnowledgeItemSummary(
                id=item.id,
                title=item.title,
                summary=item.summary,
                item_type=item.item_type,
                status=item.status.value,
                document=ReviewItemDocument(id=doc_id, title=doc_title),
                source_pages=source_pages(
                    list(item.source_span_ids or []), locators_by_id
                ),
                # model_validate (alias-keyed): `schema`/`yield` cannot be
                # passed by keyword (`yield` is a Python keyword).
                extraction=ReviewItemExtraction.model_validate(
                    {
                        "schema": structured.get("schema", "recipe.v1"),
                        "yield": structured.get("yield"),
                        "top_ingredients": top_ingredients(structured),
                        "confidence_overall": (item.confidence or {}).get("overall"),
                    }
                ),
                flags=build_review_reasons(
                    item.status.value,
                    structured,
                    confidence=item.confidence,
                    thresholds=thresholds_for_item(structured, current=current_thresholds)[0],
                ),
                edited_at=item.edited_at,
                favourited_at=favourited_at_by_id.get(item.id),
            )
        )
    return summaries
