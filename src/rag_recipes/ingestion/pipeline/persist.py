"""Persist a validated recipe candidate as a ``KnowledgeItem`` (doc 2 § 4, doc 4).

Turns a parsed ``recipe.v1`` ``ExtractedRecipe`` (Phase 9.2) plus its source
``Window`` (Phase 9.1) into the correct persistence outcome:

- hard validation fails  → persist **no** row; raise ``HardValidationError``.
- soft validation fires  → persist a ``KnowledgeItem`` with ``status="needs_review"``
  and the warning codes attached.
- fully clean            → persist a ``KnowledgeItem`` with ``status="ready"``.

In Phase 9.5 ``staging`` mode the row is instead written in the ``extracting``
staging status carrying its ``candidate_score``; warnings are still stored so
finalize re-derives the eventual ready/needs_review status (DECISIONS #1).

**JSONB write contract (DECISIONS #1):** build the full object, assign once;
never edit a JSONB attribute (``structured_data``, ``confidence``,
``source_span_ids``) in place. The columns are plain ``postgresql.JSONB`` with no
``MutableDict``/``MutableList`` wrapping, so an in-place edit after the row is
attached is NOT dirty-tracked and is silently dropped on commit. New rows here
are dirty by construction (every attribute is set once), so this build-then-assign
discipline satisfies the contract for inserts.

The caller owns the transaction: this module ``flush()``es (to surface FK /
``@validates`` errors at the call site) but never commits — mirroring
``pipeline/pdf_text`` and ``pipeline/extraction``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.config import get_settings
from rag_recipes.ingestion.pipeline.composition import normalize_title
from rag_recipes.ingestion.pipeline.extraction import ExtractedRecipe
from rag_recipes.ingestion.pipeline.windows import Window
from rag_recipes.ingestion.validation import (
    HardValidationError,
    SoftValidationThresholds,
    validate_hard,
    validate_soft,
)
from rag_recipes.storage.enums import KnowledgeItemStatus
from rag_recipes.storage.models.knowledge_item import KnowledgeItem

# ``normalize_title`` now lives in the pure ``pipeline/composition`` module (so
# the Epic 22.1 edit layer can reach it without importing a session or
# ``Settings``) and is re-exported here for this module's existing callers.
__all__ = ["normalize_title", "persist_knowledge_item", "thresholds_from_settings"]


def thresholds_from_settings() -> SoftValidationThresholds:
    """Build soft-validation thresholds from the application ``Settings``.

    Used when the caller does not pass ``thresholds`` explicitly, and by the
    Epic 22.2 edit endpoint, which must re-derive warnings against exactly the
    thresholds ingest used. Tests pass their own value object, so they exercise
    ``persist_knowledge_item`` without needing a populated environment.
    """
    settings = get_settings()
    return SoftValidationThresholds(
        min_overall_confidence=settings.extraction_min_overall_confidence,
        min_boundary_confidence=settings.extraction_min_boundary_confidence,
        min_normalization_confidence=settings.extraction_min_normalization_confidence,
        min_recipe_chars=settings.extraction_min_recipe_chars,
        max_recipe_chars=settings.extraction_max_recipe_chars,
    )


async def persist_knowledge_item(
    session: AsyncSession,
    extracted: ExtractedRecipe,
    *,
    extraction_run_id: str,
    document_id: str,
    source_version: int,
    window: Window,
    thresholds: SoftValidationThresholds | None = None,
    staging: bool = False,
    candidate_score: float | None = None,
) -> KnowledgeItem:
    """Validate ``extracted`` and persist a ``KnowledgeItem`` (or reject it).

    Runs hard validation first: any failure persists nothing and raises
    ``HardValidationError`` carrying the full failure list (the run is left
    untouched — DECISIONS #3). Otherwise runs soft validation and persists a row.

    When ``staging`` is false (default), the row's final status is set directly:
    ``status=NEEDS_REVIEW`` (warning codes in ``structured_data["warnings"]``)
    when any soft rule fires, else ``status=READY``.

    When ``staging`` is true (Phase 9.5), the row is written in the
    ``EXTRACTING`` staging status with ``candidate_score`` stored for the
    separate finalize transaction's dedup pass (DECISIONS #3). Warnings are
    still computed and stored in ``structured_data["warnings"]`` exactly as in
    the non-staging path, so finalize can re-derive the eventual ready/
    needs_review status purely from the persisted warnings (DECISIONS #1).

    Flushes before returning so composite-FK / ``@validates`` violations surface
    here; never commits. ``thresholds`` defaults to the application ``Settings``.
    """
    failures = validate_hard(extracted, window)
    if failures:
        raise HardValidationError(failures)

    if thresholds is None:
        thresholds = thresholds_from_settings()
    warnings = validate_soft(extracted, thresholds=thresholds)

    # Whole-object assembly (DECISIONS #1, #2): warning codes live alongside the
    # LLM's own warnings under structured_data; never mutated in place afterwards.
    # Stored regardless of staging — they are the source of truth from which
    # finalize re-derives the final ready/needs_review status (DECISIONS #1).
    structured_data = {
        **extracted.structured_data.model_dump(mode="json", by_alias=True),
        "warnings": [w.code for w in warnings],
    }
    if staging:
        status = KnowledgeItemStatus.EXTRACTING
    else:
        status = KnowledgeItemStatus.NEEDS_REVIEW if warnings else KnowledgeItemStatus.READY

    item = KnowledgeItem(
        document_id=document_id,
        extraction_run_id=extraction_run_id,
        source_version=source_version,
        item_type="recipe",
        title=extracted.title,
        normalized_title=normalize_title(extracted.title),
        summary=extracted.summary,
        body_text=extracted.body_text,
        source_span_ids=list(extracted.source_span_ids),
        structured_data=structured_data,
        confidence=extracted.confidence.model_dump(mode="json", by_alias=True),
        status=status,
        candidate_score=candidate_score,
    )
    session.add(item)
    await session.flush()
    return item
