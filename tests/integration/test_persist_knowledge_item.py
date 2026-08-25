"""Integration tests for ingestion.pipeline.persist.persist_knowledge_item.

Real Postgres (``test_engine`` / ``db_session``). One curated document yields all
three outcomes — ready, needs_review, and hard-fail (no row) — and the JSONB
columns are re-read after a real commit to prove the whole-object write contract
(DECISIONS #1) round-trips intact.
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
)
from rag_recipes.ingestion.pipeline.persist import persist_knowledge_item
from rag_recipes.ingestion.pipeline.windows import Window, compute_input_hash
from rag_recipes.ingestion.validation import HardValidationError, SoftValidationThresholds
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from tests.unit.ingestion.test_validation import (
    _make_recipe,
    _make_step,
    _make_structured_data,
    _recipe_confidence,
)

pytestmark = pytest.mark.asyncio

SOURCE_VERSION = 1

_THRESHOLDS = SoftValidationThresholds(
    min_overall_confidence=0.5,
    min_boundary_confidence=0.5,
    min_normalization_confidence=0.5,
    min_recipe_chars=200,
    max_recipe_chars=20000,
    assembly_min_ingredients=3,
    assembly_max_ingredients=12,
    assembly_max_chars=400,
)


def _make_span(document_id: str, page: int, text: str) -> SourceSpan:
    locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
    return SourceSpan(
        document_id=document_id,
        source_version=SOURCE_VERSION,
        source_type=SourceType.PDF,
        locator=locator,
        locator_hash=hashlib.sha256(str(locator).encode()).hexdigest(),
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


async def _setup(session: AsyncSession) -> tuple[str, str, Window]:
    """Insert a Document + two SourceSpans + a SUCCESS ExtractionRun.

    Returns ``(document_id, extraction_run_id, window)`` with consistent ids and
    source_version so the KnowledgeItem composite FK / @validates accept inserts.
    """
    repo = DocumentRepository(session)
    pdf_bytes = b"%PDF-1.4 persist fixture"
    asset = await repo.add_source_asset(
        id=new_id("asset"),
        source_type=SourceType.PDF,
        original_filename="recipe.pdf",
        storage_provider="fake",
        storage_key=f"source-assets/{new_id('asset')}/original.pdf",
        content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
        upload_status=UploadStatus.UPLOADED,
    )
    document = await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="My Recipe",
        author="Alice",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.QUEUED,
    )
    spans = [
        _make_span(document.id, 1, "Tomato and White Bean Soup\nServes 4"),
        _make_span(document.id, 2, "Heat the oil in a large pot."),
    ]
    session.add_all(spans)
    await session.flush()
    window = Window(spans=tuple(spans))

    run = ExtractionRun(
        document_id=document.id,
        source_version=SOURCE_VERSION,
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=window.span_ids,
        input_hash=compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION),
        status=ExtractionRunStatus.SUCCESS,
        output_json=None,
    )
    session.add(run)
    await session.flush()
    return document.id, run.id, window


def _clean_recipe(window: Window):  # type: ignore[no-untyped-def]
    span_a, span_b = window.span_ids
    return _make_recipe(
        source_span_ids=[span_a],
        structured_data=_make_structured_data(steps=[_make_step(source_span_ids=[span_b])]),
    )


def _soft_recipe(window: Window):  # type: ignore[no-untyped-def]
    span_a, _ = window.span_ids
    return _make_recipe(
        source_span_ids=[span_a],
        structured_data=_make_structured_data(steps=[]),
        confidence=_recipe_confidence(overall=0.3),
    )


def _hard_recipe(window: Window):  # type: ignore[no-untyped-def]
    span_a, span_b = window.span_ids
    return _make_recipe(
        source_span_ids=["span_not_in_window"],
        structured_data=_make_structured_data(steps=[_make_step(source_span_ids=[span_b])]),
    )


async def test_persist_three_outcomes_and_jsonb_round_trip(db_session: AsyncSession) -> None:
    document_id, run_id, window = await _setup(db_session)
    span_a = window.span_ids[0]

    # --- clean → ready ---
    clean_recipe = _clean_recipe(window)
    clean = await persist_knowledge_item(
        db_session,
        clean_recipe,
        extraction_run_id=run_id,
        document_id=document_id,
        source_version=SOURCE_VERSION,
        window=window,
        thresholds=_THRESHOLDS,
    )
    assert clean.status == KnowledgeItemStatus.READY
    assert clean.structured_data["warnings"] == []
    assert clean.normalized_title == "tomato and white bean soup"
    assert clean.source_span_ids == [span_a]

    # --- soft → needs_review with warnings ---
    soft = await persist_knowledge_item(
        db_session,
        _soft_recipe(window),
        extraction_run_id=run_id,
        document_id=document_id,
        source_version=SOURCE_VERSION,
        window=window,
        thresholds=_THRESHOLDS,
    )
    assert soft.status == KnowledgeItemStatus.NEEDS_REVIEW
    assert set(soft.structured_data["warnings"]) == {"no_steps", "low_overall_confidence"}

    # --- hard → raises, no row ---
    with pytest.raises(HardValidationError) as excinfo:
        await persist_knowledge_item(
            db_session,
            _hard_recipe(window),
            extraction_run_id=run_id,
            document_id=document_id,
            source_version=SOURCE_VERSION,
            window=window,
            thresholds=_THRESHOLDS,
        )
    assert excinfo.value.failures[0].code == "source_span_not_in_window"

    # Exactly the two valid candidates were inserted (hard-fail added nothing).
    count = await db_session.scalar(
        select(func.count()).select_from(KnowledgeItem).where(
            KnowledgeItem.document_id == document_id
        )
    )
    assert count == 2

    # The run is untouched by candidate-level outcomes (DECISIONS #3).
    run = await db_session.get(ExtractionRun, run_id)
    assert run is not None
    assert run.status == ExtractionRunStatus.SUCCESS

    expected_structured = {
        **clean_recipe.structured_data.model_dump(mode="json", by_alias=True),
        "warnings": [],
    }
    expected_confidence = clean_recipe.confidence.model_dump(mode="json", by_alias=True)
    clean_id = clean.id

    # Commit, drop ORM state, and re-read from Postgres: proves the JSONB write
    # contract persisted (the core regression guard).
    await db_session.commit()
    db_session.expire_all()

    reloaded = await db_session.get(KnowledgeItem, clean_id)
    assert reloaded is not None
    assert reloaded.status == KnowledgeItemStatus.READY
    assert reloaded.structured_data == expected_structured
    assert reloaded.confidence == expected_confidence
    assert reloaded.source_span_ids == [span_a]
    assert reloaded.normalized_title == "tomato and white bean soup"
