"""Unit tests for ingestion.pipeline.persist.

``normalize_title`` is pure and tested directly. ``persist_knowledge_item`` is
exercised against a no-op fake session (``add`` records, ``flush`` is a no-op) so
the validation-driven status/warnings wiring is unit-tested without a database;
the real Postgres round-trip lives in tests/integration/test_persist_knowledge_item.py.

The JSONB write contract (DECISIONS #1) is pinned with SQLAlchemy attribute
history: a whole-object assignment is dirty-tracked, an in-place edit on a
committed value is not.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm.attributes import set_committed_value

from rag_recipes.ingestion.pipeline.persist import normalize_title, persist_knowledge_item
from rag_recipes.ingestion.validation import HardValidationError, SoftValidationThresholds
from rag_recipes.storage.enums import KnowledgeItemStatus
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from tests.unit.ingestion.test_validation import (
    _make_recipe,
    _make_structured_data,
    _make_window,
    _recipe_confidence,
)

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


# --- normalize_title -------------------------------------------------------


def test_normalize_title_canonical_example() -> None:
    assert normalize_title("  Tomato   and\tWhite\nBean  Soup ") == "tomato and white bean soup"


def test_normalize_title_lowercases() -> None:
    assert normalize_title("TOMATO Soup") == "tomato soup"


def test_normalize_title_collapses_internal_whitespace() -> None:
    assert normalize_title("a\t\t b   c") == "a b c"


def test_normalize_title_strips_edges() -> None:
    assert normalize_title("   hello   ") == "hello"


def test_normalize_title_nfc_normalizes_unicode() -> None:
    # Decomposed "e" + combining acute → precomposed "é", then lowered.
    assert normalize_title("Café") == "café"
    assert normalize_title("Café") == normalize_title("café")


def test_normalize_title_is_idempotent() -> None:
    once = normalize_title("  Tomato   and\tWhite\nBean  Soup ")
    assert normalize_title(once) == once


# --- persist_knowledge_item (fake session) ---------------------------------


class _FakeSession:
    """Minimal AsyncSession stand-in: records add()s, flush() is a no-op."""

    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        return None


@pytest.mark.asyncio
async def test_persist_clean_candidate_is_ready() -> None:
    session = _FakeSession()
    item = await persist_knowledge_item(
        session,  # type: ignore[arg-type]
        _make_recipe(),
        extraction_run_id="run_x",
        document_id="doc_x",
        source_version=1,
        window=_make_window(),
        thresholds=_THRESHOLDS,
    )
    assert item.status == KnowledgeItemStatus.READY
    assert item.structured_data["warnings"] == []
    assert item.normalized_title == "tomato and white bean soup"
    assert item.item_type == "recipe"
    assert item.extraction_run_id == "run_x"
    assert item.source_version == 1
    assert item.source_span_ids == ["span_001"]
    assert item.confidence is not None
    assert session.added == [item]


@pytest.mark.asyncio
async def test_persist_soft_candidate_is_needs_review() -> None:
    session = _FakeSession()
    recipe = _make_recipe(
        structured_data=_make_structured_data(steps=[]),
        confidence=_recipe_confidence(overall=0.3),
    )
    item = await persist_knowledge_item(
        session,  # type: ignore[arg-type]
        recipe,
        extraction_run_id="run_x",
        document_id="doc_x",
        source_version=1,
        window=_make_window(),
        thresholds=_THRESHOLDS,
    )
    assert item.status == KnowledgeItemStatus.NEEDS_REVIEW
    assert set(item.structured_data["warnings"]) == {"no_steps", "low_overall_confidence"}


@pytest.mark.asyncio
async def test_persist_hard_candidate_raises_and_adds_nothing() -> None:
    session = _FakeSession()
    recipe = _make_recipe(source_span_ids=["span_999"])
    with pytest.raises(HardValidationError) as excinfo:
        await persist_knowledge_item(
            session,  # type: ignore[arg-type]
            recipe,
            extraction_run_id="run_x",
            document_id="doc_x",
            source_version=1,
            window=_make_window(),
            thresholds=_THRESHOLDS,
        )
    assert excinfo.value.failures[0].code == "source_span_not_in_window"
    assert session.added == []


# --- persist_knowledge_item staging mode (Phase 9.5) -----------------------


@pytest.mark.asyncio
async def test_persist_staging_clean_is_extracting_with_score() -> None:
    session = _FakeSession()
    item = await persist_knowledge_item(
        session,  # type: ignore[arg-type]
        _make_recipe(),
        extraction_run_id="run_x",
        document_id="doc_x",
        source_version=1,
        window=_make_window(),
        thresholds=_THRESHOLDS,
        staging=True,
        candidate_score=0.8,
    )
    assert item.status == KnowledgeItemStatus.EXTRACTING
    assert item.candidate_score == 0.8
    # Warnings still stored so finalize can re-derive ready/needs_review.
    assert item.structured_data["warnings"] == []
    assert session.added == [item]


@pytest.mark.asyncio
async def test_persist_staging_soft_failing_keeps_warnings() -> None:
    session = _FakeSession()
    recipe = _make_recipe(
        structured_data=_make_structured_data(steps=[]),
        confidence=_recipe_confidence(overall=0.3),
    )
    item = await persist_knowledge_item(
        session,  # type: ignore[arg-type]
        recipe,
        extraction_run_id="run_x",
        document_id="doc_x",
        source_version=1,
        window=_make_window(),
        thresholds=_THRESHOLDS,
        staging=True,
        candidate_score=0.4,
    )
    # Staging status regardless of soft warnings; warnings preserved for finalize.
    assert item.status == KnowledgeItemStatus.EXTRACTING
    assert item.candidate_score == 0.4
    assert set(item.structured_data["warnings"]) == {"no_steps", "low_overall_confidence"}


@pytest.mark.asyncio
async def test_persist_staging_hard_fail_raises_and_adds_nothing() -> None:
    session = _FakeSession()
    recipe = _make_recipe(source_span_ids=["span_999"])
    with pytest.raises(HardValidationError):
        await persist_knowledge_item(
            session,  # type: ignore[arg-type]
            recipe,
            extraction_run_id="run_x",
            document_id="doc_x",
            source_version=1,
            window=_make_window(),
            thresholds=_THRESHOLDS,
            staging=True,
            candidate_score=0.9,
        )
    assert session.added == []


@pytest.mark.asyncio
async def test_persist_non_staging_leaves_candidate_score_none() -> None:
    session = _FakeSession()
    item = await persist_knowledge_item(
        session,  # type: ignore[arg-type]
        _make_recipe(),
        extraction_run_id="run_x",
        document_id="doc_x",
        source_version=1,
        window=_make_window(),
        thresholds=_THRESHOLDS,
    )
    assert item.status == KnowledgeItemStatus.READY
    assert item.candidate_score is None


# --- JSONB write contract (DECISIONS #1) -----------------------------------


def test_whole_object_assignment_is_dirty_tracked() -> None:
    item = KnowledgeItem()
    item.structured_data = {"warnings": []}
    history = inspect(item).attrs.structured_data.history
    assert history.added  # whole-object assignment IS tracked


def test_in_place_edit_on_committed_value_is_not_tracked() -> None:
    item = KnowledgeItem()
    # Simulate a value loaded from the DB (committed, no pending history).
    set_committed_value(item, "structured_data", {"warnings": []})
    # The footgun: editing the dict in place mutates the committed object itself,
    # so SQLAlchemy's identity comparison sees no change → silent data loss.
    item.structured_data["warnings"].append("x")
    history = inspect(item).attrs.structured_data.history
    assert not history.added


def test_reassignment_after_committed_value_is_tracked() -> None:
    item = KnowledgeItem()
    set_committed_value(item, "structured_data", {"warnings": []})
    # The documented safe path: build the full object and assign once.
    item.structured_data = {"warnings": ["x"]}
    history = inspect(item).attrs.structured_data.history
    assert history.added == [{"warnings": ["x"]}]
