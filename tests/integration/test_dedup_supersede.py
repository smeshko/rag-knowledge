"""Integration tests for DocumentRepository.supersede_prior_items.

Real Postgres (``test_engine`` / ``db_session``). Seeds one document with two
``ExtractionRun``s and three ``KnowledgeItem`` rows, then exercises the bulk
supersede UPDATE: a keep-set sparing the "new" run flips both "old" items to
``SUPERSEDED`` and leaves the new one untouched; a keep-set covering every run is
the first-extraction no-op (0 affected); and an already-``SUPERSEDED`` row is
never re-touched.
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
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
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio

SOURCE_VERSION = 1


def _make_run(document_id: str, input_hash: str) -> ExtractionRun:
    return ExtractionRun(
        document_id=document_id,
        source_version=SOURCE_VERSION,
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=[],
        input_hash=input_hash,
        status=ExtractionRunStatus.SUCCESS,
        output_json=None,
    )


def _make_item(
    document_id: str,
    run_id: str,
    *,
    title: str,
    status: KnowledgeItemStatus,
) -> KnowledgeItem:
    return KnowledgeItem(
        document_id=document_id,
        extraction_run_id=run_id,
        source_version=SOURCE_VERSION,
        item_type="recipe",
        title=title,
        normalized_title=title.lower(),
        summary=None,
        body_text="x" * 100,
        source_span_ids=[],
        structured_data={"warnings": []},
        confidence=None,
        status=status,
    )


async def _seed(session: AsyncSession) -> tuple[str, str, str]:
    """Insert a Document + two ExtractionRuns. Returns (doc_id, old_run, new_run)."""
    repo = DocumentRepository(session)
    pdf_bytes = b"%PDF-1.4 supersede fixture"
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
    old_run = _make_run(document.id, "hash_old")
    new_run = _make_run(document.id, "hash_new")
    session.add_all([old_run, new_run])
    await session.flush()
    return document.id, old_run.id, new_run.id


async def _status_by_id(
    session: AsyncSession, item_ids: list[str]
) -> dict[str, KnowledgeItemStatus]:
    """Re-read statuses from the DB (the bulk UPDATE does not sync the ORM identity map)."""
    session.expire_all()
    out: dict[str, KnowledgeItemStatus] = {}
    for item_id in item_ids:
        item = await session.get(KnowledgeItem, item_id)
        assert item is not None
        out[item_id] = item.status
    return out


async def test_supersede_spares_kept_run_and_flips_the_rest(db_session: AsyncSession) -> None:
    document_id, old_run, new_run = await _seed(db_session)
    old_ready = _make_item(document_id, old_run, title="Soup A", status=KnowledgeItemStatus.READY)
    old_review = _make_item(
        document_id, old_run, title="Soup B", status=KnowledgeItemStatus.NEEDS_REVIEW
    )
    new_ready = _make_item(document_id, new_run, title="Soup C", status=KnowledgeItemStatus.READY)
    db_session.add_all([old_ready, old_review, new_ready])
    await db_session.flush()

    repo = DocumentRepository(db_session)
    affected = await repo.supersede_prior_items(
        document_id, keep_extraction_run_ids={new_run}
    )

    assert affected == 2
    statuses = await _status_by_id(
        db_session, [old_ready.id, old_review.id, new_ready.id]
    )
    assert statuses[old_ready.id] == KnowledgeItemStatus.SUPERSEDED
    assert statuses[old_review.id] == KnowledgeItemStatus.SUPERSEDED
    assert statuses[new_ready.id] == KnowledgeItemStatus.READY


async def test_supersede_is_noop_when_keep_covers_all_runs(db_session: AsyncSession) -> None:
    document_id, old_run, new_run = await _seed(db_session)
    item = _make_item(document_id, old_run, title="Soup A", status=KnowledgeItemStatus.READY)
    db_session.add(item)
    await db_session.flush()

    repo = DocumentRepository(db_session)
    affected = await repo.supersede_prior_items(
        document_id, keep_extraction_run_ids={old_run, new_run}
    )

    assert affected == 0
    statuses = await _status_by_id(db_session, [item.id])
    assert statuses[item.id] == KnowledgeItemStatus.READY


async def test_supersede_skips_already_superseded_rows(db_session: AsyncSession) -> None:
    document_id, old_run, new_run = await _seed(db_session)
    already = _make_item(
        document_id, old_run, title="Soup A", status=KnowledgeItemStatus.SUPERSEDED
    )
    fresh = _make_item(document_id, old_run, title="Soup B", status=KnowledgeItemStatus.READY)
    db_session.add_all([already, fresh])
    await db_session.flush()

    repo = DocumentRepository(db_session)
    affected = await repo.supersede_prior_items(
        document_id, keep_extraction_run_ids={new_run}
    )

    # Only the READY row is counted/flipped; the already-SUPERSEDED row is excluded.
    assert affected == 1
    statuses = await _status_by_id(db_session, [already.id, fresh.id])
    assert statuses[already.id] == KnowledgeItemStatus.SUPERSEDED
    assert statuses[fresh.id] == KnowledgeItemStatus.SUPERSEDED
