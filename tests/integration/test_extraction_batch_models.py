"""Integration tests for the extraction-batch tracking tables (Epic 19.2).

Runs against real Postgres (``test_engine`` applies ``alembic upgrade head``, so
this exercises the phase-19.2 migration). Proves the round-trip, the nullable
``batch_id``, and the registration-idempotency partial-unique index.
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionBatchItemStatus,
    ExtractionBatchStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models import ExtractionBatch, ExtractionBatchItem
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio


async def _make_document(session: AsyncSession) -> str:
    repo = DocumentRepository(session)
    pdf_bytes = b"%PDF-1.4 batch fixture"
    asset = await repo.add_source_asset(
        id=new_id("asset"),
        source_type=SourceType.PDF,
        original_filename="cookbook.pdf",
        storage_provider="fake",
        storage_key=f"source-assets/{new_id('asset')}/original.pdf",
        content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
        upload_status=UploadStatus.UPLOADED,
    )
    document = await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="Cookbook",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.QUEUED,
    )
    await session.flush()
    return document.id


def _item(
    document_id: str, *, input_hash: str, status: ExtractionBatchItemStatus
) -> ExtractionBatchItem:
    return ExtractionBatchItem(
        document_id=document_id,
        source_version=1,
        input_hash=input_hash,
        input_source_span_ids=["span_a", "span_b"],
        request_input="extract this window",
        request_schema={"type": "object", "additionalProperties": False},
        prompt_version="recipe-extraction-v1",
        schema_version="recipe.v1",
        status=status,
    )


async def test_batch_and_item_round_trip(db_session: AsyncSession) -> None:
    document_id = await _make_document(db_session)
    item = _item(document_id, input_hash="h1", status=ExtractionBatchItemStatus.PENDING)
    db_session.add(item)
    await db_session.flush()

    # Item can exist with batch_id None (registered before any batch exists).
    assert item.batch_id is None

    batch = ExtractionBatch(
        provider="anthropic",
        provider_batch_id=None,
        model="claude-sonnet-4-6",
        processing_status=ExtractionBatchStatus.SUBMITTING,
        request_count=1,
    )
    db_session.add(batch)
    await db_session.flush()
    item.batch_id = batch.id
    item.status = ExtractionBatchItemStatus.SUBMITTING
    await db_session.flush()

    reloaded = (
        await db_session.execute(
            select(ExtractionBatchItem).where(ExtractionBatchItem.id == item.id)
        )
    ).scalar_one()
    assert reloaded.batch_id == batch.id
    assert reloaded.request_schema == {"type": "object", "additionalProperties": False}
    assert reloaded.input_source_span_ids == ["span_a", "span_b"]
    # Epic 19.3 retry/audit columns: submit_attempts backfills to 0; audit nullable.
    assert reloaded.submit_attempts == 0
    assert reloaded.result_type is None
    assert reloaded.error_message is None


async def test_partial_unique_index_blocks_duplicate_non_terminal(
    db_session: AsyncSession,
) -> None:
    document_id = await _make_document(db_session)
    db_session.add(_item(document_id, input_hash="dup", status=ExtractionBatchItemStatus.PENDING))
    await db_session.flush()

    db_session.add(
        _item(document_id, input_hash="dup", status=ExtractionBatchItemStatus.SUBMITTING)
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_terminal_item_does_not_block_re_registration(
    db_session: AsyncSession,
) -> None:
    document_id = await _make_document(db_session)
    first = _item(document_id, input_hash="reg", status=ExtractionBatchItemStatus.REJECTED)
    db_session.add(first)
    await db_session.flush()

    # A terminal (REJECTED) item is outside the partial index, so a fresh
    # non-terminal registration of the same window is allowed.
    second = _item(document_id, input_hash="reg", status=ExtractionBatchItemStatus.PENDING)
    db_session.add(second)
    await db_session.flush()
    assert second.id != first.id
