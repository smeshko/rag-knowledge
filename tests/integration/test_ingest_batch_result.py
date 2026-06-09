"""Integration tests for ingest_batch_result (Epic 19.3, TASK-003).

Uses the transactional ``db_session`` fixture: ``ingest_batch_result`` never
commits (the poller owns the commit), so seeding + ingest + asserts all live in
one rolled-back transaction. Provider results are synthetic ``BatchResult``s with
a fake Anthropic ``Message`` so no network call happens.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.config import get_settings
from rag_recipes.ingestion.batch import ingest_batch_result
from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    RecipeExtractionOutput,
)
from rag_recipes.providers.llm.anthropic_batch import BatchResult
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionBatchItemStatus,
    ExtractionBatchStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_batch import ExtractionBatch
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from tests.unit.ingestion.test_validation import _make_recipe, _make_step, _make_structured_data

pytestmark = pytest.mark.asyncio


# --- fake Anthropic Message for the shared mapping --------------------------


@dataclass
class _ToolUseBlock:
    input: Any
    name: str = "structured_output"
    type: str = "tool_use"


@dataclass
class _Usage:
    input_tokens: int = 5
    output_tokens: int = 3


@dataclass
class _StopDetails:
    explanation: str | None = None


@dataclass
class _Message:
    content: list[Any]
    usage: _Usage = field(default_factory=_Usage)
    stop_reason: str = "tool_use"
    stop_details: _StopDetails | None = None


def _msg(
    tool_input: Any = None, *, stop_reason: str = "tool_use", explanation: str | None = None
) -> _Message:
    """A succeeded forced-tool message carries ``tool_input`` (a dict) in a tool_use
    block; a refusal (``tool_input=None``) carries no tool_use block."""
    content = [] if tool_input is None else [_ToolUseBlock(input=tool_input)]
    return _Message(
        content=content,
        stop_reason=stop_reason,
        stop_details=_StopDetails(explanation=explanation) if explanation else None,
    )


def _succeeded(tool_input: Any) -> BatchResult:
    return BatchResult(custom_id="c", result_type="succeeded", message=_msg(tool_input))


# --- seeding ---------------------------------------------------------------


async def _seed(
    session: AsyncSession,
    *,
    status: ExtractionBatchItemStatus = ExtractionBatchItemStatus.SUBMITTED,
    schema_version: str = SCHEMA_VERSION,
    submit_attempts: int = 0,
    batch_model: str = "claude-sonnet-4-6",
    input_hash: str = "hash-1",
    pages: int = 2,
) -> tuple[ExtractionBatchItem, list[SourceSpan]]:
    repo = DocumentRepository(session)
    pdf = b"%PDF-1.4 ingest fixture"
    asset = await repo.add_source_asset(
        id=new_id("asset"),
        source_type=SourceType.PDF,
        original_filename="c.pdf",
        storage_provider="fake",
        storage_key=f"source-assets/{new_id('asset')}/o.pdf",
        content_hash=hashlib.sha256(pdf + input_hash.encode()).hexdigest(),
        upload_status=UploadStatus.UPLOADED,
    )
    document = await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="C",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.EXTRACTING_ITEMS,
    )
    spans = []
    for page in range(1, pages + 1):
        locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
        text = f"Page {page} recipe text. Heat the oil. Serves {page}." * 6
        span = SourceSpan(
            document_id=document.id,
            source_version=1,
            source_type=SourceType.PDF,
            locator=locator,
            locator_hash=hashlib.sha256(f"{document.id}{page}".encode()).hexdigest(),
            text=text,
            text_hash=hashlib.sha256(text.encode()).hexdigest(),
        )
        session.add(span)
        spans.append(span)
    await session.flush()

    batch = ExtractionBatch(
        provider="anthropic",
        provider_batch_id="msgbatch_x",
        model=batch_model,
        processing_status=ExtractionBatchStatus.SUBMITTED,
        request_count=1,
    )
    session.add(batch)
    await session.flush()

    item = ExtractionBatchItem(
        document_id=document.id,
        source_version=1,
        input_hash=input_hash,
        input_source_span_ids=[s.id for s in spans],
        request_input="window",
        request_schema={"type": "object"},
        prompt_version=PROMPT_VERSION,
        schema_version=schema_version,
        status=status,
        batch_id=batch.id,
        submit_attempts=submit_attempts,
    )
    session.add(item)
    await session.flush()
    return item, spans


async def _runs(session: AsyncSession, document_id: str) -> list[ExtractionRun]:
    return list(
        (
            await session.execute(
                select(ExtractionRun).where(ExtractionRun.document_id == document_id)
            )
        )
        .scalars()
        .all()
    )


_SETTINGS = get_settings().model_copy(
    update={"anthropic_llm_model": "claude-sonnet-4-6", "anthropic_batch_max_submit_attempts": 2}
)


async def test_succeeded_empty_items_creates_success_run(db_session: AsyncSession) -> None:
    item, _ = await _seed(db_session)
    await ingest_batch_result(db_session, item, _succeeded({"items": []}), settings=_SETTINGS)

    runs = await _runs(db_session, item.document_id)
    assert len(runs) == 1
    assert runs[0].status is ExtractionRunStatus.SUCCESS
    assert runs[0].input_hash == item.input_hash
    assert runs[0].provider == "anthropic"
    assert runs[0].model == "claude-sonnet-4-6"  # sourced from the batch
    assert item.status is ExtractionBatchItemStatus.SUCCEEDED
    assert item.result_type == "succeeded"


async def test_succeeded_with_recipe_persists_staging_candidate(
    db_session: AsyncSession,
) -> None:
    item, spans = await _seed(db_session)
    span_id = spans[0].id
    recipe = _make_recipe(
        source_span_ids=[span_id],
        structured_data=_make_structured_data(steps=[_make_step(source_span_ids=[span_id])]),
    )
    payload = RecipeExtractionOutput(items=[recipe]).model_dump(by_alias=True, mode="json")
    await ingest_batch_result(db_session, item, _succeeded(payload), settings=_SETTINGS)

    runs = await _runs(db_session, item.document_id)
    assert len(runs) == 1 and runs[0].status is ExtractionRunStatus.SUCCESS
    kis = list(
        (
            await db_session.execute(
                select(KnowledgeItem).where(KnowledgeItem.document_id == item.document_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(kis) == 1
    assert kis[0].status is KnowledgeItemStatus.EXTRACTING
    assert kis[0].candidate_score is not None
    assert kis[0].extraction_run_id == runs[0].id
    assert item.status is ExtractionBatchItemStatus.SUCCEEDED


async def test_succeeded_refusal_rejected(db_session: AsyncSession) -> None:
    item, _ = await _seed(db_session)
    result = BatchResult(
        custom_id="c",
        result_type="succeeded",
        message=_msg(None, stop_reason="refusal", explanation="nope"),
    )
    await ingest_batch_result(db_session, item, result, settings=_SETTINGS)

    runs = await _runs(db_session, item.document_id)
    assert len(runs) == 1 and runs[0].status is ExtractionRunStatus.REJECTED
    assert runs[0].error_message and "refused" in runs[0].error_message
    assert item.status is ExtractionBatchItemStatus.REJECTED


async def test_succeeded_pydantic_invalid_rejected_with_output(
    db_session: AsyncSession,
) -> None:
    item, _ = await _seed(db_session)
    await ingest_batch_result(
        db_session, item, _succeeded({"items": "not-a-list"}), settings=_SETTINGS
    )
    runs = await _runs(db_session, item.document_id)
    assert len(runs) == 1 and runs[0].status is ExtractionRunStatus.REJECTED
    assert runs[0].output_json == {"items": "not-a-list"}
    assert item.status is ExtractionBatchItemStatus.REJECTED


async def test_errored_invalid_request_rejected(db_session: AsyncSession) -> None:
    item, _ = await _seed(db_session)
    result = BatchResult(
        custom_id="c", result_type="errored", error_type="invalid_request_error", retryable=False
    )
    await ingest_batch_result(db_session, item, result, settings=_SETTINGS)
    runs = await _runs(db_session, item.document_id)
    assert len(runs) == 1 and runs[0].status is ExtractionRunStatus.REJECTED
    assert item.status is ExtractionBatchItemStatus.REJECTED
    assert item.result_type == "errored"


async def test_canceled_rejected(db_session: AsyncSession) -> None:
    item, _ = await _seed(db_session)
    await ingest_batch_result(
        db_session, item, BatchResult(custom_id="c", result_type="canceled"), settings=_SETTINGS
    )
    runs = await _runs(db_session, item.document_id)
    assert len(runs) == 1 and runs[0].status is ExtractionRunStatus.REJECTED
    assert item.status is ExtractionBatchItemStatus.CANCELED


async def test_expired_under_cap_reverts_and_bumps_heartbeat(
    db_session: AsyncSession,
) -> None:
    item, _ = await _seed(db_session, submit_attempts=0)
    await ingest_batch_result(
        db_session, item, BatchResult(custom_id="c", result_type="expired"), settings=_SETTINGS
    )
    # No run; item reverted to PENDING for re-submission, attempts bumped.
    assert await _runs(db_session, item.document_id) == []
    assert item.status is ExtractionBatchItemStatus.PENDING
    assert item.batch_id is None
    assert item.submit_attempts == 1
    # Heartbeat bumped so 19.2's stale-PENDING sweep doesn't reap the retrying doc.
    doc = await db_session.get(Document, item.document_id)
    assert doc is not None and doc.last_progress_at is not None


async def test_expired_at_cap_rejected(db_session: AsyncSession) -> None:
    # submit_attempts already at the cap (2) → no more retry, REJECTED audit run.
    item, _ = await _seed(db_session, submit_attempts=2)
    await ingest_batch_result(
        db_session, item, BatchResult(custom_id="c", result_type="expired"), settings=_SETTINGS
    )
    runs = await _runs(db_session, item.document_id)
    assert len(runs) == 1 and runs[0].status is ExtractionRunStatus.REJECTED
    assert item.status is ExtractionBatchItemStatus.EXPIRED


async def test_idempotent_replay_creates_nothing(db_session: AsyncSession) -> None:
    item, _ = await _seed(db_session)
    await ingest_batch_result(db_session, item, _succeeded({"items": []}), settings=_SETTINGS)
    runs_after_first = await _runs(db_session, item.document_id)
    assert len(runs_after_first) == 1
    # Item is now terminal; a second ingest must no-op (status guard).
    await ingest_batch_result(db_session, item, _succeeded({"items": []}), settings=_SETTINGS)
    assert len(await _runs(db_session, item.document_id)) == 1


async def test_idempotent_duplicate_input_hash_run_exists(db_session: AsyncSession) -> None:
    # Simulate 19.2 reconcile re-submit: a second SUBMITTED item for the same
    # window (same input_hash) whose run already exists → no new run, item terminal.
    item, spans = await _seed(db_session, input_hash="dup")
    # Pre-existing terminal run for this (doc, version, input_hash).
    db_session.add(
        ExtractionRun(
            document_id=item.document_id,
            source_version=1,
            provider="anthropic",
            model="claude-sonnet-4-6",
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            input_source_span_ids=item.input_source_span_ids,
            input_hash="dup",
            status=ExtractionRunStatus.SUCCESS,
            output_json={"items": []},
        )
    )
    await db_session.flush()
    await ingest_batch_result(db_session, item, _succeeded({"items": []}), settings=_SETTINGS)
    # Still exactly one run; the duplicate item converges terminal.
    assert len(await _runs(db_session, item.document_id)) == 1
    assert item.status is ExtractionBatchItemStatus.SUCCEEDED
