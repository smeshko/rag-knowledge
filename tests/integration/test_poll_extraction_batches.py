"""Integration tests for the batch poller (Epic 19.3, TASK-004).

The poller opens its own sessions, so seeding is committed and cleaned up per
test. The Anthropic batch provider is faked via monkeypatching
``_build_batch_provider``; the worker ``ctx`` redis is an AsyncMock.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag_recipes.config import Settings, get_settings
from rag_recipes.ingestion import batch as batch_module
from rag_recipes.ingestion.batch import poll_extraction_batches
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.providers.llm.anthropic_batch import BatchResult, BatchStatus
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionBatchItemStatus,
    ExtractionBatchStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_batch import ExtractionBatch
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.session import build_session_factory

pytestmark = pytest.mark.asyncio

_TEST_CATEGORY = "batch-poller-test"


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 5
    output_tokens: int = 3


@dataclass
class _Message:
    content: list[_Block]
    usage: _Usage = field(default_factory=_Usage)
    stop_reason: str = "end_turn"
    stop_details: Any = None


def _msg(text: str) -> _Message:
    return _Message(content=[_Block(text=text)])


class _FakePollProvider:
    def __init__(self, *, status: str = "ended", results: list[BatchResult] | None = None) -> None:
        self._status = status
        self._results = results or []
        self.retrieve_calls: list[str] = []

    async def retrieve_batch(self, provider_batch_id: str) -> BatchStatus:
        self.retrieve_calls.append(provider_batch_id)
        return BatchStatus(processing_status=self._status)

    async def iter_results(self, provider_batch_id: str) -> AsyncIterator[BatchResult]:
        for result in self._results:
            yield result


@pytest_asyncio.fixture
async def session_factory(test_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return build_session_factory(test_engine)


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    yield
    async with session_factory() as session:
        doc_ids = list(
            (
                await session.execute(
                    select(Document.id).where(Document.category == _TEST_CATEGORY)
                )
            )
            .scalars()
            .all()
        )
        await session.execute(delete(ExtractionBatchItem))
        await session.execute(delete(ExtractionBatch))
        if doc_ids:
            await session.execute(
                delete(KnowledgeItem).where(KnowledgeItem.document_id.in_(doc_ids))
            )
            await session.execute(
                delete(ExtractionRun).where(ExtractionRun.document_id.in_(doc_ids))
            )
            await session.execute(
                delete(SourceSpan).where(SourceSpan.document_id.in_(doc_ids))
            )
            asset_ids = list(
                (
                    await session.execute(
                        select(Document.asset_id).where(Document.id.in_(doc_ids))
                    )
                )
                .scalars()
                .all()
            )
            await session.execute(delete(Document).where(Document.id.in_(doc_ids)))
            await session.execute(
                delete(SourceAsset).where(SourceAsset.id.in_(asset_ids))
            )
        await session.commit()


def _settings(**overrides: Any) -> Settings:
    base = {
        "llm_provider": "anthropic",
        "anthropic_api_key": "sk-ant-test",
        "anthropic_llm_model": "claude-sonnet-4-6",
    }
    base.update(overrides)
    return get_settings().model_copy(update=base)


async def _seed_submitted_batch(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: ExtractionBatchStatus = ExtractionBatchStatus.SUBMITTED,
    items: int = 2,
    provider_batch_id: str = "msgbatch_live",
) -> tuple[str, list[str]]:
    async with session_factory() as session:
        repo = DocumentRepository(session)
        pdf = new_id("x").encode()
        asset = await repo.add_source_asset(
            id=new_id("asset"),
            source_type=SourceType.PDF,
            original_filename="c.pdf",
            storage_provider="fake",
            storage_key=f"source-assets/{new_id('asset')}/o.pdf",
            content_hash=hashlib.sha256(pdf).hexdigest(),
            upload_status=UploadStatus.UPLOADED,
        )
        document = await repo.add_document(
            asset_id=asset.id,
            category=_TEST_CATEGORY,
            subcategory=None,
            title="C",
            author="",
            source_type=SourceType.PDF,
            language=None,
            active_source_version=None,
            status=DocumentStatus.EXTRACTING_ITEMS,
        )
        batch = ExtractionBatch(
            provider="anthropic",
            provider_batch_id=provider_batch_id,
            model="claude-sonnet-4-6",
            processing_status=status,
            request_count=items,
        )
        session.add(batch)
        await session.flush()
        item_objs = [
            ExtractionBatchItem(
                document_id=document.id,
                source_version=1,
                input_hash=f"hash-{document.id}-{i}",
                input_source_span_ids=["span_x"],
                request_input="window",
                request_schema={"type": "object"},
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                status=ExtractionBatchItemStatus.SUBMITTED,
                batch_id=batch.id,
            )
            for i in range(items)
        ]
        session.add_all(item_objs)
        await session.flush()  # populate the Python-side id default before we read it
        item_ids = [item.id for item in item_objs]
        await session.commit()
        return batch.id, item_ids


def _succeeded_results(item_ids: list[str]) -> list[BatchResult]:
    return [
        BatchResult(custom_id=iid, result_type="succeeded", message=_msg('{"items": []}'))
        for iid in item_ids
    ]


async def _runs_count(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        rows = (await session.execute(select(ExtractionRun.id))).scalars().all()
    return len(list(rows))


async def test_ended_batch_ingests_and_marks_ended(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id, item_ids = await _seed_submitted_batch(session_factory, items=2)
    fake = _FakePollProvider(status="ended", results=_succeeded_results(item_ids))
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    redis = AsyncMock()
    ctx = {"settings": _settings(), "session_factory": session_factory, "redis": redis}

    ingested = await poll_extraction_batches(ctx)

    assert ingested == 2
    assert await _runs_count(session_factory) == 2
    async with session_factory() as session:
        batch = await session.get(ExtractionBatch, batch_id)
        assert batch is not None
        assert batch.processing_status is ExtractionBatchStatus.ENDED
        assert batch.completed_at is not None
        items = (
            await session.execute(
                select(ExtractionBatchItem.status).where(
                    ExtractionBatchItem.batch_id == batch_id
                )
            )
        ).scalars().all()
        assert all(s is ExtractionBatchItemStatus.SUCCEEDED for s in items)
    # All items terminal → the doc is re-driven via the resume path.
    redis.enqueue_job.assert_awaited_once()


async def test_in_progress_batch_left_pending(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id, item_ids = await _seed_submitted_batch(session_factory, items=1)
    fake = _FakePollProvider(status="in_progress", results=_succeeded_results(item_ids))
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {
        "settings": _settings(),
        "session_factory": session_factory,
        "redis": AsyncMock(),
    }

    ingested = await poll_extraction_batches(ctx)

    assert ingested == 0
    assert await _runs_count(session_factory) == 0
    async with session_factory() as session:
        batch = await session.get(ExtractionBatch, batch_id)
        assert batch is not None
        assert batch.processing_status is ExtractionBatchStatus.IN_PROGRESS


async def test_idempotent_re_poll_creates_nothing(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, item_ids = await _seed_submitted_batch(session_factory, items=2)
    fake = _FakePollProvider(status="ended", results=_succeeded_results(item_ids))
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {
        "settings": _settings(),
        "session_factory": session_factory,
        "redis": AsyncMock(),
    }

    assert await poll_extraction_batches(ctx) == 2
    # Second poll: the batch is now ENDED (not pollable) → nothing ingested.
    assert await poll_extraction_batches(ctx) == 0
    assert await _runs_count(session_factory) == 2


async def test_provider_disabled_is_noop(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_submitted_batch(session_factory, items=1)

    def _boom(settings: Settings) -> Any:
        raise AssertionError("provider must not be built when disabled")

    monkeypatch.setattr(batch_module, "_build_batch_provider", _boom)
    ctx = {
        "settings": _settings(llm_provider="openai"),
        "session_factory": session_factory,
        "redis": AsyncMock(),
    }
    assert await poll_extraction_batches(ctx) == 0
    assert await _runs_count(session_factory) == 0


async def test_concurrent_polls_do_not_double_ingest(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, item_ids = await _seed_submitted_batch(session_factory, items=3)
    fake = _FakePollProvider(status="ended", results=_succeeded_results(item_ids))
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {
        "settings": _settings(),
        "session_factory": session_factory,
        "redis": AsyncMock(),
    }

    # FOR UPDATE SKIP LOCKED → one poller processes the batch, the other skips it.
    await asyncio.gather(
        poll_extraction_batches(ctx),
        poll_extraction_batches(ctx),
    )
    # Exactly one run per item — no double-ingest.
    assert await _runs_count(session_factory) == 3


class _MixedPollProvider:
    """Ended for both batches, but one batch's result stream raises mid-iteration."""

    def __init__(self, ok_results: dict[str, list[BatchResult]], raising_id: str) -> None:
        self._ok = ok_results
        self._raising_id = raising_id

    async def retrieve_batch(self, provider_batch_id: str) -> BatchStatus:
        return BatchStatus(processing_status="ended")

    async def iter_results(self, provider_batch_id: str) -> AsyncIterator[BatchResult]:
        if provider_batch_id == self._raising_id:
            # Simulate a normalized mid-stream provider failure (review #1.1).
            from rag_recipes.providers.errors import LLMTechnicalError

            raise LLMTechnicalError("stream dropped")
        for result in self._ok.get(provider_batch_id, []):
            yield result


async def test_healthy_doc_finalized_when_another_batch_errors(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Doc A's batch ends cleanly (→ complete → re-driven); Doc B's batch errors
    # mid-stream. The error must NOT prevent Doc A from being finalized this tick,
    # and Doc B's batch stays non-terminal for the next tick (review #1.1).
    good_batch, good_items = await _seed_submitted_batch(
        session_factory, items=2, provider_batch_id="mb_ok"
    )
    bad_batch, _ = await _seed_submitted_batch(
        session_factory, items=2, provider_batch_id="mb_err"
    )
    async with session_factory() as session:
        good_doc = await session.scalar(
            select(ExtractionBatchItem.document_id).where(
                ExtractionBatchItem.batch_id == good_batch
            )
        )
        bad_doc = await session.scalar(
            select(ExtractionBatchItem.document_id).where(
                ExtractionBatchItem.batch_id == bad_batch
            )
        )
    provider = _MixedPollProvider({"mb_ok": _succeeded_results(good_items)}, raising_id="mb_err")
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda s: provider)
    redis = AsyncMock()
    ctx = {"settings": _settings(), "session_factory": session_factory, "redis": redis}

    # Must not raise despite the mid-stream error on mb_err.
    ingested = await poll_extraction_batches(ctx)

    assert ingested == 2  # only Doc A's results
    # Doc A is complete → re-driven exactly once; Doc B is not.
    redis.enqueue_job.assert_awaited_once()
    assert redis.enqueue_job.await_args.args[0] == "process_document"
    assert redis.enqueue_job.await_args.args[1] == good_doc

    async with session_factory() as session:
        ok_batch = (
            await session.execute(
                select(ExtractionBatch).where(ExtractionBatch.provider_batch_id == "mb_ok")
            )
        ).scalar_one()
        err_batch = (
            await session.execute(
                select(ExtractionBatch).where(ExtractionBatch.provider_batch_id == "mb_err")
            )
        ).scalar_one()
        assert ok_batch.processing_status is ExtractionBatchStatus.ENDED
        # The errored batch is left non-terminal (rolled back) for the next tick.
        assert err_batch.processing_status in (
            ExtractionBatchStatus.SUBMITTED,
            ExtractionBatchStatus.IN_PROGRESS,
        )
        bad_items = (
            await session.execute(
                select(ExtractionBatchItem.status).where(
                    ExtractionBatchItem.document_id == bad_doc
                )
            )
        ).scalars().all()
        assert all(s is ExtractionBatchItemStatus.SUBMITTED for s in bad_items)
