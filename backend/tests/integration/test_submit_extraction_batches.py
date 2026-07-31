"""Integration tests for the cron submitter (Epic 19.2, TASK-005).

The submitter drains *all* PENDING ExtractionBatchItems globally, so each test
cleans the two batch tables (and its docs) afterwards to stay isolated. The
Anthropic batch provider is replaced with a fake via monkeypatching
``_build_batch_provider`` so no network call happens.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag_recipes.config import Settings, get_settings
from rag_recipes.ingestion import batch as batch_module
from rag_recipes.ingestion.batch import submit_extraction_batches
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.anthropic_batch import (
    BatchExtractionRequest,
    BatchSubmitResult,
)
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
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.session import build_session_factory

pytestmark = pytest.mark.asyncio

_TEST_CATEGORY = "batch-submitter-test"


class _FakeBatchProvider:
    def __init__(self, *, fail: bool = False, on_submit: Any = None) -> None:
        self.submit_calls: list[tuple[list[BatchExtractionRequest], str | None]] = []
        self._fail = fail
        self._on_submit = on_submit
        self._counter = 0

    async def submit_batch(
        self,
        requests: Sequence[BatchExtractionRequest],
        *,
        idempotency_key: str | None = None,
    ) -> BatchSubmitResult:
        self.submit_calls.append((list(requests), idempotency_key))
        if self._on_submit is not None:
            await self._on_submit(list(requests), idempotency_key)
        if self._fail:
            raise LLMTechnicalError("simulated pre-acceptance failure")
        self._counter += 1
        return BatchSubmitResult(
            provider_batch_id=f"msgbatch_{self._counter}", request_count=len(requests)
        )


@pytest_asyncio.fixture
async def session_factory(test_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return build_session_factory(test_engine)


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_batches(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    yield
    async with session_factory() as session:
        await session.execute(delete(ExtractionBatchItem))
        await session.execute(delete(ExtractionBatch))
        doc_ids = list(
            (
                await session.execute(
                    select(Document.id).where(Document.category == _TEST_CATEGORY)
                )
            )
            .scalars()
            .all()
        )
        if doc_ids:
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
        "anthropic_max_tokens": 8192,
    }
    base.update(overrides)
    return get_settings().model_copy(update=base)


async def _make_document(session: AsyncSession) -> str:
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
    await session.flush()
    return document.id


async def _seed_pending_items(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    count: int,
    schema_version: str = SCHEMA_VERSION,
    request_input: str = "extract this window",
) -> list[str]:
    async with session_factory() as session:
        document_id = await _make_document(session)
        ids = []
        for i in range(count):
            item = ExtractionBatchItem(
                document_id=document_id,
                source_version=1,
                input_hash=f"hash-{document_id}-{i}",
                input_source_span_ids=["span_a"],
                request_input=request_input,
                request_schema={"type": "object", "additionalProperties": False},
                prompt_version=PROMPT_VERSION,
                schema_version=schema_version,
                status=ExtractionBatchItemStatus.PENDING,
            )
            session.add(item)
            ids.append(item.id)
        await session.commit()
    return ids


async def _items_by_status(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, int]:
    async with session_factory() as session:
        rows = (await session.execute(select(ExtractionBatchItem.status))).scalars().all()
    counts: dict[str, int] = {}
    for status in rows:
        counts[status.value] = counts.get(status.value, 0) + 1
    return counts


# --- happy path -------------------------------------------------------------


async def test_happy_path_claims_then_submits(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_pending_items(session_factory, count=3)

    # Asserts the claim is committed (items + batch SUBMITTING) BEFORE the
    # provider call — the core of the claim-before-call protocol.
    async def _on_submit(requests: list[Any], key: str | None) -> None:
        async with session_factory() as session:
            statuses = (
                await session.execute(select(ExtractionBatchItem.status))
            ).scalars().all()
            assert all(s is ExtractionBatchItemStatus.SUBMITTING for s in statuses)
            batch = (
                await session.execute(select(ExtractionBatch))
            ).scalars().one()
            assert batch.processing_status is ExtractionBatchStatus.SUBMITTING
            assert batch.provider_batch_id is None
        # idempotency key is the local batch id.
        assert key is not None

    fake = _FakeBatchProvider(on_submit=_on_submit)
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)

    ctx = {"settings": _settings(), "session_factory": session_factory}
    submitted = await submit_extraction_batches(ctx)

    assert submitted == 3
    assert len(fake.submit_calls) == 1
    requests, key = fake.submit_calls[0]
    assert len(requests) == 3
    assert key is not None  # == the ExtractionBatch.id
    counts = await _items_by_status(session_factory)
    assert counts == {"submitted": 3}
    async with session_factory() as session:
        batch = (await session.execute(select(ExtractionBatch))).scalars().one()
        assert batch.processing_status is ExtractionBatchStatus.SUBMITTED
        assert batch.provider_batch_id == "msgbatch_1"


async def test_provider_disabled_is_noop(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_pending_items(session_factory, count=2)

    def _boom(settings: Settings) -> Any:
        raise AssertionError("provider must not be built when disabled")

    monkeypatch.setattr(batch_module, "_build_batch_provider", _boom)
    ctx = {
        "settings": _settings(llm_provider="openai"),
        "session_factory": session_factory,
    }
    submitted = await submit_extraction_batches(ctx)
    assert submitted == 0
    counts = await _items_by_status(session_factory)
    assert counts == {"pending": 2}


async def test_idempotent_steady_state(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_pending_items(session_factory, count=2)
    fake = _FakeBatchProvider()
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {"settings": _settings(), "session_factory": session_factory}

    assert await submit_extraction_batches(ctx) == 2
    # Second run: nothing PENDING, so no submit.
    assert await submit_extraction_batches(ctx) == 0
    assert len(fake.submit_calls) == 1


async def test_schema_drift_item_skipped(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_pending_items(session_factory, count=1, schema_version="recipe.v0")
    fake = _FakeBatchProvider()
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {"settings": _settings(), "session_factory": session_factory}

    submitted = await submit_extraction_batches(ctx)
    assert submitted == 0
    assert len(fake.submit_calls) == 0
    # The drifted item stays PENDING (left for the operator / sweep).
    counts = await _items_by_status(session_factory)
    assert counts == {"pending": 1}


async def test_byte_cap_splits_into_multiple_batches(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two items each ~1KB; a 1500-byte cap forces one item per batch.
    await _seed_pending_items(session_factory, count=2, request_input="x" * 1000)
    fake = _FakeBatchProvider()
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {
        "settings": _settings(anthropic_batch_max_bytes=1500),
        "session_factory": session_factory,
    }

    submitted = await submit_extraction_batches(ctx)
    assert submitted == 2
    # Two separate submit calls, each with exactly one request.
    assert len(fake.submit_calls) == 2
    assert all(len(reqs) == 1 for reqs, _ in fake.submit_calls)


async def test_clean_failure_reverts_to_pending(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_pending_items(session_factory, count=2)
    fake = _FakeBatchProvider(fail=True)
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {"settings": _settings(), "session_factory": session_factory}

    submitted = await submit_extraction_batches(ctx)
    assert submitted == 0
    # Items reverted to PENDING; the batch row is FAILED.
    counts = await _items_by_status(session_factory)
    assert counts == {"pending": 2}
    async with session_factory() as session:
        batch = (await session.execute(select(ExtractionBatch))).scalars().one()
        assert batch.processing_status is ExtractionBatchStatus.FAILED
        # The reverted items are unlinked, eligible for the next tick.
        unlinked = (
            await session.execute(
                select(ExtractionBatchItem).where(
                    ExtractionBatchItem.batch_id.is_(None)
                )
            )
        ).scalars().all()
        assert len(list(unlinked)) == 2


# --- reconciliation ---------------------------------------------------------


async def _seed_stale_submitting_batch(
    session_factory: async_sessionmaker[AsyncSession], *, count: int
) -> str:
    """Create a SUBMITTING batch (created long ago) + SUBMITTING items, committed."""
    from datetime import UTC, datetime, timedelta

    old = datetime.now(tz=UTC) - timedelta(hours=3)
    async with session_factory() as session:
        document_id = await _make_document(session)
        batch = ExtractionBatch(
            provider="anthropic",
            provider_batch_id=None,
            model="claude-sonnet-4-6",
            processing_status=ExtractionBatchStatus.SUBMITTING,
            request_count=count,
            created_at=old,
        )
        session.add(batch)
        await session.flush()
        for i in range(count):
            session.add(
                ExtractionBatchItem(
                    document_id=document_id,
                    source_version=1,
                    input_hash=f"stale-{document_id}-{i}",
                    input_source_span_ids=["span_a"],
                    request_input="window",
                    request_schema={"type": "object"},
                    prompt_version=PROMPT_VERSION,
                    schema_version=SCHEMA_VERSION,
                    status=ExtractionBatchItemStatus.SUBMITTING,
                    batch_id=batch.id,
                )
            )
        await session.commit()
        return batch.id


async def test_reconcile_reverts_stale_submitting_batch_then_resubmits(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stale SUBMITTING batch (prior crash) is always reverted to PENDING — we do
    # NOT confirm via a count-only list-match (that could false-link to the wrong
    # provider batch and strand items SUBMITTED forever; review #2.1). The reverted
    # items are then re-claimed and re-submitted in the same run (dedup-safe in
    # 19.3 on input_hash), so the original batch ends FAILED and a NEW batch
    # carries the items to SUBMITTED.
    batch_id = await _seed_stale_submitting_batch(session_factory, count=2)
    fake = _FakeBatchProvider()
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {"settings": _settings(), "session_factory": session_factory}

    await submit_extraction_batches(ctx)

    async with session_factory() as session:
        original = await session.get(ExtractionBatch, batch_id)
        assert original is not None
        assert original.processing_status is ExtractionBatchStatus.FAILED
    # Items re-submitted via a fresh batch.
    counts = await _items_by_status(session_factory)
    assert counts == {"submitted": 2}
    assert len(fake.submit_calls) == 1


async def test_reconcile_does_not_touch_fresh_submitting_batch(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A SUBMITTING batch within the timeout (created just now) is a healthy in-flight
    # submit, not a crash — reconcile must leave it alone.
    async with session_factory() as session:
        document_id = await _make_document(session)
        batch = ExtractionBatch(
            provider="anthropic",
            provider_batch_id="msgbatch_live",
            model="claude-sonnet-4-6",
            processing_status=ExtractionBatchStatus.SUBMITTING,
            request_count=1,
        )
        session.add(batch)
        await session.flush()
        session.add(
            ExtractionBatchItem(
                document_id=document_id,
                source_version=1,
                input_hash=f"fresh-{document_id}",
                input_source_span_ids=["span_a"],
                request_input="w",
                request_schema={"type": "object"},
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                status=ExtractionBatchItemStatus.SUBMITTING,
                batch_id=batch.id,
            )
        )
        await session.commit()
        fresh_id = batch.id

    fake = _FakeBatchProvider()
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {"settings": _settings(), "session_factory": session_factory}
    await submit_extraction_batches(ctx)

    async with session_factory() as session:
        batch = await session.get(ExtractionBatch, fresh_id)
        assert batch is not None
        assert batch.processing_status is ExtractionBatchStatus.SUBMITTING
    assert len(fake.submit_calls) == 0


# --- concurrency ------------------------------------------------------------


async def test_concurrent_runs_claim_disjoint_sets(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_pending_items(session_factory, count=6)
    fake = _FakeBatchProvider()
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda settings: fake)
    ctx = {"settings": _settings(), "session_factory": session_factory}

    results = await asyncio.gather(
        submit_extraction_batches(ctx),
        submit_extraction_batches(ctx),
    )
    # Together they submit every item exactly once — no double submission.
    assert sum(results) == 6
    submitted_custom_ids = [
        req.custom_id for reqs, _ in fake.submit_calls for req in reqs
    ]
    assert len(submitted_custom_ids) == 6
    assert len(set(submitted_custom_ids)) == 6
    counts = await _items_by_status(session_factory)
    assert counts == {"submitted": 6}
