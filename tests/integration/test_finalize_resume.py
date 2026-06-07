"""Integration tests for finalize-resume (Epic 19.3, TASK-005).

Drives the full batch tail end-to-end against a real DB: seed a doc parked in
EXTRACTING_ITEMS with a SUBMITTED batch whose items carry the *real* window
``input_hash`` (so the resume path's ``done_hashes`` skips them), poll a fake
provider, ingest, then re-drive ``process_document`` (resume) and assert the doc
reaches READY/NEEDS_REVIEW via the existing finalize machinery — no finalize
reimplementation. Completion is per-document (windows may span batches).
"""

from __future__ import annotations

import hashlib
import json
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
from rag_recipes.ingestion.jobs import process_document
from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    RecipeExtractionOutput,
)
from rag_recipes.ingestion.pipeline.windows import build_windows, compute_input_hash
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.llm.anthropic_batch import BatchResult
from rag_recipes.providers.llm.fake import FakeLLMProvider
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
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_batch import ExtractionBatch
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.session import build_session_factory
from tests.unit.ingestion.test_validation import _make_recipe, _make_step, _make_structured_data

pytestmark = pytest.mark.asyncio

_TEST_CATEGORY = "batch-finalize-test"


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


class _FakePollProvider:
    def __init__(self, results_by_batch: dict[str, list[BatchResult]]) -> None:
        self._results = results_by_batch

    async def retrieve_batch(self, provider_batch_id: str) -> Any:
        from rag_recipes.providers.llm.anthropic_batch import BatchStatus

        # Only batches this provider knows results for are "ended"; others are
        # still in flight (so a split-across-batches doc isn't ended prematurely).
        status = "ended" if provider_batch_id in self._results else "in_progress"
        return BatchStatus(processing_status=status)

    async def iter_results(self, provider_batch_id: str) -> Any:
        for result in self._results.get(provider_batch_id, []):
            yield result


@pytest_asyncio.fixture
async def session_factory(test_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return build_session_factory(test_engine)


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(session_factory: async_sessionmaker[AsyncSession]) -> Any:
    yield
    async with session_factory() as session:
        doc_ids = list(
            (await session.execute(select(Document.id).where(Document.category == _TEST_CATEGORY)))
            .scalars()
            .all()
        )
        await session.execute(delete(ExtractionBatchItem))
        await session.execute(delete(ExtractionBatch))
        if doc_ids:
            for table in (ChunkEmbedding,):
                await session.execute(
                    delete(table).where(
                        table.chunk_id.in_(select(Chunk.id).where(Chunk.document_id.in_(doc_ids)))
                    )
                )
            await session.execute(delete(Chunk).where(Chunk.document_id.in_(doc_ids)))
            await session.execute(
                delete(KnowledgeItem).where(KnowledgeItem.document_id.in_(doc_ids))
            )
            await session.execute(
                delete(ExtractionRun).where(ExtractionRun.document_id.in_(doc_ids))
            )
            await session.execute(delete(SourceSpan).where(SourceSpan.document_id.in_(doc_ids)))
            asset_ids = list(
                (await session.execute(select(Document.asset_id).where(Document.id.in_(doc_ids))))
                .scalars()
                .all()
            )
            await session.execute(delete(Document).where(Document.id.in_(doc_ids)))
            await session.execute(delete(SourceAsset).where(SourceAsset.id.in_(asset_ids)))
        await session.commit()


def _settings() -> Settings:
    return get_settings().model_copy(
        update={
            "llm_provider": "anthropic",
            "anthropic_api_key": "sk-ant-test",
            "anthropic_llm_model": "claude-sonnet-4-6",
        }
    )


def _ctx(session_factory: async_sessionmaker[AsyncSession]) -> dict[str, Any]:
    return {"settings": _settings(), "session_factory": session_factory, "redis": AsyncMock()}


def _recipe_payload(span_ids: list[str]) -> str:
    span = span_ids[0]
    recipe = _make_recipe(
        source_span_ids=[span],
        structured_data=_make_structured_data(steps=[_make_step(source_span_ids=[span])]),
    )
    return json.dumps(RecipeExtractionOutput(items=[recipe]).model_dump(by_alias=True, mode="json"))


async def _make_doc_and_spans(session: AsyncSession, *, pages: int) -> tuple[str, list[SourceSpan]]:
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
    spans = []
    for page in range(1, pages + 1):
        locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
        text = f"Page {page}: Tomato soup. Heat the oil in a large pot. Serves 4. " * 8
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
    return document.id, spans


async def _register_batch(
    session: AsyncSession,
    document_id: str,
    spans: list[SourceSpan],
    settings: Settings,
    *,
    provider_batch_id: str,
) -> tuple[str, list[ExtractionBatchItem]]:
    """Create a SUBMITTED batch + items carrying the real per-window input_hash."""
    windows = build_windows(spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages)
    batch = ExtractionBatch(
        provider="anthropic",
        provider_batch_id=provider_batch_id,
        model="claude-sonnet-4-6",
        processing_status=ExtractionBatchStatus.SUBMITTED,
        request_count=len(windows),
    )
    session.add(batch)
    await session.flush()
    items = []
    for window in windows:
        item = ExtractionBatchItem(
            document_id=document_id,
            source_version=1,
            input_hash=compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION),
            input_source_span_ids=window.span_ids,
            request_input="window",
            request_schema={"type": "object"},
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            status=ExtractionBatchItemStatus.SUBMITTED,
            batch_id=batch.id,
        )
        session.add(item)
        items.append(item)
    await session.flush()
    return batch.id, items


async def _doc_status(
    session_factory: async_sessionmaker[AsyncSession], doc_id: str
) -> DocumentStatus:
    async with session_factory() as session:
        doc = await session.get(Document, doc_id)
        assert doc is not None
        return doc.status


async def _resume(session_factory: async_sessionmaker[AsyncSession], doc_id: str) -> None:
    ctx = {
        "settings": _settings(),
        "session_factory": session_factory,
        "llm_provider": FakeLLMProvider(),  # never called on resume (windows skipped)
        "embedding_provider": FakeEmbeddingProvider(),
    }
    await process_document(ctx, doc_id)


async def test_happy_path_end_to_end_reaches_ready(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    async with session_factory() as session:
        doc_id, spans = await _make_doc_and_spans(session, pages=2)
        batch_id, items = await _register_batch(
            session, doc_id, spans, settings, provider_batch_id="mb_1"
        )
        results = [
            BatchResult(
                custom_id=item.id,
                result_type="succeeded",
                message=_Message(content=[_Block(_recipe_payload(item.input_source_span_ids))]),
            )
            for item in items
        ]
        await session.commit()

    fake = _FakePollProvider({"mb_1": results})
    monkeypatch.setattr(batch_module, "_build_batch_provider", lambda s: fake)

    await poll_extraction_batches(_ctx(session_factory))
    # All items terminal → resume → finalize → chunk → embed → index → READY.
    await _resume(session_factory, doc_id)

    assert await _doc_status(session_factory, doc_id) is DocumentStatus.READY
    async with session_factory() as session:
        ready = list(
            (
                await session.execute(
                    select(KnowledgeItem).where(
                        KnowledgeItem.document_id == doc_id,
                        KnowledgeItem.status == KnowledgeItemStatus.READY,
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(ready) >= 1  # recipes are queryable


async def test_completion_detection_split_across_batches(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A doc whose windows are registered across two batches must be re-driven only
    # after the SECOND batch ends (not the first), or finalize drops windows.
    settings = _settings()
    async with session_factory() as session:
        doc_id, spans = await _make_doc_and_spans(session, pages=4)
        windows = build_windows(spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages)
        assert len(windows) >= 2
        # Split: window 0 in batch A, the rest in batch B.
        batch_a = ExtractionBatch(
            provider="anthropic",
            provider_batch_id="mb_a",
            model="m",
            processing_status=ExtractionBatchStatus.SUBMITTED,
            request_count=1,
        )
        batch_b = ExtractionBatch(
            provider="anthropic",
            provider_batch_id="mb_b",
            model="m",
            processing_status=ExtractionBatchStatus.SUBMITTED,
            request_count=len(windows) - 1,
        )
        session.add_all([batch_a, batch_b])
        await session.flush()
        items = []
        for i, window in enumerate(windows):
            item = ExtractionBatchItem(
                document_id=doc_id,
                source_version=1,
                input_hash=compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION),
                input_source_span_ids=window.span_ids,
                request_input="w",
                request_schema={"type": "object"},
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                status=ExtractionBatchItemStatus.SUBMITTED,
                batch_id=batch_a.id if i == 0 else batch_b.id,
            )
            session.add(item)
            items.append(item)
        await session.flush()
        results_a = [
            BatchResult(
                custom_id=items[0].id,
                result_type="succeeded",
                message=_Message(content=[_Block(_recipe_payload(items[0].input_source_span_ids))]),
            )
        ]
        results_b = [
            BatchResult(
                custom_id=it.id,
                result_type="succeeded",
                message=_Message(content=[_Block(_recipe_payload(it.input_source_span_ids))]),
            )
            for it in items[1:]
        ]
        await session.commit()

    redis_a = AsyncMock()
    # Only mb_a is ended on the first poll; mb_b is still in flight.
    monkeypatch.setattr(
        batch_module,
        "_build_batch_provider",
        lambda s: _FakePollProvider({"mb_a": results_a}),
    )
    await poll_extraction_batches(
        {"settings": settings, "session_factory": session_factory, "redis": redis_a}
    )
    # Only batch A ended; batch B's items still SUBMITTED → NOT complete → no re-drive.
    redis_a.enqueue_job.assert_not_awaited()

    redis_b = AsyncMock()
    monkeypatch.setattr(
        batch_module, "_build_batch_provider", lambda s: _FakePollProvider({"mb_b": results_b})
    )
    await poll_extraction_batches(
        {"settings": settings, "session_factory": session_factory, "redis": redis_b}
    )
    # Now all items terminal → re-drive.
    redis_b.enqueue_job.assert_awaited_once()


async def test_partial_batch_failure_healthy_ready_failed_needs_review(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    async with session_factory() as session:
        good_id, good_spans = await _make_doc_and_spans(session, pages=2)
        bad_id, bad_spans = await _make_doc_and_spans(session, pages=2)
        _, good_items = await _register_batch(
            session, good_id, good_spans, settings, provider_batch_id="mb_mix"
        )
        # Reuse the same batch for the bad doc's items.
        bad_windows = build_windows(
            bad_spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages
        )
        batch = (
            await session.execute(
                select(ExtractionBatch).where(ExtractionBatch.provider_batch_id == "mb_mix")
            )
        ).scalar_one()
        bad_items = []
        for window in bad_windows:
            item = ExtractionBatchItem(
                document_id=bad_id,
                source_version=1,
                input_hash=compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION),
                input_source_span_ids=window.span_ids,
                request_input="w",
                request_schema={"type": "object"},
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                status=ExtractionBatchItemStatus.SUBMITTED,
                batch_id=batch.id,
            )
            session.add(item)
            bad_items.append(item)
        await session.flush()
        results = [
            BatchResult(
                custom_id=it.id,
                result_type="succeeded",
                message=_Message(content=[_Block(_recipe_payload(it.input_source_span_ids))]),
            )
            for it in good_items
        ] + [
            BatchResult(
                custom_id=it.id,
                result_type="errored",
                error_type="invalid_request_error",
                retryable=False,
            )
            for it in bad_items
        ]
        await session.commit()

    monkeypatch.setattr(
        batch_module, "_build_batch_provider", lambda s: _FakePollProvider({"mb_mix": results})
    )
    await poll_extraction_batches(_ctx(session_factory))
    await _resume(session_factory, good_id)
    await _resume(session_factory, bad_id)

    assert await _doc_status(session_factory, good_id) is DocumentStatus.READY
    # The all-failed doc resolves to NEEDS_REVIEW (zero ready items), with its
    # windows recorded as REJECTED runs (surfaced, not dropped).
    assert await _doc_status(session_factory, bad_id) is DocumentStatus.NEEDS_REVIEW
    async with session_factory() as session:
        bad_runs = (
            (
                await session.execute(
                    select(ExtractionRun.status).where(ExtractionRun.document_id == bad_id)
                )
            )
            .scalars()
            .all()
        )
        assert bad_runs and all(s is ExtractionRunStatus.REJECTED for s in bad_runs)


async def test_idempotent_replay_after_ready(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    async with session_factory() as session:
        doc_id, spans = await _make_doc_and_spans(session, pages=2)
        _, items = await _register_batch(
            session, doc_id, spans, settings, provider_batch_id="mb_idem"
        )
        results = [
            BatchResult(
                custom_id=it.id,
                result_type="succeeded",
                message=_Message(content=[_Block(_recipe_payload(it.input_source_span_ids))]),
            )
            for it in items
        ]
        await session.commit()

    monkeypatch.setattr(
        batch_module, "_build_batch_provider", lambda s: _FakePollProvider({"mb_idem": results})
    )
    await poll_extraction_batches(_ctx(session_factory))
    await _resume(session_factory, doc_id)
    assert await _doc_status(session_factory, doc_id) is DocumentStatus.READY

    async with session_factory() as session:
        runs_before = len(
            (
                await session.execute(
                    select(ExtractionRun.id).where(ExtractionRun.document_id == doc_id)
                )
            )
            .scalars()
            .all()
        )

    # Re-poll: batch is ENDED (not pollable) → nothing new; resume no-ops (READY).
    await poll_extraction_batches(_ctx(session_factory))
    await _resume(session_factory, doc_id)
    assert await _doc_status(session_factory, doc_id) is DocumentStatus.READY
    async with session_factory() as session:
        runs_after = len(
            (
                await session.execute(
                    select(ExtractionRun.id).where(ExtractionRun.document_id == doc_id)
                )
            )
            .scalars()
            .all()
        )
    assert runs_after == runs_before
