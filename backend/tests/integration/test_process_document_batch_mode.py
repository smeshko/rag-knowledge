"""Integration tests for process_document(batch_mode=True) — window registration.

Seeds a committed Document at EXTRACTING_ITEMS with SourceSpans (the "resume"
entry, so no PDF/PyMuPdf needed) and drives ``process_document(batch_mode=True)``
directly. Asserts windows are registered as PENDING ExtractionBatchItems carrying
the rendered input + sanitized schema, the doc stays in EXTRACTING_ITEMS, the LLM
is never called, and no ExtractionRun / KnowledgeItem rows are written. The full
QUEUED→fresh→register flow is covered by the endpoint test (TASK-004).
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from rag_recipes.config import get_settings
from rag_recipes.ingestion.jobs import process_document
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.ingestion.pipeline.windows import build_windows, compute_input_hash
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionBatchItemStatus,
    ExtractionRunStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.session import build_session_factory

pytestmark = pytest.mark.asyncio

SOURCE_VERSION = 1


class _SpyLLMProvider:
    """LLM provider that records calls; the batch path must never call it."""

    provider = "spy"
    default_model = "spy-model"

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured_output(self, request: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("generate_structured_output must not run in batch mode")


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


async def _seed_document_with_spans(
    session: AsyncSession, *, pages: int = 4
) -> tuple[str, list[SourceSpan]]:
    repo = DocumentRepository(session)
    pdf_bytes = b"%PDF-1.4 batch-mode fixture"
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
        status=DocumentStatus.EXTRACTING_ITEMS,
    )
    spans = [
        _make_span(document.id, page, f"Page {page} recipe text, serves {page}.")
        for page in range(1, pages + 1)
    ]
    session.add_all(spans)
    await session.flush()
    return document.id, spans


async def _cleanup(session_factory: Any, document_id: str) -> None:
    async with session_factory() as session:
        await session.execute(
            delete(ExtractionBatchItem).where(
                ExtractionBatchItem.document_id == document_id
            )
        )
        await session.execute(
            delete(ExtractionRun).where(ExtractionRun.document_id == document_id)
        )
        await session.execute(
            delete(SourceSpan).where(SourceSpan.document_id == document_id)
        )
        doc = await session.get(Document, document_id)
        asset_id = doc.asset_id if doc is not None else None
        if doc is not None:
            await session.delete(doc)
        await session.commit()
    if asset_id is not None:
        async with session_factory() as session:
            from rag_recipes.storage.models.source_asset import SourceAsset

            asset = await session.get(SourceAsset, asset_id)
            if asset is not None:
                await session.delete(asset)
            await session.commit()


def _ctx(session_factory: Any, spy: _SpyLLMProvider) -> dict[str, Any]:
    return {
        "settings": get_settings(),
        "session_factory": session_factory,
        "llm_provider": spy,
    }


async def test_batch_mode_registers_pending_items(test_engine: AsyncEngine) -> None:
    session_factory = build_session_factory(test_engine)
    settings = get_settings()
    async with session_factory() as session:
        document_id, spans = await _seed_document_with_spans(session)
        await session.commit()

    expected_windows = build_windows(
        spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages
    )
    spy = _SpyLLMProvider()
    try:
        registered = await process_document(
            _ctx(session_factory, spy), document_id, batch_mode=True
        )
        assert registered == len(expected_windows)
        assert spy.calls == 0

        async with session_factory() as session:
            items = list(
                (
                    await session.execute(
                        select(ExtractionBatchItem).where(
                            ExtractionBatchItem.document_id == document_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(items) == len(expected_windows)
            assert all(i.status is ExtractionBatchItemStatus.PENDING for i in items)
            assert all(i.batch_id is None for i in items)
            assert all(i.request_input for i in items)
            assert all(i.request_schema for i in items)
            assert all(i.schema_version == SCHEMA_VERSION for i in items)
            # Stored input_hashes match what a synchronous run would compute.
            expected_hashes = {
                compute_input_hash(w, PROMPT_VERSION, SCHEMA_VERSION)
                for w in expected_windows
            }
            assert {i.input_hash for i in items} == expected_hashes

            # No synchronous artefacts.
            runs = (
                await session.execute(
                    select(ExtractionRun).where(
                        ExtractionRun.document_id == document_id
                    )
                )
            ).scalars().all()
            assert list(runs) == []
            items_ki = (
                await session.execute(
                    select(KnowledgeItem).where(
                        KnowledgeItem.document_id == document_id
                    )
                )
            ).scalars().all()
            assert list(items_ki) == []

            # Doc stays parked in EXTRACTING_ITEMS.
            doc = await session.get(Document, document_id)
            assert doc is not None
            assert doc.status is DocumentStatus.EXTRACTING_ITEMS
    finally:
        await _cleanup(session_factory, document_id)


async def test_batch_mode_is_idempotent_on_replay(test_engine: AsyncEngine) -> None:
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        document_id, _ = await _seed_document_with_spans(session)
        await session.commit()

    spy = _SpyLLMProvider()
    try:
        first = await process_document(
            _ctx(session_factory, spy), document_id, batch_mode=True
        )
        assert first > 0
        # Re-running registers nothing new (existing non-terminal items skipped).
        second = await process_document(
            _ctx(session_factory, spy), document_id, batch_mode=True
        )
        assert second == 0
        async with session_factory() as session:
            count = (
                await session.execute(
                    select(ExtractionBatchItem).where(
                        ExtractionBatchItem.document_id == document_id
                    )
                )
            ).scalars().all()
            assert len(list(count)) == first
    finally:
        await _cleanup(session_factory, document_id)


async def test_batch_mode_skips_window_with_existing_extraction_run(
    test_engine: AsyncEngine,
) -> None:
    session_factory = build_session_factory(test_engine)
    settings = get_settings()
    async with session_factory() as session:
        document_id, spans = await _seed_document_with_spans(session)
        await session.commit()

    windows = build_windows(
        spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages
    )
    # Pre-create a terminal ExtractionRun for the first window's hash.
    first_hash = compute_input_hash(windows[0], PROMPT_VERSION, SCHEMA_VERSION)
    async with session_factory() as session:
        session.add(
            ExtractionRun(
                document_id=document_id,
                source_version=SOURCE_VERSION,
                provider="openai",
                model="gpt-4.1",
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                input_source_span_ids=windows[0].span_ids,
                input_hash=first_hash,
                status=ExtractionRunStatus.SUCCESS,
                output_json={"items": []},
            )
        )
        await session.commit()

    spy = _SpyLLMProvider()
    try:
        registered = await process_document(
            _ctx(session_factory, spy), document_id, batch_mode=True
        )
        assert registered == len(windows) - 1
        async with session_factory() as session:
            hashes = set(
                (
                    await session.execute(
                        select(ExtractionBatchItem.input_hash).where(
                            ExtractionBatchItem.document_id == document_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert first_hash not in hashes
    finally:
        await _cleanup(session_factory, document_id)


async def test_batch_mode_concurrent_registration_no_duplicate_items(
    test_engine: AsyncEngine,
) -> None:
    # Two registration passes racing on the same document: the partial-unique
    # index forces the losing insert to IntegrityError, which the branch swallows
    # as a skip. The invariant — exactly one item per window, no raised error —
    # holds whether a window was skipped via the read or via the IntegrityError.
    session_factory = build_session_factory(test_engine)
    settings = get_settings()
    async with session_factory() as session:
        document_id, spans = await _seed_document_with_spans(session)
        await session.commit()

    expected = build_windows(
        spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages
    )
    spy_a, spy_b = _SpyLLMProvider(), _SpyLLMProvider()
    try:
        # Neither call raises despite racing on the unique index.
        await asyncio.gather(
            process_document(_ctx(session_factory, spy_a), document_id, batch_mode=True),
            process_document(_ctx(session_factory, spy_b), document_id, batch_mode=True),
        )
        async with session_factory() as session:
            hashes = list(
                (
                    await session.execute(
                        select(ExtractionBatchItem.input_hash).where(
                            ExtractionBatchItem.document_id == document_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            # Exactly one item per window — no duplicates from the race.
            assert len(hashes) == len(expected)
            assert len(set(hashes)) == len(expected)
    finally:
        await _cleanup(session_factory, document_id)
