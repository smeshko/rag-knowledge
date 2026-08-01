"""A partial resume must not retire the windows it never re-extracted.

Regression cover for the bug that emptied a live cookbook: an ingest whose first
pass completed most windows (items promoted to ``READY``) but left a few without
an ``ExtractionRun``. Re-driving ``process_document`` enters as ``resume``, and
``_run_extraction_batches``' ``done_hashes`` skip set correctly re-extracts only
the missing windows — but ``_finalize_extraction`` then ran its keep-set
supersede over a candidate set holding *only* those few windows' items, so every
recipe the earlier pass had already promoted flipped to ``SUPERSEDED``. A 108-item
book was left with the 12 recipes of its last 6 windows.

The fix carries the untouched windows' promoted items into the dedup, so they
compete (and normally survive) instead of being retired by omission. These tests
pin both halves of that: a resume preserves what it did not revisit, and the
carried items still lose to a better-scoring duplicate when one appears.
``test_process_document_reuse.py`` pins the opposite case — a *full* same-version
re-extraction still replaces the prior pass wholesale.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag_recipes.config import get_settings
from rag_recipes.ingestion.jobs import process_document
from rag_recipes.ingestion.pipeline.chunking import persist_chunks_for_ready_items
from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    RecipeExtractionOutput,
    _render_prompt,
    build_recipe_v1_json_schema,
)
from rag_recipes.ingestion.pipeline.persist import persist_knowledge_item
from rag_recipes.ingestion.pipeline.windows import (
    Window,
    build_windows,
    compute_input_hash,
    format_window_for_llm,
)
from rag_recipes.ingestion.status import transition_to
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.ingestion_failure import IngestionFailure
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.session import build_session_factory
from tests.unit.ingestion.test_validation import (
    _make_recipe,
    _make_step,
    _make_structured_data,
    _recipe_confidence,
)

pytestmark = pytest.mark.asyncio

# Five pages at window size 3 / overlap 1 give exactly two windows: pages 1-3 and
# 3-5. Page 2 carries the recipe an earlier pass already promoted (it appears only
# in the first window); page 5 carries the one the resume still has to find.
_WINDOW_SIZE = 3
_OVERLAP = 1
_DONE_SPAN = "span_p2"
_PENDING_SPAN = "span_p5"


async def _noop_extract(*args: object, **kwargs: object) -> int:
    """Spans are pre-seeded; the resume path must not re-run the PDF stage anyway."""
    return 5


def _build_spans(document_id: str) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    for page in range(1, 6):
        if page == 2:
            text = "Tomato Soup. Ingredients: tomatoes. Method: simmer."
        elif page == 5:
            text = "Carrot Cake. Ingredients: carrots. Method: bake."
        else:
            text = f"Filler text for page {page}."
        locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
        spans.append(
            SourceSpan(
                id=f"span_p{page}",
                document_id=document_id,
                source_version=1,
                source_type=SourceType.PDF,
                locator=locator,
                locator_hash=hashlib.sha256(str(locator).encode()).hexdigest(),
                text=text,
                text_hash=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    return spans


def _request_hash_for(window: Window) -> str:
    """Replicate run_extraction's request so the fake can be keyed per window."""
    request = StructuredOutputRequest(
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input=_render_prompt(format_window_for_llm(window)),
        json_schema=build_recipe_v1_json_schema(),
    )
    return FakeLLMProvider.request_hash(request)


def _recipe_output(*, title: str, cited_span_id: str, overall: float) -> dict[str, object]:
    recipe = _make_recipe(
        title=title,
        source_span_ids=[cited_span_id],
        structured_data=_make_structured_data(steps=[_make_step(source_span_ids=[cited_span_id])]),
        confidence=_recipe_confidence(overall=overall, boundary=overall),
    )
    return RecipeExtractionOutput(items=[recipe]).model_dump(mode="json", by_alias=True)


async def _seed_document(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[str, str]:
    async with session_factory() as session:
        repo = DocumentRepository(session)
        pdf_bytes = b"%PDF-1.4 partial-resume fixture"
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
            author="Alice",
            source_type=SourceType.PDF,
            language=None,
            active_source_version=None,
            status=DocumentStatus.QUEUED,
        )
        await session.commit()
        return document.id, asset.id


async def _seed_finished_window(
    session_factory: async_sessionmaker[AsyncSession],
    document_id: str,
    window: Window,
    *,
    title: str,
    overall: float,
) -> str:
    """Land one window exactly as a completed earlier pass leaves it.

    A committed ``ExtractionRun`` carrying the window's real ``input_hash`` (so the
    resume's skip set short-circuits it), the promoted ``READY`` item that pass's
    finalize produced, and that item's chunks. Returns the item id.
    """
    async with session_factory() as session:
        run = ExtractionRun(
            id=new_id("run"),
            document_id=document_id,
            source_version=1,
            provider="fake",
            model="fake-model",
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            input_source_span_ids=list(window.span_ids),
            input_hash=compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION),
            status=ExtractionRunStatus.SUCCESS,
            output_json=_recipe_output(title=title, cited_span_id=_DONE_SPAN, overall=overall),
        )
        session.add(run)
        await session.flush()
        parsed = RecipeExtractionOutput.model_validate(run.output_json)
        item = await persist_knowledge_item(
            session,
            parsed.items[0],
            extraction_run_id=run.id,
            document_id=document_id,
            source_version=1,
            window=window,
        )
        item_id = item.id
        await persist_chunks_for_ready_items(
            session, document_id=document_id, source_version=1, category="recipes"
        )
        await transition_to(session, document_id, DocumentStatus.EXTRACTING_ITEMS)
        await session.commit()
        return item_id


def _ctx(
    session_factory: async_sessionmaker[AsyncSession],
    provider: FakeLLMProvider,
    tmp_path: Path,
) -> dict[str, Any]:
    settings = get_settings().model_copy(
        update={
            "pdf_window_size_pages": _WINDOW_SIZE,
            "pdf_overlap_pages": _OVERLAP,
            "local_storage_root": str(tmp_path),
        }
    )
    return {
        "settings": settings,
        "session_factory": session_factory,
        "observability": None,
        "llm_provider": provider,
        "embedding_provider": FakeEmbeddingProvider(),
    }


async def _cleanup(
    session_factory: async_sessionmaker[AsyncSession], document_id: str, asset_id: str
) -> None:
    async with session_factory() as session:
        await session.execute(
            delete(ChunkEmbedding).where(
                ChunkEmbedding.chunk_id.in_(
                    select(Chunk.id).where(Chunk.document_id == document_id)
                )
            )
        )
        for model in (Chunk, KnowledgeItem, ExtractionRun, SourceSpan, IngestionFailure):
            await session.execute(delete(model).where(model.document_id == document_id))
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.execute(delete(SourceAsset).where(SourceAsset.id == asset_id))
        await session.commit()


async def _statuses(
    session_factory: async_sessionmaker[AsyncSession], document_id: str
) -> dict[str, KnowledgeItemStatus]:
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(KnowledgeItem.id, KnowledgeItem.status).where(
                    KnowledgeItem.document_id == document_id
                )
            )
        ).all()
        return {item_id: status for item_id, status in rows}


async def _chunk_count(session_factory: async_sessionmaker[AsyncSession], item_id: str) -> int:
    async with session_factory() as session:
        rows = (await session.execute(select(Chunk.id).where(Chunk.parent_id == item_id))).all()
        return len(rows)


async def _run_resume(
    session_factory: async_sessionmaker[AsyncSession],
    document_id: str,
    pending_window: Window,
    tmp_path: Path,
    *,
    title: str,
    overall: float,
) -> int:
    """Re-drive process_document; only ``pending_window`` has no committed run."""
    provider = FakeLLMProvider(
        responses_by_hash={
            _request_hash_for(pending_window): _recipe_output(
                title=title, cited_span_id=_PENDING_SPAN, overall=overall
            )
        }
    )
    return await process_document(_ctx(session_factory, provider, tmp_path), document_id)


async def test_resume_preserves_items_from_windows_it_did_not_reextract(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The already-promoted recipe survives a resume that only covers the rest."""
    monkeypatch.setattr("rag_recipes.ingestion.jobs.extract_and_persist_spans", _noop_extract)
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        spans = _build_spans(document_id)
        async with session_factory() as session:
            session.add_all(spans)
            await session.commit()
        done_window, pending_window = build_windows(spans, _WINDOW_SIZE, _OVERLAP)

        done_item_id = await _seed_finished_window(
            session_factory, document_id, done_window, title="Tomato Soup", overall=0.9
        )
        chunks_before = await _chunk_count(session_factory, done_item_id)
        assert chunks_before > 0

        await _run_resume(
            session_factory,
            document_id,
            pending_window,
            tmp_path,
            title="Carrot Cake",
            overall=0.8,
        )

        statuses = await _statuses(session_factory, document_id)
        # The bug flipped this to SUPERSEDED: its window was never revisited, so the
        # resume held no replacement for it.
        assert statuses[done_item_id] is KnowledgeItemStatus.READY
        # ...and the resume's own find landed alongside it, not instead of it.
        assert sorted(statuses.values(), key=lambda s: s.value) == [
            KnowledgeItemStatus.READY,
            KnowledgeItemStatus.READY,
        ]
        # The carried item keeps exactly its original chunks — finalize re-runs the
        # chunk builder over a READY set that now includes it.
        assert await _chunk_count(session_factory, done_item_id) == chunks_before

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            assert doc.status is DocumentStatus.READY
            assert doc.active_source_version == 1
    finally:
        await _cleanup(session_factory, document_id, asset_id)


async def test_resume_carried_item_still_loses_to_a_better_duplicate(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Carrying prior items into the dedup does not disable the dedup itself."""
    monkeypatch.setattr("rag_recipes.ingestion.jobs.extract_and_persist_spans", _noop_extract)
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        spans = _build_spans(document_id)
        async with session_factory() as session:
            session.add_all(spans)
            await session.commit()
        done_window, pending_window = build_windows(spans, _WINDOW_SIZE, _OVERLAP)

        # Same recipe, weaker extraction, from the window the resume will not touch.
        done_item_id = await _seed_finished_window(
            session_factory, document_id, done_window, title="Carrot Cake", overall=0.5
        )

        await _run_resume(
            session_factory,
            document_id,
            pending_window,
            tmp_path,
            title="Carrot Cake",
            overall=0.95,
        )

        statuses = await _statuses(session_factory, document_id)
        assert statuses[done_item_id] is KnowledgeItemStatus.SUPERSEDED
        winners = [
            item_id for item_id, status in statuses.items() if status is KnowledgeItemStatus.READY
        ]
        assert len(winners) == 1
        assert winners[0] != done_item_id
    finally:
        await _cleanup(session_factory, document_id, asset_id)
