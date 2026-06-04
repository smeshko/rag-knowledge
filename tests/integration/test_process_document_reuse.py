"""End-to-end reuse-source-spans reprocess test (Epic 11.1).

Drives the *real* ``process_document`` twice against the Postgres ``test_engine``:
a fresh run that lands the document at ``ready`` (v1), then a
``reuse_source_spans=True`` re-run at the *same* version. The PDF layer is
bypassed exactly as in ``test_process_document_dedup.py`` (spans seeded directly,
``extract_and_persist_spans`` monkeypatched), so the reuse path's contract is
asserted end-to-end:

* the reuse run never re-extracts PDF text (``extract_and_persist_spans`` raises
  if called) and never writes new spans;
* it produces new ``KnowledgeItem``s at the same ``source_version``;
* the prior pass's items flip to ``SUPERSEDED`` while the new pass's survive;
* the prior pass's ``Chunk`` rows are retained;
* ``Document.active_source_version`` is unchanged.

**Changed-prompt is required** (PLAN Risks / DECISIONS #2): a same-version reuse
with an *unchanged* ``PROMPT_VERSION`` re-derives the same ``input_hash`` for every
window, so the skip-set (and ``run_extraction``'s cache) short-circuit the LLM
stage — no new runs, no new items, no supersede. Both ``PROMPT_VERSION`` bindings
(``jobs`` for the skip-set, ``extraction`` for ``input_hash`` + cache + request)
are monkeypatched on the reuse leg so the windows get fresh hashes and the
provider is actually called.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag_recipes.config import get_settings
from rag_recipes.ingestion.jobs import process_document
from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    RecipeExtractionOutput,
    _render_prompt,
    build_recipe_v1_json_schema,
)
from rag_recipes.ingestion.pipeline.windows import (
    Window,
    build_windows,
    format_window_for_llm,
)
from rag_recipes.ingestion.status import transition_to
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from rag_recipes.storage.enums import (
    DocumentStatus,
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

_WINDOW_SIZE = 3
_OVERLAP = 1
_REUSE_PROMPT_VERSION = "reuse-prompt-v2"


async def _noop_extract(*args: object, **kwargs: object) -> int:
    """Stand-in for extract_and_persist_spans: spans are pre-seeded, so do nothing."""
    return 5


async def _boom_extract(*args: object, **kwargs: object) -> int:
    """A reuse run must never re-extract PDF text — fail loudly if it does."""
    raise AssertionError("reuse_source_spans run must not call extract_and_persist_spans")


def _build_spans(document_id: str) -> list[SourceSpan]:
    """Five detached per-page spans (explicit ids) — page 3 carries the recipe."""
    spans: list[SourceSpan] = []
    for page in range(1, 6):
        text = (
            "Tomato Soup. Ingredients: tomatoes. Method: simmer."
            if page == 3
            else f"Filler text for page {page}."
        )
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


def _request_hash_for(window: Window, prompt_version: str) -> str:
    """Replicate run_extraction's request so the fake can be keyed per window.

    ``run_extraction`` builds the request from the rendered prompt, the provider
    labels (``fake``/``fake-model``) and the *current* ``PROMPT_VERSION``; the fake
    keys its canned responses on exactly this hash. Passing ``prompt_version``
    explicitly lets the reuse leg key its fake against the bumped version.
    """
    request = StructuredOutputRequest(
        provider="fake",
        model="fake-model",
        prompt_version=prompt_version,
        schema_version=SCHEMA_VERSION,
        input=_render_prompt(format_window_for_llm(window)),
        json_schema=build_recipe_v1_json_schema(),
    )
    return FakeLLMProvider.request_hash(request)


def _recipe_output(
    *, title: str, cited_span_id: str, overall: float
) -> dict[str, object]:
    """A one-recipe ``recipe.v1`` output for ``title`` citing ``cited_span_id``."""
    recipe = _make_recipe(
        title=title,
        source_span_ids=[cited_span_id],
        structured_data=_make_structured_data(
            steps=[_make_step(source_span_ids=[cited_span_id])]
        ),
        confidence=_recipe_confidence(overall=overall, boundary=overall),
    )
    return RecipeExtractionOutput(items=[recipe]).model_dump(mode="json", by_alias=True)


def _empty_output() -> dict[str, object]:
    """A zero-item ``recipe.v1`` output — the provider accepted nothing."""
    return RecipeExtractionOutput(items=[]).model_dump(mode="json", by_alias=True)


async def _seed_document(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[str, str]:
    """Insert a QUEUED Document + SourceAsset; return (document_id, asset_id)."""
    async with session_factory() as session:
        repo = DocumentRepository(session)
        pdf_bytes = b"%PDF-1.4 reuse fixture"
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
        await session.commit()
        return document.id, asset.id


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


async def _seed_spans_and_run_fresh(
    session_factory: async_sessionmaker[AsyncSession],
    document_id: str,
    tmp_path: Path,
) -> tuple[list[SourceSpan], int]:
    """Seed v1 spans and drive the fresh leg to READY; return (spans, fresh_count)."""
    spans = _build_spans(document_id)
    async with session_factory() as session:
        session.add_all(spans)
        await session.commit()
    window_low, window_high = build_windows(spans, _WINDOW_SIZE, _OVERLAP)
    fresh_provider = FakeLLMProvider(
        responses_by_hash={
            _request_hash_for(window_low, PROMPT_VERSION): _recipe_output(
                title="Tomato Soup", cited_span_id="span_p3", overall=0.6
            ),
            _request_hash_for(window_high, PROMPT_VERSION): _recipe_output(
                title="Tomato Soup", cited_span_id="span_p3", overall=0.9
            ),
        }
    )
    fresh_count = await process_document(
        _ctx(session_factory, fresh_provider, tmp_path), document_id
    )
    return spans, fresh_count


async def _requeue(
    session_factory: async_sessionmaker[AsyncSession], document_id: str
) -> None:
    """Transition a terminal-accepted (READY) doc back to QUEUED for a reuse run.

    Stands in for the reprocess endpoint's guarded READY -> QUEUED UPDATE; the
    reuse job only enters the reuse path from QUEUED (DECISIONS #2)."""
    async with session_factory() as session:
        await transition_to(session, document_id, DocumentStatus.QUEUED)
        await session.commit()


def _bump_prompt_version(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    """Bump both PROMPT_VERSION bindings so a same-version reuse re-extracts.

    ``jobs.PROMPT_VERSION`` keys the resume skip-set; ``extraction.PROMPT_VERSION``
    feeds ``compute_input_hash`` (the stored ExtractionRun.input_hash + the cache
    lookup) and the StructuredOutputRequest. Both must change or the reuse windows
    are skipped / cache-hit and the LLM stage no-ops."""
    monkeypatch.setattr(
        "rag_recipes.ingestion.pipeline.extraction.PROMPT_VERSION", version
    )
    monkeypatch.setattr("rag_recipes.ingestion.jobs.PROMPT_VERSION", version)


async def test_reuse_same_version_supersedes_prior_and_retains_chunks(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rag_recipes.ingestion.jobs.extract_and_persist_spans", _noop_extract
    )
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        spans, fresh_count = await _seed_spans_and_run_fresh(
            session_factory, document_id, tmp_path
        )
        assert fresh_count == 1
        window_low, window_high = build_windows(spans, _WINDOW_SIZE, _OVERLAP)

        # Snapshot the post-fresh state: one ready item + its chunks at v1.
        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            assert doc.status is DocumentStatus.READY
            active_before = doc.active_source_version
            assert active_before == 1
            old_items = list(
                (
                    await session.execute(
                        select(KnowledgeItem).where(
                            KnowledgeItem.document_id == document_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(old_items) == 1
            old_item_id = old_items[0].id
            assert old_items[0].normalized_title == "tomato soup"
            span_count_before = await session.scalar(
                select(func.count())
                .select_from(SourceSpan)
                .where(SourceSpan.document_id == document_id)
            )
            old_chunk_ids = set(
                (
                    await session.execute(
                        select(Chunk.id).where(Chunk.parent_id == old_item_id)
                    )
                )
                .scalars()
                .all()
            )
            assert len(old_chunk_ids) == 5

        await _requeue(session_factory, document_id)

        # Reuse leg: bumped prompt + a changed recipe + a guard that the PDF text
        # stage is never re-run.
        _bump_prompt_version(monkeypatch, _REUSE_PROMPT_VERSION)
        monkeypatch.setattr(
            "rag_recipes.ingestion.jobs.extract_and_persist_spans", _boom_extract
        )
        reuse_provider = FakeLLMProvider(
            responses_by_hash={
                _request_hash_for(window_low, _REUSE_PROMPT_VERSION): _recipe_output(
                    title="Carrot Soup", cited_span_id="span_p3", overall=0.6
                ),
                _request_hash_for(window_high, _REUSE_PROMPT_VERSION): _recipe_output(
                    title="Carrot Soup", cited_span_id="span_p3", overall=0.9
                ),
            }
        )
        reuse_count = await process_document(
            _ctx(session_factory, reuse_provider, tmp_path),
            document_id,
            source_version=1,
            reuse_source_spans=True,
        )
        # Real work, not a _resume_or_fresh skip-no-op.
        assert reuse_count == 1

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            # Active version is untouched by reuse.
            assert doc.active_source_version == active_before

            # No new spans written.
            span_count_after = await session.scalar(
                select(func.count())
                .select_from(SourceSpan)
                .where(SourceSpan.document_id == document_id)
            )
            assert span_count_after == span_count_before

            # Prior pass's item superseded; new pass's item ready at the same version.
            old_item = await session.get(KnowledgeItem, old_item_id)
            assert old_item is not None
            assert old_item.status is KnowledgeItemStatus.SUPERSEDED
            ready_items = list(
                (
                    await session.execute(
                        select(KnowledgeItem).where(
                            KnowledgeItem.document_id == document_id,
                            KnowledgeItem.status == KnowledgeItemStatus.READY,
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(ready_items) == 1
            assert ready_items[0].id != old_item_id
            assert ready_items[0].normalized_title == "carrot soup"
            assert ready_items[0].source_version == 1

            # The prior pass's chunks are retained (not deleted).
            retained = set(
                (
                    await session.execute(
                        select(Chunk.id).where(Chunk.id.in_(old_chunk_ids))
                    )
                )
                .scalars()
                .all()
            )
            assert retained == old_chunk_ids
    finally:
        await _cleanup(session_factory, document_id, asset_id)


async def test_reuse_zero_winner_does_not_supersede_prior(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reuse pass that accepts no items must not retire the live set.

    With ``len(chosen) == 0`` the supersede call is skipped entirely (round-1 #1),
    so the prior ready item stays ready rather than being orphaned with no
    replacement."""
    monkeypatch.setattr(
        "rag_recipes.ingestion.jobs.extract_and_persist_spans", _noop_extract
    )
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        spans, fresh_count = await _seed_spans_and_run_fresh(
            session_factory, document_id, tmp_path
        )
        assert fresh_count == 1
        window_low, window_high = build_windows(spans, _WINDOW_SIZE, _OVERLAP)

        async with session_factory() as session:
            old_item_id = await session.scalar(
                select(KnowledgeItem.id).where(
                    KnowledgeItem.document_id == document_id,
                    KnowledgeItem.status == KnowledgeItemStatus.READY,
                )
            )
            assert old_item_id is not None

        await _requeue(session_factory, document_id)

        # Reuse leg returns zero accepted items for every window.
        _bump_prompt_version(monkeypatch, _REUSE_PROMPT_VERSION)
        reuse_provider = FakeLLMProvider(
            responses_by_hash={
                _request_hash_for(window_low, _REUSE_PROMPT_VERSION): _empty_output(),
                _request_hash_for(window_high, _REUSE_PROMPT_VERSION): _empty_output(),
            }
        )
        reuse_count = await process_document(
            _ctx(session_factory, reuse_provider, tmp_path),
            document_id,
            source_version=1,
            reuse_source_spans=True,
        )
        assert reuse_count == 0  # no winners

        async with session_factory() as session:
            # The prior ready item is untouched — never superseded.
            old_item = await session.get(KnowledgeItem, old_item_id)
            assert old_item is not None
            assert old_item.status is KnowledgeItemStatus.READY
            superseded = await session.scalar(
                select(func.count())
                .select_from(KnowledgeItem)
                .where(
                    KnowledgeItem.document_id == document_id,
                    KnowledgeItem.status == KnowledgeItemStatus.SUPERSEDED,
                )
            )
            assert superseded == 0
    finally:
        await _cleanup(session_factory, document_id, asset_id)


async def test_reuse_needs_review_winner_does_not_finalize_ready_off_stale_chunks(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reuse whose only winner is NEEDS_REVIEW must not finalize READY (review #1).

    The prior pass's chunks are retained (parented to the now-superseded items),
    but they must not make the document look searchable: the terminal status and
    the embedding stage count/select only chunks of currently-READY items, so a
    reuse with no READY winner ends NEEDS_REVIEW with nothing live."""
    monkeypatch.setattr(
        "rag_recipes.ingestion.jobs.extract_and_persist_spans", _noop_extract
    )
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        spans, fresh_count = await _seed_spans_and_run_fresh(
            session_factory, document_id, tmp_path
        )
        assert fresh_count == 1
        window_low, window_high = build_windows(spans, _WINDOW_SIZE, _OVERLAP)

        async with session_factory() as session:
            old_item_id = await session.scalar(
                select(KnowledgeItem.id).where(
                    KnowledgeItem.document_id == document_id,
                    KnowledgeItem.status == KnowledgeItemStatus.READY,
                )
            )
            assert old_item_id is not None

        await _requeue(session_factory, document_id)

        # Reuse leg: a winner with low overall confidence (< 0.5 threshold) carries
        # a soft-validation warning, so finalize promotes it to NEEDS_REVIEW — no
        # new chunks are written.
        _bump_prompt_version(monkeypatch, _REUSE_PROMPT_VERSION)
        reuse_provider = FakeLLMProvider(
            responses_by_hash={
                _request_hash_for(window_low, _REUSE_PROMPT_VERSION): _recipe_output(
                    title="Carrot Soup", cited_span_id="span_p3", overall=0.3
                ),
                _request_hash_for(window_high, _REUSE_PROMPT_VERSION): _recipe_output(
                    title="Carrot Soup", cited_span_id="span_p3", overall=0.3
                ),
            }
        )
        reuse_count = await process_document(
            _ctx(session_factory, reuse_provider, tmp_path),
            document_id,
            source_version=1,
            reuse_source_spans=True,
        )
        assert reuse_count == 1  # one (needs_review) winner

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            # The crux: stale superseded chunks must NOT finalize the doc READY.
            assert doc.status is DocumentStatus.NEEDS_REVIEW
            # Prior item superseded; new winner is needs_review.
            old_item = await session.get(KnowledgeItem, old_item_id)
            assert old_item is not None
            assert old_item.status is KnowledgeItemStatus.SUPERSEDED
            ready_count = await session.scalar(
                select(func.count())
                .select_from(KnowledgeItem)
                .where(
                    KnowledgeItem.document_id == document_id,
                    KnowledgeItem.status == KnowledgeItemStatus.READY,
                )
            )
            assert ready_count == 0
            # No chunk of a currently-READY item exists, so nothing is live/embedded.
            live_chunks = await session.scalar(
                select(func.count())
                .select_from(Chunk)
                .join(KnowledgeItem, Chunk.parent_id == KnowledgeItem.id)
                .where(
                    Chunk.document_id == document_id,
                    KnowledgeItem.status == KnowledgeItemStatus.READY,
                )
            )
            assert live_chunks == 0
    finally:
        await _cleanup(session_factory, document_id, asset_id)


async def test_reuse_technical_failure_marks_failed_and_keeps_prior_items(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reuse run that hits a caught provider failure leaves the prior set active.

    ``FakeLLMProvider(fail_technically=True)`` raises ``LLMTechnicalError``;
    ``process_document`` marks the document FAILED and re-raises (round-1 #7).
    Finalize never runs, so the prior ready items are not superseded and
    ``active_source_version`` is unchanged."""
    monkeypatch.setattr(
        "rag_recipes.ingestion.jobs.extract_and_persist_spans", _noop_extract
    )
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        _, fresh_count = await _seed_spans_and_run_fresh(
            session_factory, document_id, tmp_path
        )
        assert fresh_count == 1

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            active_before = doc.active_source_version
            assert active_before == 1
            old_item_id = await session.scalar(
                select(KnowledgeItem.id).where(
                    KnowledgeItem.document_id == document_id,
                    KnowledgeItem.status == KnowledgeItemStatus.READY,
                )
            )
            assert old_item_id is not None

        await _requeue(session_factory, document_id)

        # Reuse leg fails mid-extraction (bumped prompt so the provider is actually
        # called, not skip-set / cache short-circuited).
        _bump_prompt_version(monkeypatch, _REUSE_PROMPT_VERSION)
        failing_provider = FakeLLMProvider(fail_technically=True)
        with pytest.raises(LLMTechnicalError):
            await process_document(
                _ctx(session_factory, failing_provider, tmp_path),
                document_id,
                source_version=1,
                reuse_source_spans=True,
            )

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            assert doc.status is DocumentStatus.FAILED
            # Active version untouched by the failed reuse.
            assert doc.active_source_version == active_before
            # The prior ready item survives — finalize never ran, so no supersede.
            old_item = await session.get(KnowledgeItem, old_item_id)
            assert old_item is not None
            assert old_item.status is KnowledgeItemStatus.READY
    finally:
        await _cleanup(session_factory, document_id, asset_id)
