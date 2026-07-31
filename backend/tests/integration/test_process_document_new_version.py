"""End-to-end new-source-version reprocess test (Epic 11.2).

Drives the *real* ``process_document`` twice against the Postgres ``test_engine``:
a fresh run that lands the document ready at v1 (active_source_version=1), then a
``process_document(..., source_version=2, reuse_source_spans=False)`` new-version
run. The new-version run uses the unflagged fresh path, so it re-extracts PDF text
into a brand-new version — here ``extract_and_persist_spans`` is monkeypatched to
seed v2 spans directly (the production extractor isn't injectable), mirroring how
``test_process_document_dedup.py`` bypasses the PDF layer.

Asserts the active-version handoff: on an accepted (a surviving READY item) v2 run
the document's ``active_source_version`` flips 1→2 and the v1 items are SUPERSEDED;
on an all-needs_review v2 run there is no flip and the v1 items stay READY; on a
mid-run failure there is no flip, the v1 items stay READY, and the partial v2
artifacts (spans) exist but are inactive. v1 and v2 spans coexist throughout.
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
from rag_recipes.providers.errors import EmbeddingTechnicalError, LLMTechnicalError
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


def _build_spans(document_id: str, *, source_version: int) -> list[SourceSpan]:
    """Five per-page spans at ``source_version`` (version-scoped ids) — page 3 has
    the recipe. v1 and v2 ids/locator_hashes differ so they coexist in the table."""
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
                id=f"span_v{source_version}_p{page}",
                document_id=document_id,
                source_version=source_version,
                source_type=SourceType.PDF,
                locator=locator,
                locator_hash=hashlib.sha256(
                    f"v{source_version}-p{page}".encode()
                ).hexdigest(),
                text=text,
                text_hash=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    return spans


def _request_hash_for(window: Window) -> str:
    request = StructuredOutputRequest(
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input=_render_prompt(format_window_for_llm(window)),
        json_schema=build_recipe_v1_json_schema(),
    )
    return FakeLLMProvider.request_hash(request)


def _recipe_output(
    *, title: str, cited_span_id: str, overall: float
) -> dict[str, object]:
    recipe = _make_recipe(
        title=title,
        source_span_ids=[cited_span_id],
        structured_data=_make_structured_data(
            steps=[_make_step(source_span_ids=[cited_span_id])]
        ),
        confidence=_recipe_confidence(overall=overall, boundary=overall),
    )
    return RecipeExtractionOutput(items=[recipe]).model_dump(mode="json", by_alias=True)


def _provider_for(
    spans: list[SourceSpan], *, title: str, overall_high: float
) -> FakeLLMProvider:
    """A fake keyed on a span set's two windows; page-3 recipe cited from page 3."""
    window_low, window_high = build_windows(spans, _WINDOW_SIZE, _OVERLAP)
    cited = next(s.id for s in spans if s.locator["page_start"] == 3)
    return FakeLLMProvider(
        responses_by_hash={
            _request_hash_for(window_low): _recipe_output(
                title=title, cited_span_id=cited, overall=min(overall_high, 0.6)
            ),
            _request_hash_for(window_high): _recipe_output(
                title=title, cited_span_id=cited, overall=overall_high
            ),
        }
    )


async def _seed_document(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[str, str]:
    async with session_factory() as session:
        repo = DocumentRepository(session)
        pdf_bytes = b"%PDF-1.4 new-version fixture"
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


async def _run_fresh_v1(
    session_factory: async_sessionmaker[AsyncSession],
    document_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """Seed v1 spans + run the fresh leg to READY; return the v1 ready item id."""
    v1_spans = _build_spans(document_id, source_version=1)
    async with session_factory() as session:
        session.add_all(v1_spans)
        await session.commit()
    monkeypatch.setattr(
        "rag_recipes.ingestion.jobs.extract_and_persist_spans",
        _noop_extract,
    )
    provider = _provider_for(v1_spans, title="Tomato Soup", overall_high=0.9)
    count = await process_document(_ctx(session_factory, provider, tmp_path), document_id)
    assert count == 1
    async with session_factory() as session:
        item_id = await session.scalar(
            select(KnowledgeItem.id).where(
                KnowledgeItem.document_id == document_id,
                KnowledgeItem.status == KnowledgeItemStatus.READY,
            )
        )
        assert item_id is not None
        doc = await session.get(Document, document_id)
        assert doc is not None and doc.active_source_version == 1
    return item_id


async def _noop_extract(*args: object, **kwargs: object) -> int:
    return 5


def _patch_extract_seeds_v2(
    monkeypatch: pytest.MonkeyPatch,
    document_id: str,
) -> None:
    """Make the fresh path's extract_and_persist_spans seed v2 spans (the prod
    extractor isn't injectable), so the new-version run has source text at v2."""

    async def _seed_v2(session: AsyncSession, **kwargs: object) -> int:
        session.add_all(_build_spans(document_id, source_version=2))
        await session.flush()
        return 5

    monkeypatch.setattr(
        "rag_recipes.ingestion.jobs.extract_and_persist_spans", _seed_v2
    )


async def test_new_version_success_flips_active_and_supersedes_v1(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        v1_item_id = await _run_fresh_v1(
            session_factory, document_id, tmp_path, monkeypatch
        )

        async with session_factory() as session:
            await transition_to(session, document_id, DocumentStatus.QUEUED)
            await session.commit()

        # New-version leg: the fresh path seeds v2 spans, extracts a new recipe.
        _patch_extract_seeds_v2(monkeypatch, document_id)
        v2_spans = _build_spans(document_id, source_version=2)
        provider = _provider_for(v2_spans, title="Carrot Soup", overall_high=0.9)
        count = await process_document(
            _ctx(session_factory, provider, tmp_path),
            document_id,
            source_version=2,
            reuse_source_spans=False,
        )
        assert count == 1

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            # Active version flipped 1 -> 2.
            assert doc.active_source_version == 2
            # v1 items superseded; the v2 winner is ready at v2.
            v1_item = await session.get(KnowledgeItem, v1_item_id)
            assert v1_item is not None
            assert v1_item.status is KnowledgeItemStatus.SUPERSEDED
            ready = list(
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
            assert len(ready) == 1
            assert ready[0].source_version == 2
            assert ready[0].normalized_title == "carrot soup"
            # v1 and v2 spans coexist; version-scoped counts are isolated.
            v1_count = await session.scalar(
                select(func.count())
                .select_from(SourceSpan)
                .where(
                    SourceSpan.document_id == document_id,
                    SourceSpan.source_version == 1,
                )
            )
            v2_count = await session.scalar(
                select(func.count())
                .select_from(SourceSpan)
                .where(
                    SourceSpan.document_id == document_id,
                    SourceSpan.source_version == 2,
                )
            )
            assert v1_count == 5
            assert v2_count == 5
    finally:
        await _cleanup(session_factory, document_id, asset_id)


async def test_new_version_all_needs_review_does_not_flip(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        v1_item_id = await _run_fresh_v1(
            session_factory, document_id, tmp_path, monkeypatch
        )

        async with session_factory() as session:
            await transition_to(session, document_id, DocumentStatus.QUEUED)
            await session.commit()

        # v2 winner is low-confidence → needs_review → not an accepted replacement.
        _patch_extract_seeds_v2(monkeypatch, document_id)
        v2_spans = _build_spans(document_id, source_version=2)
        provider = _provider_for(v2_spans, title="Carrot Soup", overall_high=0.3)
        await process_document(
            _ctx(session_factory, provider, tmp_path),
            document_id,
            source_version=2,
            reuse_source_spans=False,
        )

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            # No accepted READY winner → no flip; v1 stays active and ready.
            assert doc.active_source_version == 1
            v1_item = await session.get(KnowledgeItem, v1_item_id)
            assert v1_item is not None
            assert v1_item.status is KnowledgeItemStatus.READY
    finally:
        await _cleanup(session_factory, document_id, asset_id)


class _RaisingEmbeddingProvider(FakeEmbeddingProvider):
    """A FakeEmbeddingProvider whose embed_batch always raises (failure scaffolding)."""

    async def embed_batch(self, texts: list[Any], *, trace_context: Any = None) -> Any:
        raise EmbeddingTechnicalError("boom")


class _RejectsTextEmbeddingProvider(FakeEmbeddingProvider):
    """Raises if asked to embed any text containing ``reject`` — lets a test prove
    the embedding stage only receives the current version's chunk texts."""

    def __init__(self, *, reject: str) -> None:
        super().__init__()
        self._reject = reject

    async def embed_batch(self, texts: list[Any], *, trace_context: Any = None) -> Any:
        if any(self._reject in text for text in texts):
            raise EmbeddingTechnicalError(f"refused text containing {self._reject!r}")
        return await super().embed_batch(texts, trace_context=trace_context)


async def test_new_version_embedding_is_scoped_to_its_own_version(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v2 run's embedding stage must receive only v2 chunk texts (review #2).

    The prior v1 items stay READY until the handoff at the READY gate, so an
    unscoped embed would feed the v1 ("Tomato") chunks to the provider too. Here the
    provider rejects any "Tomato" text; the v2 run must still reach READY and flip
    active to 2, proving it never touched the prior version's chunks."""
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        await _run_fresh_v1(session_factory, document_id, tmp_path, monkeypatch)

        async with session_factory() as session:
            await transition_to(session, document_id, DocumentStatus.QUEUED)
            await session.commit()

        _patch_extract_seeds_v2(monkeypatch, document_id)
        v2_spans = _build_spans(document_id, source_version=2)
        provider = _provider_for(v2_spans, title="Carrot Soup", overall_high=0.9)
        ctx = _ctx(session_factory, provider, tmp_path)
        # Would raise if the v2 run ever embedded a v1 ("Tomato") chunk.
        ctx["embedding_provider"] = _RejectsTextEmbeddingProvider(reject="Tomato")
        count = await process_document(
            ctx, document_id, source_version=2, reuse_source_spans=False
        )
        assert count == 1

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            assert doc.status is DocumentStatus.READY
            assert doc.active_source_version == 2
    finally:
        await _cleanup(session_factory, document_id, asset_id)


async def test_new_version_embedding_failure_keeps_prior_version_active(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v2 run that fails AFTER finalize (during embedding) must not have flipped
    the active version or superseded v1 (review #1).

    The active-version handoff (flip + cross-version supersede) is gated on reaching
    READY, which is after embedding. So an embedding failure leaves the document
    FAILED with active_source_version still 1 and the v1 items still READY — the
    rollback path is intact; the v2 artifacts exist but are inactive."""
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        v1_item_id = await _run_fresh_v1(
            session_factory, document_id, tmp_path, monkeypatch
        )

        async with session_factory() as session:
            await transition_to(session, document_id, DocumentStatus.QUEUED)
            await session.commit()

        # v2 extraction succeeds (a READY winner) but embedding then fails.
        _patch_extract_seeds_v2(monkeypatch, document_id)
        v2_spans = _build_spans(document_id, source_version=2)
        provider = _provider_for(v2_spans, title="Carrot Soup", overall_high=0.9)
        ctx = _ctx(session_factory, provider, tmp_path)
        ctx["embedding_provider"] = _RaisingEmbeddingProvider()
        with pytest.raises(EmbeddingTechnicalError):
            await process_document(
                ctx,
                document_id,
                source_version=2,
                reuse_source_spans=False,
            )

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            assert doc.status is DocumentStatus.FAILED
            # The handoff never reached the READY gate: no flip, no supersede.
            assert doc.active_source_version == 1
            v1_item = await session.get(KnowledgeItem, v1_item_id)
            assert v1_item is not None
            assert v1_item.status is KnowledgeItemStatus.READY
            # The v2 winner exists (it was promoted in finalize) but is not active.
            v2_ready = await session.scalar(
                select(func.count())
                .select_from(KnowledgeItem)
                .where(
                    KnowledgeItem.document_id == document_id,
                    KnowledgeItem.source_version == 2,
                    KnowledgeItem.status == KnowledgeItemStatus.READY,
                )
            )
            assert v2_ready == 1
    finally:
        await _cleanup(session_factory, document_id, asset_id)


async def test_new_version_failure_mid_run_does_not_flip(
    test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = build_session_factory(test_engine)
    document_id, asset_id = await _seed_document(session_factory)
    try:
        v1_item_id = await _run_fresh_v1(
            session_factory, document_id, tmp_path, monkeypatch
        )

        async with session_factory() as session:
            await transition_to(session, document_id, DocumentStatus.QUEUED)
            await session.commit()

        # The v2 extraction fails after the spans are written (partial artifacts).
        _patch_extract_seeds_v2(monkeypatch, document_id)
        provider = FakeLLMProvider(fail_technically=True)
        with pytest.raises(LLMTechnicalError):
            await process_document(
                _ctx(session_factory, provider, tmp_path),
                document_id,
                source_version=2,
                reuse_source_spans=False,
            )

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            assert doc is not None
            assert doc.status is DocumentStatus.FAILED
            # No flip, no supersede — finalize never ran.
            assert doc.active_source_version == 1
            v1_item = await session.get(KnowledgeItem, v1_item_id)
            assert v1_item is not None
            assert v1_item.status is KnowledgeItemStatus.READY
            # Partial v2 artifacts exist (the spans were written) but are inactive.
            v2_count = await session.scalar(
                select(func.count())
                .select_from(SourceSpan)
                .where(
                    SourceSpan.document_id == document_id,
                    SourceSpan.source_version == 2,
                )
            )
            assert v2_count == 5
    finally:
        await _cleanup(session_factory, document_id, asset_id)
