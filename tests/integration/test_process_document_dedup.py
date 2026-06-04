"""End-to-end dedup test: overlapping windows of one recipe collapse to one item.

Drives the *real* ``process_document`` through its extraction + dedup stage. The
PDF layer is bypassed by seeding ``SourceSpan`` rows directly (with explicit ids
so the LLM fake can be keyed by the exact rendered prompt) and monkeypatching
``extract_and_persist_spans`` to a no-op — exactly the dedup path
(``build_windows`` → ``run_extraction`` → ``persist`` → ``select_best`` → prune),
without a fragile multi-page PDF fixture (TASK-004).

Window math (pinned): 5 spans, ``pdf_window_size_pages=3``, ``pdf_overlap_pages=1``
→ windows ``pages 1-3`` and ``pages 3-5`` (per 9.1). The recipe lives on page 3,
so it appears in *both* windows; the fake returns the same ``normalized_title``
with a higher confidence for the ``3-5`` window, so dedup must keep that one.
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
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.ingestion_failure import IngestionFailure
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.repositories.failures import FailuresRepository
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


async def _noop_extract(*args: object, **kwargs: object) -> int:
    """Stand-in for extract_and_persist_spans: spans are pre-seeded, so do nothing."""
    return 5


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


def _request_hash_for(window: Window) -> str:
    """Replicate run_extraction's request so the fake can be keyed per window.

    run_extraction builds the request from the rendered prompt and the provider's
    ``provider``/``default_model`` labels (here ``fake``/``fake-model``); the fake
    keys its canned responses on exactly this hash.
    """
    request = StructuredOutputRequest(
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input=_render_prompt(format_window_for_llm(window)),
        json_schema=build_recipe_v1_json_schema(),
    )
    return FakeLLMProvider.request_hash(request)


def _recipe_output(*, cited_span_id: str, overall: float) -> dict[str, object]:
    """A one-recipe ``recipe.v1`` output citing ``cited_span_id`` at confidence ``overall``."""
    recipe = _make_recipe(
        title="Tomato Soup",
        source_span_ids=[cited_span_id],
        structured_data=_make_structured_data(
            steps=[_make_step(source_span_ids=[cited_span_id])]
        ),
        confidence=_recipe_confidence(overall=overall, boundary=overall),
    )
    return RecipeExtractionOutput(items=[recipe]).model_dump(mode="json", by_alias=True)


async def _seed_document(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[str, str]:
    """Insert a QUEUED Document + SourceAsset; return (document_id, asset_id)."""
    async with session_factory() as session:
        repo = DocumentRepository(session)
        pdf_bytes = b"%PDF-1.4 dedup fixture"
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
        for model in (KnowledgeItem, ExtractionRun, SourceSpan, IngestionFailure):
            await session.execute(
                delete(model).where(model.document_id == document_id)
            )
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
    }


async def test_overlapping_windows_resolve_to_one_knowledge_item(
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
        spans = _build_spans(document_id)
        async with session_factory() as session:
            session.add_all(spans)
            await session.commit()

        # Window 1-3 (no page 5) scores lower; window 3-5 (has page 5) scores
        # higher — so the 3-5 capture must be the survivor.
        window_low, window_high = build_windows(spans, _WINDOW_SIZE, _OVERLAP)
        provider = FakeLLMProvider(
            responses_by_hash={
                _request_hash_for(window_low): _recipe_output(
                    cited_span_id="span_p3", overall=0.6
                ),
                _request_hash_for(window_high): _recipe_output(
                    cited_span_id="span_p3", overall=0.9
                ),
            }
        )

        result = await process_document(
            _ctx(session_factory, provider, tmp_path), document_id
        )
        assert result == 1  # one surviving candidate

        async with session_factory() as session:
            status = await session.scalar(
                select(Document.status).where(Document.id == document_id)
            )
            assert status == DocumentStatus.CREATING_CHUNKS

            items = (
                await session.execute(
                    select(KnowledgeItem).where(
                        KnowledgeItem.document_id == document_id
                    )
                )
            ).scalars().all()
            # Two overlapping captures collapsed to exactly one item...
            assert len(items) == 1
            assert items[0].normalized_title == "tomato soup"
            # ...and the survivor is the higher-scored (0.9) capture.
            assert items[0].confidence is not None
            assert items[0].confidence["overall"] == pytest.approx(0.9)

            # The audit trail is preserved: one ExtractionRun per window (the
            # losing *item* was pruned, not its run).
            run_count = await session.scalar(
                select(func.count())
                .select_from(ExtractionRun)
                .where(ExtractionRun.document_id == document_id)
            )
            assert run_count == 2
    finally:
        await _cleanup(session_factory, document_id, asset_id)


async def test_llm_technical_failure_marks_document_failed(
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
        async with session_factory() as session:
            session.add_all(_build_spans(document_id))
            await session.commit()

        provider = FakeLLMProvider(fail_technically=True)
        # process_document re-raises after mark_failed (so arq's result reflects it).
        with pytest.raises(LLMTechnicalError):
            await process_document(
                _ctx(session_factory, provider, tmp_path), document_id
            )

        async with session_factory() as session:
            status = await session.scalar(
                select(Document.status).where(Document.id == document_id)
            )
            assert status == DocumentStatus.FAILED

            failures = await FailuresRepository(session).list_failures(document_id)
            assert len(failures) == 1
            assert failures[0].reason == "llm_extraction_failed"
    finally:
        await _cleanup(session_factory, document_id, asset_id)
