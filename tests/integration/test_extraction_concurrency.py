"""Concurrent window extraction: fan out the provider call, keep writes serial.

``_run_extraction_batches`` used to await one ``run_extraction`` per window in a
plain loop, so a book's wall time was the sum of its windows — ~151s each on a
real cookbook. The provider call is ~99% of that time and touches no database,
but it could not simply be gathered: every window shared the batch's single
``AsyncSession``, and an ``AsyncSession`` is not safe for concurrent use (its
flushes would interleave on one connection).

The split these tests pin: ``call_provider_for_window`` fans out under a
semaphore, then ``record_window_extraction`` writes each result sequentially on
the batch session. So the assertions are about *both* halves — that calls really
do overlap when allowed to, and that nothing about ordering, cache hits, or
failure handling changed underneath.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest
from sqlalchemy import Integer, delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag_recipes.config import get_settings
from rag_recipes.ingestion.jobs import _run_extraction_batches
from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    RecipeExtractionOutput,
)
from rag_recipes.ingestion.pipeline.windows import build_windows, compute_input_hash
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    TokenUsage,
)
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from tests.unit.ingestion.test_validation import (
    _make_recipe,
    _make_step,
    _make_structured_data,
)

pytestmark = pytest.mark.asyncio

#: 8 pages at window=3/overlap=1 → step 2 → 4 windows, enough that a
#: concurrency of 4 is visibly different from 1 without a slow test.
_PAGES = 8
_WINDOW_SIZE = 3
_OVERLAP = 1
_CALL_DELAY = 0.05


class _ConcurrencyProbe(LLMProvider):
    """Records the high-water mark of simultaneously in-flight provider calls.

    The point of the whole change is that these overlap, and the only honest way
    to assert that is to watch them overlap. A plain call counter would pass just
    as happily against the old sequential loop.
    """

    provider = "fake"
    default_model = "fake-model"

    def __init__(self, *, fail_on_input: str | None = None) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[str] = []
        self._fail_on_input = fail_on_input

    async def generate_structured_output(
        self, request: StructuredOutputRequest, **kwargs: Any
    ) -> StructuredOutputResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls.append(request.input)
        try:
            await asyncio.sleep(_CALL_DELAY)
            if self._fail_on_input is not None and self._fail_on_input in request.input:
                raise LLMTechnicalError("provider exploded")
            return StructuredOutputResponse(
                output_json=_recipe_output(title=f"Recipe {len(self.calls)}"),
                parse_error=None,
                raw_text="{}",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                provider=self.provider,
                model=self.default_model,
            )
        finally:
            self.in_flight -= 1


def _recipe_output(*, title: str) -> dict[str, Any]:
    recipe = _make_recipe(
        title=title,
        source_span_ids=["span_c1"],
        structured_data=_make_structured_data(steps=[_make_step(source_span_ids=["span_c1"])]),
    )
    return RecipeExtractionOutput(items=[recipe]).model_dump(mode="json", by_alias=True)


def _settings(concurrency: int) -> Any:
    return get_settings().model_copy(
        update={
            "pdf_window_size_pages": _WINDOW_SIZE,
            "pdf_overlap_pages": _OVERLAP,
            # One batch holds every window, so the semaphore is the only thing
            # bounding concurrency — a batch boundary would confound the probe.
            "extraction_commit_batch_size": _PAGES,
            "extraction_max_concurrent_windows": concurrency,
        }
    )


async def _seed_document_row(session_factory: async_sessionmaker[AsyncSession]) -> str:
    """A document row and nothing else — enough to satisfy the ExtractionRun FK."""
    async with session_factory() as session:
        repo = DocumentRepository(session)
        asset = await repo.add_source_asset(
            id=new_id("asset"),
            source_type=SourceType.PDF,
            original_filename="other.pdf",
            storage_provider="fake",
            storage_key=f"source-assets/{new_id('asset')}/original.pdf",
            content_hash=hashlib.sha256(new_id("h").encode()).hexdigest(),
            upload_status=UploadStatus.UPLOADED,
        )
        document = await repo.add_document(
            asset_id=asset.id,
            category="recipes",
            subcategory=None,
            title="Someone else's cookbook",
            author="",
            source_type=SourceType.PDF,
            language=None,
            active_source_version=None,
            status=DocumentStatus.QUEUED,
        )
        await session.commit()
        return document.id


async def _seed(session_factory: async_sessionmaker[AsyncSession]) -> str:
    async with session_factory() as session:
        repo = DocumentRepository(session)
        asset = await repo.add_source_asset(
            id=new_id("asset"),
            source_type=SourceType.PDF,
            original_filename="cookbook.pdf",
            storage_provider="fake",
            storage_key=f"source-assets/{new_id('asset')}/original.pdf",
            content_hash=hashlib.sha256(new_id("h").encode()).hexdigest(),
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
        for page in range(1, _PAGES + 1):
            text = f"Page {page}. Ingredients: tomatoes. Method: simmer."
            locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
            session.add(
                SourceSpan(
                    id=f"span_c{page}" if page == 1 else new_id("span"),
                    document_id=document.id,
                    source_version=1,
                    source_type=SourceType.PDF,
                    locator=locator,
                    locator_hash=hashlib.sha256(f"{document.id}{page}".encode()).hexdigest(),
                    text=text,
                    text_hash=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        await session.commit()
        return document.id


@pytest.fixture
async def session_factory(test_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(test_engine, expire_on_commit=False)


@pytest.fixture(autouse=True)
async def _cleanup(session_factory: async_sessionmaker[AsyncSession]):
    yield
    async with session_factory() as session:
        await session.execute(delete(KnowledgeItem))
        await session.execute(delete(ExtractionRun))
        await session.execute(delete(SourceSpan))
        await session.execute(delete(Document))
        await session.execute(delete(SourceAsset))
        await session.commit()


async def test_concurrency_one_keeps_calls_strictly_sequential(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # The default. Nothing about the old behaviour may change for operators who
    # never touch the new setting.
    document_id = await _seed(session_factory)
    provider = _ConcurrencyProbe()

    await _run_extraction_batches(
        session_factory,
        document_id=document_id,
        source_version=1,
        settings=_settings(1),
        provider=provider,
        observability=None,
    )

    assert provider.max_in_flight == 1
    assert len(provider.calls) == 4


async def test_windows_overlap_when_concurrency_allows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    document_id = await _seed(session_factory)
    provider = _ConcurrencyProbe()

    await _run_extraction_batches(
        session_factory,
        document_id=document_id,
        source_version=1,
        settings=_settings(4),
        provider=provider,
        observability=None,
    )

    assert provider.max_in_flight == 4
    async with session_factory() as session:
        runs = (
            (
                await session.execute(
                    select(ExtractionRun).where(ExtractionRun.document_id == document_id)
                )
            )
            .scalars()
            .all()
        )
    # Every window still lands exactly one terminal run — the writes did not race.
    assert len(runs) == 4
    assert {r.status for r in runs} == {ExtractionRunStatus.SUCCESS}
    assert all(r.prompt_version == PROMPT_VERSION for r in runs)
    assert all(r.schema_version == SCHEMA_VERSION for r in runs)


async def test_provider_failure_still_raises_under_concurrency(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # gather() must not swallow the failure into a result list: the job's
    # mark_failed path depends on it propagating, exactly as in the serial loop.
    document_id = await _seed(session_factory)
    provider = _ConcurrencyProbe(fail_on_input="Page 3")

    with pytest.raises(LLMTechnicalError):
        await _run_extraction_batches(
            session_factory,
            document_id=document_id,
            source_version=1,
            settings=_settings(4),
            provider=provider,
            observability=None,
        )

    # Siblings were dispatched rather than abandoned to the first failure...
    assert len(provider.calls) == 4
    # ...and their results were *committed*, not rolled back with the failure:
    # a SUCCESS row only feeds the extraction cache once it survives a commit,
    # so a resume must find three cached windows plus the one FAILED audit row.
    async with session_factory() as session:
        runs = (
            (
                await session.execute(
                    select(ExtractionRun).where(ExtractionRun.document_id == document_id)
                )
            )
            .scalars()
            .all()
        )
    failed = sum(1 for r in runs if r.status is ExtractionRunStatus.FAILED)
    succeeded = sum(1 for r in runs if r.status is ExtractionRunStatus.SUCCESS)
    expected_failures = sum(1 for call in provider.calls if "Page 3" in call)
    assert failed == expected_failures >= 1
    assert succeeded == len(provider.calls) - expected_failures >= 1


async def test_cache_hits_never_reach_the_provider(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # The cache lookup moved off the per-window path into a pre-pass. A hit must
    # still short-circuit before any dispatch, or concurrency would quietly turn
    # free windows back into paid ones.
    #
    # ``_find_cached_run`` keys on (input_hash, provider, model, prompt_version,
    # schema_version) and NOT on document, so a prior run over the same span ids
    # under ANOTHER document id is a legitimate hit. Seeding it that way also
    # keeps ``done_hashes`` — which is scoped to this document + version — from
    # short-circuiting first and hiding what is under test.
    document_id = await _seed(session_factory)
    other_document_id = await _seed_document_row(session_factory)
    async with session_factory() as session:
        spans = (
            (
                await session.execute(
                    select(SourceSpan)
                    .where(SourceSpan.document_id == document_id)
                    .order_by(SourceSpan.locator["page_start"].astext.cast(Integer))
                )
            )
            .scalars()
            .all()
        )
        for window in build_windows(list(spans), _WINDOW_SIZE, _OVERLAP):
            session.add(
                ExtractionRun(
                    id=new_id("run"),
                    document_id=other_document_id,
                    source_version=1,
                    provider="fake",
                    model="fake-model",
                    prompt_version=PROMPT_VERSION,
                    schema_version=SCHEMA_VERSION,
                    input_source_span_ids=list(window.span_ids),
                    input_hash=compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION),
                    status=ExtractionRunStatus.SUCCESS,
                    output_json=_recipe_output(title="Cached Recipe"),
                )
            )
        await session.commit()

    provider = _ConcurrencyProbe()
    await _run_extraction_batches(
        session_factory,
        document_id=document_id,
        source_version=1,
        settings=_settings(4),
        provider=provider,
        observability=None,
    )

    assert provider.calls == []
    async with session_factory() as session:
        runs = (
            (
                await session.execute(
                    select(ExtractionRun).where(ExtractionRun.document_id == document_id)
                )
            )
            .scalars()
            .all()
        )
    assert len(runs) == 4
    assert {r.status for r in runs} == {ExtractionRunStatus.SUCCESS}
