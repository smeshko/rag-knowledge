"""End-to-end retrieval-eval → baseline → diff loop (Epic 16 Phase 16.2).

Seeds a synthetic two-item corpus **directly into the DB with fake providers**
(modeled on ``test_search.py``'s helpers — no ingestion pipeline, no arq
worker, no live LLM/embedding call), runs ``run_retrieval_eval`` in-process,
saves a baseline, changes ``recipe_keyword_boost_title`` in the overridden
``Settings``, re-runs, and asserts the diff reflects the ranking change.

The corpus seeds two competing items where one carries a ``RECIPE_TITLE``
chunk matching the query: ``recipe_keyword_boost_title`` applies only to
title chunks, so lowering it demotes that item and the diff must show a
regression. Everything is written under ``tmp_path`` — never into the repo's
``evals/reports/`` or ``evals/baselines/``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from evals.reports import (
    ReportRun,
    _unwrap,
    diff_against_baseline,
    diff_retrieval,
    save_as_baseline,
)
from evals.retrieval import run_retrieval_eval
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import (
    get_embedding_provider,
    get_reranker_provider,
    get_session,
    get_settings,
)
from rag_recipes.config import Settings
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.storage.enums import (
    ChunkParentType,
    ChunkType,
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from tests.integration.conftest import AUTH_HEADERS, TEST_API_TOKEN

pytestmark = pytest.mark.asyncio

_FAKE_PROVIDER = "fake"
_FAKE_MODEL = "fake-embedding"
_QUERY = "white beans soup"


@asynccontextmanager
async def _client(
    db_session: AsyncSession, *, title_boost: float | None = None
) -> AsyncIterator[tuple[httpx.AsyncClient, Settings]]:
    update: dict[str, Any] = {
        "personal_api_token": TEST_API_TOKEN,
        "embedding_provider": _FAKE_PROVIDER,
        "embedding_model": _FAKE_MODEL,
        "reranking_enabled": False,
    }
    if title_boost is not None:
        update["recipe_keyword_boost_title"] = title_boost
    settings = get_settings().model_copy(update=update)
    fake = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: fake
    app.dependency_overrides[get_reranker_provider] = lambda: None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers=AUTH_HEADERS,
        ) as client:
            yield client, settings
    finally:
        for dep in (get_session, get_settings, get_embedding_provider, get_reranker_provider):
            app.dependency_overrides.pop(dep, None)


async def _make_document(session: AsyncSession) -> str:
    repo = DocumentRepository(session)
    aid = new_id("asset")
    asset = await repo.add_source_asset(
        id=aid,
        source_type=SourceType.PDF,
        original_filename="cookbook.pdf",
        storage_provider="fake",
        storage_key=f"source-assets/{aid}/original.pdf",
        content_hash=hashlib.sha256(new_id("h").encode()).hexdigest(),
        upload_status=UploadStatus.UPLOADED,
    )
    doc = await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="Simple Thai Food",
        author="Leela",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=1,
        status=DocumentStatus.READY,
    )
    return doc.id


async def _make_span(session: AsyncSession, document_id: str, *, page: int) -> str:
    locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
    span = SourceSpan(
        id=new_id("span"),
        document_id=document_id,
        source_version=1,
        source_type=SourceType.PDF,
        locator=locator,
        locator_hash=hashlib.sha256(f"{document_id}-{page}".encode()).hexdigest(),
        text="recipe text",
        text_hash=hashlib.sha256(f"t{page}".encode()).hexdigest(),
    )
    session.add(span)
    await session.flush()
    return span.id


async def _make_recipe(
    session: AsyncSession,
    document_id: str,
    *,
    title: str,
    chunk_type: ChunkType,
    chunk_text: str,
    embed_text: str,
    provider: FakeEmbeddingProvider,
    span_ids: list[str],
) -> str:
    run = ExtractionRun(
        document_id=document_id,
        source_version=1,
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=[],
        input_hash=new_id("hash"),
        status=ExtractionRunStatus.SUCCESS,
        output_json=None,
    )
    session.add(run)
    await session.flush()
    item = KnowledgeItem(
        document_id=document_id,
        extraction_run_id=run.id,
        source_version=1,
        item_type="recipe",
        title=title,
        normalized_title=title.lower(),
        summary="A cozy bowl of white beans.",
        body_text="x" * 100,
        source_span_ids=[],
        structured_data={"schema": "recipe.v1", "warnings": []},
        confidence={"overall": 0.88},
        status=KnowledgeItemStatus.READY,
    )
    session.add(item)
    await session.flush()
    chunk = Chunk(
        document_id=document_id,
        parent_type=ChunkParentType.KNOWLEDGE_ITEM,
        parent_id=item.id,
        chunk_type=chunk_type,
        text=chunk_text,
        text_hash=hashlib.sha256(chunk_text.encode()).hexdigest(),
        source_span_ids=span_ids,
    )
    session.add(chunk)
    await session.flush()
    emb = await provider.embed_text(embed_text)
    session.add(
        ChunkEmbedding(
            chunk_id=chunk.id,
            embedding_provider=_FAKE_PROVIDER,
            embedding_model=_FAKE_MODEL,
            embedding_dimensions=emb.dimensions,
            embedding_vector=emb.vector,
        )
    )
    await session.flush()
    return item.id


async def _seed_competing_items(session: AsyncSession) -> tuple[str, str]:
    """Two competing items; only the first carries a matching RECIPE_TITLE chunk."""
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    doc = await _make_document(session)
    span = await _make_span(session, doc, page=1)
    title_item = await _make_recipe(
        session,
        doc,
        title="White Beans Soup",
        chunk_type=ChunkType.RECIPE_TITLE,
        chunk_text=_QUERY,
        embed_text="off vector text one",
        provider=provider,
        span_ids=[span],
    )
    full_item = await _make_recipe(
        session,
        doc,
        title="Hearty Bean Stew",
        chunk_type=ChunkType.RECIPE_FULL,
        chunk_text="a hearty stew of white beans soup and herbs",
        embed_text="off vector text two",
        provider=provider,
        span_ids=[span],
    )
    return title_item, full_item


def _write_fixture_set(tmp_path: Path, expected_item_id: str) -> Path:
    """Synthetic golden set with the qrels id resolved from the seeded corpus."""
    fixture_root = tmp_path / "fixtures"
    set_dir = fixture_root / "queries" / "integration"
    set_dir.mkdir(parents=True)
    (set_dir / "queries.tsv").write_text(f"q1\t{_QUERY}\n", encoding="utf-8")
    (set_dir / "qrels.tsv").write_text(f"q1\t{expected_item_id}\t1\n", encoding="utf-8")
    return fixture_root


async def _run_eval(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    label: str,
    fixtures_root: Path,
    reports_root: Path,
) -> ReportRun:
    async def _search(query_text: str, *, mode: str, limit: int) -> dict[str, Any]:
        response = await client.post(
            "/api/v1/search",
            json={
                "query": query_text,
                "mode": mode,
                "limit": limit,
                "filters": {"exclude_needs_review": True},
            },
        )
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body

    return await run_retrieval_eval(
        "integration",
        k=10,
        label=label,
        mode="keyword",
        search=_search,
        report_factory=lambda run_label: ReportRun(
            run_label, reports_root=reports_root, settings=settings
        ),
        settings=settings,
        fixtures_root=fixtures_root,
    )


def _expected_rank(report: ReportRun, item_id: str) -> int | None:
    payload = json.loads((report.path / "results.json").read_text(encoding="utf-8"))["results"]
    rank: int | None = payload["per_query"]["q1"]["expected_item_ranks"][item_id]
    return rank


async def test_boost_change_produces_a_regression_diff(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    title_item, full_item = await _seed_competing_items(db_session)
    fixtures_root = _write_fixture_set(tmp_path, title_item)
    reports_root = tmp_path / "reports"

    # Pass 1: default recipe_keyword_boost_title (1.40) — the title item leads.
    async with _client(db_session) as (client, settings):
        before = await _run_eval(
            client, settings, label="before", fixtures_root=fixtures_root,
            reports_root=reports_root,
        )
    baseline_path = save_as_baseline(
        before.path, "retrieval", baselines_root=tmp_path / "baselines"
    )

    # Pass 2: crush the title boost — the RECIPE_TITLE item is demoted below
    # the competing RECIPE_FULL item, worsening the expected item's rank.
    async with _client(db_session, title_boost=0.01) as (client, settings):
        after = await _run_eval(
            client, settings, label="after", fixtures_root=fixtures_root,
            reports_root=reports_root,
        )

    rank_before = _expected_rank(before, title_item)
    rank_after = _expected_rank(after, title_item)
    assert rank_before is not None
    assert rank_after is None or rank_after > rank_before  # the boost change demoted it

    result = diff_against_baseline(baseline_path, after.path)
    assert result.status == "regression"
    assert "NDCG@10" in result.summary
    assert "[REGRESSION" in result.summary

    # The pure diff pinpoints the specific query.
    baseline_payload = _unwrap(json.loads(baseline_path.read_text(encoding="utf-8")))
    current_payload = _unwrap(
        json.loads((after.path / "results.json").read_text(encoding="utf-8"))
    )
    diff = diff_retrieval(baseline_payload, current_payload)
    assert diff.per_query_ndcg_delta["q1"] < 0
    assert diff.warnings == []  # same query set / mode / rerank state / model

    # Reports and baselines stayed under tmp_path; both runs produced the
    # full report trio.
    for report in (before, after):
        assert report.path.is_relative_to(reports_root)
        assert (report.path / "per_query.md").is_file()
        assert (report.path / "summary.md").is_file()
    assert baseline_path.is_relative_to(tmp_path)


async def test_identical_reruns_diff_clean(db_session: AsyncSession, tmp_path: Path) -> None:
    title_item, _ = await _seed_competing_items(db_session)
    fixtures_root = _write_fixture_set(tmp_path, title_item)
    reports_root = tmp_path / "reports"

    async with _client(db_session) as (client, settings):
        first = await _run_eval(
            client, settings, label="first", fixtures_root=fixtures_root,
            reports_root=reports_root,
        )
        second = await _run_eval(
            client, settings, label="second", fixtures_root=fixtures_root,
            reports_root=reports_root,
        )
    baseline_path = save_as_baseline(
        first.path, "retrieval", baselines_root=tmp_path / "baselines"
    )
    result = diff_against_baseline(baseline_path, second.path)
    assert result.status == "no_change"
    assert result.changes == []
