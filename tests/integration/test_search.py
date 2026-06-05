"""Integration tests for POST /api/v1/search (doc 6 § 7).

Real end-to-end against the Epic 12 retrieval facade (FTS + pgvector) via
``db_session`` (the route is read-only, so no savepoint dance is needed). The
embedding provider and settings are overridden so the fake's stamped
``(provider, model)`` matches the configured ``embedding_provider``/``embedding_model``
(the 12.2 cross-space guard), and ``debug_endpoints_enabled`` is toggled per test.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import (
    get_embedding_provider,
    get_session,
    get_settings,
)
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
    db_session: AsyncSession, *, debug_enabled: bool = False, with_auth: bool = True
) -> AsyncIterator[httpx.AsyncClient]:
    settings = get_settings().model_copy(
        update={
            "personal_api_token": TEST_API_TOKEN,
            "embedding_provider": _FAKE_PROVIDER,
            "embedding_model": _FAKE_MODEL,
            "debug_endpoints_enabled": debug_enabled,
        }
    )
    fake = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: fake
    transport = httpx.ASGITransport(app=app)
    headers = AUTH_HEADERS if with_auth else {}
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=headers
        ) as client:
            yield client
    finally:
        for dep in (get_session, get_settings, get_embedding_provider):
            app.dependency_overrides.pop(dep, None)


async def _make_document(session: AsyncSession, *, category: str = "recipes") -> str:
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
        category=category,
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
    chunk_text: str,
    embed_text: str,
    provider: FakeEmbeddingProvider,
    span_ids: list[str],
    structured_data: dict[str, Any] | None = None,
    status: KnowledgeItemStatus = KnowledgeItemStatus.READY,
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
        structured_data=structured_data or {"schema": "recipe.v1", "warnings": []},
        confidence={"overall": 0.88},
        status=status,
    )
    session.add(item)
    await session.flush()
    chunk = Chunk(
        document_id=document_id,
        parent_type=ChunkParentType.KNOWLEDGE_ITEM,
        parent_id=item.id,
        chunk_type=ChunkType.RECIPE_FULL,
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


_RECIPE_STRUCTURED = {
    "schema": "recipe.v1",
    "yield": "Serves 4",
    "cook_time": "35 minutes",
    "ingredients": [
        {
            "raw_text": "1 cup white beans",
            "item_text": "white beans",
            "item_normalized": "white beans",
        },
        {"raw_text": "2 cups stock", "item_text": "stock", "item_normalized": "vegetable stock"},
    ],
    "warnings": [],
}


async def _seed_one_recipe(session: AsyncSession, provider: FakeEmbeddingProvider) -> str:
    doc = await _make_document(session)
    span = await _make_span(session, doc, page=42)
    return await _make_recipe(
        session,
        doc,
        title="Cozy White Bean Soup",
        chunk_text="a cozy soup with creamy white beans",
        embed_text=_QUERY,
        provider=provider,
        span_ids=[span],
        structured_data=_RECIPE_STRUCTURED,
    )


async def test_hybrid_search_happy_path_envelope(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    item_id = await _seed_one_recipe(db_session, provider)
    async with _client(db_session) as client:
        resp = await client.post("/api/v1/search", json={"query": _QUERY, "mode": "hybrid"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query"] == _QUERY  # echoed normalized form
    assert "debug" not in body  # not requested
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["type"] == "knowledge_item_result"
    assert result["item"]["id"] == item_id
    assert result["item"]["schema"] == "recipe.v1"
    assert result["item"]["confidence"] == {"overall": 0.88}
    # structured_preview uses the reserved `schema`/`yield` JSON keys.
    assert result["structured_preview"]["schema"] == "recipe.preview.v1"
    assert result["structured_preview"]["yield"] == "Serves 4"
    assert "white beans" in result["structured_preview"]["top_ingredients"]
    # display badges from yield + a time field; subtitle from doc title + citation.
    assert "Serves 4" in result["display"]["badges"]
    assert "35 minutes" in result["display"]["badges"]
    assert result["display"]["subtitle"] == "Simple Thai Food · page 42"
    # citations carry the Epic-12 label and the fetched locator.
    citation = result["source_citations"][0]
    assert citation["label"] == "page 42"
    assert citation["locator"]["page_start"] == 42
    assert result["matched_chunks"][0]["score"] > 0


async def test_keyword_and_vector_modes_can_order_differently(
    db_session: AsyncSession,
) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    doc = await _make_document(db_session)
    span = await _make_span(db_session, doc, page=1)
    # A: keyword-dense text, embedding far from the query.
    kw_item = await _make_recipe(
        db_session,
        doc,
        title="Keyword Winner",
        chunk_text="white beans soup white beans soup white beans soup",
        embed_text="completely unrelated machinery",
        provider=provider,
        span_ids=[span],
    )
    # B: sparse keyword text, embedding == the query vector.
    vec_item = await _make_recipe(
        db_session,
        doc,
        title="Vector Winner",
        chunk_text="white beans",
        embed_text=_QUERY,
        provider=provider,
        span_ids=[span],
    )

    async with _client(db_session) as client:
        kw = (await client.post("/api/v1/search", json={"query": _QUERY, "mode": "keyword"})).json()
        vec = (await client.post("/api/v1/search", json={"query": _QUERY, "mode": "vector"})).json()
    # Keyword ranks the dense item first; vector ranks the on-vector item first.
    assert kw["results"][0]["item"]["id"] == kw_item
    assert vec["results"][0]["item"]["id"] == vec_item


async def test_debug_gated_by_both_flags(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    await _seed_one_recipe(db_session, provider)

    # include_debug=true AND debug_endpoints_enabled=true → debug present.
    async with _client(db_session, debug_enabled=True) as client:
        body = (
            await client.post(
                "/api/v1/search",
                json={"query": _QUERY, "mode": "hybrid", "include_debug": True},
            )
        ).json()
    assert "debug" in body
    assert body["debug"]["retrieval_mode"] == "hybrid"
    assert body["debug"]["embedding_model"] == _FAKE_MODEL

    # include_debug=true but debug_endpoints_enabled=false → debug absent.
    async with _client(db_session, debug_enabled=False) as client:
        body = (
            await client.post(
                "/api/v1/search",
                json={"query": _QUERY, "mode": "hybrid", "include_debug": True},
            )
        ).json()
    assert "debug" not in body


async def test_empty_query_and_invalid_mode_return_400(db_session: AsyncSession) -> None:
    async with _client(db_session) as client:
        empty = await client.post("/api/v1/search", json={"query": "   "})
        bad_mode = await client.post("/api/v1/search", json={"query": _QUERY, "mode": "bogus"})
    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "invalid_request"
    assert bad_mode.status_code == 400
    assert bad_mode.json()["error"]["code"] == "invalid_request"
    assert bad_mode.json()["error"]["details"]["field"] == "mode"


async def test_missing_token_returns_401(db_session: AsyncSession) -> None:
    async with _client(db_session, with_auth=False) as client:
        resp = await client.post("/api/v1/search", json={"query": _QUERY})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
