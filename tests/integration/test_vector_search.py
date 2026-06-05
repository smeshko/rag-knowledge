"""Integration tests for retrieval.vector.vector_search (doc 7 § 5).

Real pgvector via ``db_session`` with a ``FakeEmbeddingProvider`` (deterministic,
provider/model-seeded vectors — no paid API). Epic 10.2 (the embedding pipeline)
and 10.3 (the HNSW index) are merged, but to stay self-contained these tests seed
``chunk_embeddings`` rows directly. Correctness holds with or without the HNSW
index (pgvector computes exact cosine distance on a sequential scan); the
``EXPLAIN``/index-usage assertion is intentionally not made here (flaky on a tiny
fixture).
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.providers._observability import TraceContext
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.embeddings.types import Embedding
from rag_recipes.retrieval.filters import build_filters
from rag_recipes.retrieval.normalize import normalize_query
from rag_recipes.retrieval.types import SearchRequest
from rag_recipes.retrieval.vector import vector_search
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
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio

_PROVIDER = "fake"
_MODEL_A = "model-a"
_MODEL_B = "model-b"


async def _make_document(
    session: AsyncSession,
    *,
    category: str = "recipes",
    subcategory: str | None = None,
    active_source_version: int | None = 1,
) -> str:
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
    document = await repo.add_document(
        asset_id=asset.id,
        category=category,
        subcategory=subcategory,
        title="A Cookbook",
        author="Chef",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=active_source_version,
        status=DocumentStatus.READY,
    )
    return document.id


async def _make_ready_chunk(
    session: AsyncSession,
    document_id: str,
    *,
    text: str,
    source_version: int = 1,
    status: KnowledgeItemStatus = KnowledgeItemStatus.READY,
) -> str:
    run = ExtractionRun(
        document_id=document_id,
        source_version=source_version,
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
        source_version=source_version,
        item_type="recipe",
        title=text[:40],
        normalized_title=text[:40].lower(),
        summary=None,
        body_text="x" * 100,
        source_span_ids=[],
        structured_data={"warnings": []},
        confidence=None,
        status=status,
    )
    session.add(item)
    await session.flush()
    chunk = Chunk(
        document_id=document_id,
        parent_type=ChunkParentType.KNOWLEDGE_ITEM,
        parent_id=item.id,
        chunk_type=ChunkType.RECIPE_FULL,
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        source_span_ids=[],
    )
    session.add(chunk)
    await session.flush()
    return chunk.id


async def _embed_chunk(
    session: AsyncSession,
    chunk_id: str,
    *,
    provider: FakeEmbeddingProvider,
    model: str,
    text: str,
) -> None:
    """Seed a ChunkEmbedding for ``chunk_id`` from ``provider.embed_text(text)``."""
    emb = await provider.embed_text(text)
    session.add(
        ChunkEmbedding(
            chunk_id=chunk_id,
            embedding_provider=_PROVIDER,
            embedding_model=model,
            embedding_dimensions=emb.dimensions,
            embedding_vector=emb.vector,
        )
    )
    await session.flush()


async def test_ranks_by_cosine_distance_closest_first(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_PROVIDER, model=_MODEL_A)
    doc = await _make_document(db_session)

    near = await _make_ready_chunk(db_session, doc, text="cozy weeknight dinner soup")
    far = await _make_ready_chunk(db_session, doc, text="industrial machine lubricant")
    # Seed each chunk's embedding from its own text; the query matches `near` exactly.
    await _embed_chunk(
        db_session, near, provider=provider, model=_MODEL_A, text="cozy weeknight dinner soup"
    )
    await _embed_chunk(
        db_session, far, provider=provider, model=_MODEL_A, text="industrial machine lubricant"
    )

    results = await vector_search(
        db_session,
        normalize_query("cozy weeknight dinner soup"),
        build_filters(SearchRequest(query="cozy weeknight dinner soup")),
        provider=provider,
        embedding_provider=_PROVIDER,
        embedding_model=_MODEL_A,
        top_k=50,
    )

    assert [c.chunk_id for c in results] == [near, far]
    top = results[0]
    assert top.retrieval_source == "vector"
    assert top.rank == 1
    assert top.distance is not None and top.distance == pytest.approx(0.0, abs=1e-6)
    assert top.similarity is not None and top.similarity == pytest.approx(1.0, abs=1e-6)
    assert top.raw_score == pytest.approx(top.similarity)
    # ranks ascending by distance.
    assert [c.rank for c in results] == [1, 2]
    assert results[0].distance <= results[1].distance


async def test_filters_by_provider_and_model(db_session: AsyncSession) -> None:
    prov_a = FakeEmbeddingProvider(provider=_PROVIDER, model=_MODEL_A)
    prov_b = FakeEmbeddingProvider(provider=_PROVIDER, model=_MODEL_B)
    doc = await _make_document(db_session)

    chunk_a = await _make_ready_chunk(db_session, doc, text="alpha recipe")
    chunk_b = await _make_ready_chunk(db_session, doc, text="beta recipe")
    # chunk_a embedded under model-a; chunk_b under model-b. Same chunk could carry
    # both, but separate chunks keep the assertion crisp.
    await _embed_chunk(db_session, chunk_a, provider=prov_a, model=_MODEL_A, text="alpha recipe")
    await _embed_chunk(db_session, chunk_b, provider=prov_b, model=_MODEL_B, text="beta recipe")

    results = await vector_search(
        db_session,
        normalize_query("alpha recipe"),
        build_filters(SearchRequest(query="alpha recipe")),
        provider=prov_a,
        embedding_provider=_PROVIDER,
        embedding_model=_MODEL_A,
        top_k=50,
    )
    # Only the model-a embedding is in the queried space; model-b is filtered out.
    assert [c.chunk_id for c in results] == [chunk_a]


async def test_metadata_filters_and_top_k_apply(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_PROVIDER, model=_MODEL_A)
    doc = await _make_document(db_session, category="recipes")
    other = await _make_document(db_session, category="cookbooks")

    in_cat = await _make_ready_chunk(db_session, doc, text="white beans stew")
    out_cat = await _make_ready_chunk(db_session, other, text="white beans stew")
    # Excluded: a needs_review item in the right category.
    nr = await _make_ready_chunk(
        db_session, doc, text="draft white beans", status=KnowledgeItemStatus.NEEDS_REVIEW
    )
    for cid in (in_cat, out_cat, nr):
        await _embed_chunk(
            db_session, cid, provider=provider, model=_MODEL_A, text="white beans stew"
        )

    results = await vector_search(
        db_session,
        normalize_query("white beans stew"),
        build_filters(SearchRequest(query="white beans stew", category="recipes")),
        provider=provider,
        embedding_provider=_PROVIDER,
        embedding_model=_MODEL_A,
        top_k=50,
    )
    assert [c.chunk_id for c in results] == [in_cat]

    # top_k caps the returned list when more match.
    extra = await _make_ready_chunk(db_session, doc, text="more white beans")
    await _embed_chunk(
        db_session, extra, provider=provider, model=_MODEL_A, text="more white beans"
    )
    capped = await vector_search(
        db_session,
        normalize_query("white beans stew"),
        build_filters(SearchRequest(query="white beans stew", category="recipes")),
        provider=provider,
        embedding_provider=_PROVIDER,
        embedding_model=_MODEL_A,
        top_k=1,
    )
    assert len(capped) == 1
    assert capped[0].rank == 1


async def test_filter_labels_must_match_query_embedding_space(
    db_session: AsyncSession,
) -> None:
    """A filter targeting a different (provider, model) than the query embedding's
    own space is rejected — comparing across spaces would corrupt ranking (review #1).
    """
    # The provider produces model-a vectors, but the caller asks to filter model-b.
    provider = FakeEmbeddingProvider(provider=_PROVIDER, model=_MODEL_A)
    with pytest.raises(ValueError, match="does not match the query embedding"):
        await vector_search(
            db_session,
            normalize_query("alpha recipe"),
            build_filters(SearchRequest(query="alpha recipe")),
            provider=provider,
            embedding_provider=_PROVIDER,
            embedding_model=_MODEL_B,
            top_k=50,
        )


async def test_empty_query_returns_no_candidates_without_embedding(
    db_session: AsyncSession,
) -> None:
    class _RaisingProvider(FakeEmbeddingProvider):
        async def embed_text(
            self, text: str, *, trace_context: TraceContext | None = None
        ) -> Embedding:
            raise AssertionError("embed_text must not be called for an empty query")

    results = await vector_search(
        db_session,
        normalize_query("   "),  # normalizes to ""
        build_filters(SearchRequest(query="")),
        provider=_RaisingProvider(provider=_PROVIDER, model=_MODEL_A),
        embedding_provider=_PROVIDER,
        embedding_model=_MODEL_A,
        top_k=50,
    )
    assert results == []
