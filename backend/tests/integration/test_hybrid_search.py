"""End-to-end integration test for the retrieval facade ``search()`` (doc 7 §§ 6-11).

Real Postgres FTS + pgvector via ``db_session`` with a ``FakeEmbeddingProvider``.
Epic 10.2/10.3 are merged, so the keyword leg's ``ts_vector`` column and the vector
leg both run directly; ``chunk_embeddings`` are seeded here to keep the test
self-contained. ``Settings`` is overridden so ``embedding_provider``/``embedding_model``
match the fake's stamped ``(provider, model)`` (the 12.2 cross-model guard).
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.config import get_settings
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.retrieval.search import search
from rag_recipes.retrieval.types import SearchRequest
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

pytestmark = pytest.mark.asyncio

_FAKE_PROVIDER = "fake"
_FAKE_MODEL = "fake-embedding"
_QUERY = "cozy soup with white beans"


def _settings():
    return get_settings().model_copy(
        update={"embedding_provider": _FAKE_PROVIDER, "embedding_model": _FAKE_MODEL}
    )


async def _make_document(
    session: AsyncSession, *, category: str = "recipes", subcategory: str | None = None
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
    doc = await repo.add_document(
        asset_id=asset.id,
        category=category,
        subcategory=subcategory,
        title="A Cookbook",
        author="Chef",
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


async def _make_item(
    session: AsyncSession,
    document_id: str,
    *,
    title: str,
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
        summary="a cozy soup",
        body_text="x" * 100,
        source_span_ids=[],
        structured_data={"warnings": []},
        confidence=None,
        status=status,
    )
    session.add(item)
    await session.flush()
    return item.id


async def _make_chunk(
    session: AsyncSession,
    document_id: str,
    item_id: str,
    *,
    text: str,
    chunk_type: ChunkType,
    span_ids: list[str],
    embed_text: str,
    provider: FakeEmbeddingProvider,
) -> str:
    chunk = Chunk(
        document_id=document_id,
        parent_type=ChunkParentType.KNOWLEDGE_ITEM,
        parent_id=item_id,
        chunk_type=chunk_type,
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
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
    return chunk.id


async def _seed_corpus(
    session: AsyncSession, provider: FakeEmbeddingProvider
) -> dict[str, str]:
    """Seed a recipes cookbook (the target) and a cookbooks one (off-category)."""
    doc = await _make_document(session, category="recipes", subcategory="soups")
    span = await _make_span(session, doc, page=42)

    # The target item: a chunk whose text matches the keyword query AND whose
    # embedding is the query's own vector — so it surfaces in BOTH legs.
    target = await _make_item(session, doc, title="Cozy White Bean Soup")
    both = await _make_chunk(
        session,
        doc,
        target,
        text="a cozy soup with creamy white beans",
        chunk_type=ChunkType.RECIPE_FULL,
        span_ids=[span],
        embed_text=_QUERY,
        provider=provider,
    )
    # A second chunk type on the same item (title) so grouping has a supporting type.
    await _make_chunk(
        session,
        doc,
        target,
        text="cozy white bean soup",
        chunk_type=ChunkType.RECIPE_TITLE,
        span_ids=[span],
        embed_text="unrelated title vector",
        provider=provider,
    )

    # Excluded: a needs_review item with the same matching content.
    nr = await _make_item(
        session, doc, title="Draft Soup", status=KnowledgeItemStatus.NEEDS_REVIEW
    )
    await _make_chunk(
        session,
        doc,
        nr,
        text="a cozy soup with white beans draft",
        chunk_type=ChunkType.RECIPE_FULL,
        span_ids=[span],
        embed_text=_QUERY,
        provider=provider,
    )
    # Excluded: a superseded item.
    sup = await _make_item(
        session, doc, title="Old Soup", status=KnowledgeItemStatus.SUPERSEDED
    )
    await _make_chunk(
        session,
        doc,
        sup,
        text="a cozy soup with white beans old",
        chunk_type=ChunkType.RECIPE_FULL,
        span_ids=[span],
        embed_text=_QUERY,
        provider=provider,
    )
    return {"target_item": target, "both_chunk": both}


async def test_hybrid_search_groups_items_with_both_signals(
    db_session: AsyncSession,
) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    seeded = await _seed_corpus(db_session, provider)

    result = await search(
        db_session,
        SearchRequest(query=_QUERY, mode="hybrid", category="recipes"),
        provider=provider,
        settings=_settings(),
    )

    # Exactly the one ready item; needs_review/superseded excluded.
    assert [r.item.knowledge_item_id for r in result.items] == [seeded["target_item"]]
    top = result.items[0]
    assert top.item.title == "Cozy White Bean Soup"
    assert top.item.status == "ready"
    assert top.document.author == "Chef"
    # The full chunk matched BOTH legs (keyword text + query embedding).
    both = next(m for m in top.matched_chunks if m.chunk_id == seeded["both_chunk"])
    assert both.score > 0
    # Both keyword and vector contributed to the run.
    assert result.debug.keyword_candidates > 0
    assert result.debug.vector_candidates > 0
    # Citation label from the seeded page-42 span.
    assert any(c.label == "page 42" for c in top.source_citations)


async def test_non_positive_limit_falls_back_to_default(
    db_session: AsyncSession,
) -> None:
    """A non-positive limit must not slice off ranked items (review #1)."""
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    seeded = await _seed_corpus(db_session, provider)
    # limit=-1 would otherwise slice grouped[:-1] and drop the only result.
    result = await search(
        db_session,
        SearchRequest(query=_QUERY, mode="hybrid", category="recipes", limit=-1),
        provider=provider,
        settings=_settings(),
    )
    assert [r.item.knowledge_item_id for r in result.items] == [seeded["target_item"]]


async def test_keyword_and_vector_modes_return_item_results(
    db_session: AsyncSession,
) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    seeded = await _seed_corpus(db_session, provider)

    kw = await search(
        db_session,
        SearchRequest(query=_QUERY, mode="keyword", category="recipes"),
        provider=provider,
        settings=_settings(),
    )
    assert [r.item.knowledge_item_id for r in kw.items] == [seeded["target_item"]]
    assert kw.debug.keyword_candidates > 0
    assert kw.debug.vector_candidates == 0

    vec = await search(
        db_session,
        SearchRequest(query=_QUERY, mode="vector", category="recipes"),
        provider=provider,
        settings=_settings(),
    )
    assert [r.item.knowledge_item_id for r in vec.items] == [seeded["target_item"]]
    assert vec.debug.vector_candidates > 0
    assert vec.debug.keyword_candidates == 0
