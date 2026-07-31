"""Integration tests for retrieval.keyword.keyword_search (doc 7 § 4).

Real Postgres FTS via ``db_session``. The ``chunks.ts_vector`` generated column and
its GIN index ship in the Epic 10.3 migration (``77b2b79ace95``), so the query runs
directly against the migrated schema — no stopgap DDL needed.

Seeds two documents in different categories/subcategories, each with a READY item +
chunks, plus the exclusion cases (needs_review, superseded, a stale-source_version
item) so every filter/allowlist path is exercised.
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.retrieval.filters import build_filters
from rag_recipes.retrieval.keyword import keyword_search
from rag_recipes.retrieval.normalize import normalize_query
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
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio


async def _make_document(
    session: AsyncSession,
    *,
    category: str,
    subcategory: str | None,
    active_source_version: int | None,
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


async def _make_run(
    session: AsyncSession, document_id: str, *, source_version: int
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
    return run.id


async def _make_item(
    session: AsyncSession,
    document_id: str,
    run_id: str,
    *,
    title: str,
    status: KnowledgeItemStatus,
    source_version: int,
    item_type: str = "recipe",
) -> str:
    item = KnowledgeItem(
        document_id=document_id,
        extraction_run_id=run_id,
        source_version=source_version,
        item_type=item_type,
        title=title,
        normalized_title=title.lower(),
        summary=None,
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
    chunk_type: ChunkType = ChunkType.RECIPE_FULL,
) -> str:
    chunk = Chunk(
        document_id=document_id,
        parent_type=ChunkParentType.KNOWLEDGE_ITEM,
        parent_id=item_id,
        chunk_type=chunk_type,
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        source_span_ids=[],
    )
    session.add(chunk)
    await session.flush()
    return chunk.id


async def _ready_item_with_chunk(
    session: AsyncSession,
    document_id: str,
    *,
    title: str,
    text: str,
    source_version: int = 1,
    item_type: str = "recipe",
    chunk_type: ChunkType = ChunkType.RECIPE_FULL,
) -> tuple[str, str]:
    run_id = await _make_run(session, document_id, source_version=source_version)
    item_id = await _make_item(
        session,
        document_id,
        run_id,
        title=title,
        status=KnowledgeItemStatus.READY,
        source_version=source_version,
        item_type=item_type,
    )
    chunk_id = await _make_chunk(
        session, document_id, item_id, text=text, chunk_type=chunk_type
    )
    return item_id, chunk_id


async def test_returns_only_ready_active_version_in_category(
    db_session: AsyncSession,
) -> None:
    # Document A — recipes/soups, active v1.
    doc_a = await _make_document(
        db_session, category="recipes", subcategory="soups", active_source_version=1
    )
    _, ready_chunk = await _ready_item_with_chunk(
        db_session, doc_a, title="White Bean Soup", text="hearty white beans soup"
    )

    # Excluded: needs_review item (same doc, same text).
    nr_run = await _make_run(db_session, doc_a, source_version=1)
    nr_item = await _make_item(
        db_session,
        doc_a,
        nr_run,
        title="Draft Beans",
        status=KnowledgeItemStatus.NEEDS_REVIEW,
        source_version=1,
    )
    await _make_chunk(db_session, doc_a, nr_item, text="white beans draft")

    # Excluded: superseded item.
    sup_run = await _make_run(db_session, doc_a, source_version=1)
    sup_item = await _make_item(
        db_session,
        doc_a,
        sup_run,
        title="Old Beans",
        status=KnowledgeItemStatus.SUPERSEDED,
        source_version=1,
    )
    await _make_chunk(db_session, doc_a, sup_item, text="white beans old")

    # Excluded: a READY item at a stale source_version (doc is active on v1).
    stale_run = await _make_run(db_session, doc_a, source_version=2)
    stale_item = await _make_item(
        db_session,
        doc_a,
        stale_run,
        title="Stale Beans",
        status=KnowledgeItemStatus.READY,
        source_version=2,
    )
    await _make_chunk(db_session, doc_a, stale_item, text="white beans stale")

    # Excluded: a READY item in a different category.
    doc_b = await _make_document(
        db_session, category="cookbooks", subcategory=None, active_source_version=1
    )
    await _ready_item_with_chunk(
        db_session, doc_b, title="Bean Stew", text="white beans stew"
    )

    results = await keyword_search(
        db_session,
        normalize_query("white beans"),
        build_filters(SearchRequest(query="white beans", category="recipes")),
        top_k=50,
    )

    assert [c.chunk_id for c in results] == [ready_chunk]
    candidate = results[0]
    assert candidate.retrieval_source == "keyword"
    assert candidate.rank == 1
    assert candidate.raw_score > 0
    assert candidate.distance is None and candidate.similarity is None
    assert candidate.chunk_type == ChunkType.RECIPE_FULL


async def test_category_and_subcategory_and_document_ids_filters(
    db_session: AsyncSession,
) -> None:
    doc_a = await _make_document(
        db_session, category="recipes", subcategory="soups", active_source_version=1
    )
    _, a_chunk = await _ready_item_with_chunk(
        db_session, doc_a, title="White Bean Soup", text="white beans soup"
    )
    doc_b = await _make_document(
        db_session, category="cookbooks", subcategory=None, active_source_version=1
    )
    _, b_chunk = await _ready_item_with_chunk(
        db_session, doc_b, title="Bean Stew", text="white beans stew"
    )

    async def _search(**kw: object) -> list[str]:
        res = await keyword_search(
            db_session,
            normalize_query("white beans"),
            build_filters(SearchRequest(query="white beans", **kw)),  # type: ignore[arg-type]
            top_k=50,
        )
        return [c.chunk_id for c in res]

    # category routes between the two cookbooks.
    assert await _search(category="recipes") == [a_chunk]
    assert await _search(category="cookbooks") == [b_chunk]
    # A non-null subcategory filter excludes the null-subcategory doc B.
    assert await _search(category="cookbooks", subcategory="soups") == []
    assert await _search(category="recipes", subcategory="soups") == [a_chunk]
    # document_ids allowlist restricts to the listed docs.
    assert await _search(category="recipes", document_ids=[doc_a]) == [a_chunk]
    assert await _search(category="cookbooks", document_ids=[doc_a]) == []


async def test_ranking_top_k_and_positional_rank(db_session: AsyncSession) -> None:
    doc = await _make_document(
        db_session, category="recipes", subcategory=None, active_source_version=1
    )
    # Three matching chunks; the most term-dense should rank first.
    _, c_dense = await _ready_item_with_chunk(
        db_session, doc, title="Beans Beans", text="white beans white beans white beans"
    )
    _, c_mid = await _ready_item_with_chunk(
        db_session, doc, title="Beans", text="white beans and rice"
    )
    _, c_low = await _ready_item_with_chunk(
        db_session, doc, title="Mentions", text="a stew that mentions white beans once"
    )

    results = await keyword_search(
        db_session,
        normalize_query("white beans"),
        build_filters(SearchRequest(query="white beans", category="recipes")),
        top_k=50,
    )
    assert len(results) == 3
    # ranks are 1-based, contiguous, and ordered by score desc.
    assert [c.rank for c in results] == [1, 2, 3]
    scores = [c.raw_score for c in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk_id == c_dense
    assert {c_dense, c_mid, c_low} == {c.chunk_id for c in results}

    # top_k smaller than the matching set caps the returned list.
    capped = await keyword_search(
        db_session,
        normalize_query("white beans"),
        build_filters(SearchRequest(query="white beans", category="recipes")),
        top_k=2,
    )
    assert len(capped) == 2
    assert [c.rank for c in capped] == [1, 2]


async def test_punctuation_in_keyword_still_matches_via_tsquery(
    db_session: AsyncSession,
) -> None:
    doc = await _make_document(
        db_session, category="recipes", subcategory=None, active_source_version=1
    )
    _, chunk = await _ready_item_with_chunk(
        db_session, doc, title="Beans", text="a bowl of white beans"
    )
    # normalize_query keeps the "!!"; plainto_tsquery reduces it to lexemes.
    nq = normalize_query("WHITE beans!!")
    assert nq.keyword == "white beans!!"
    results = await keyword_search(
        db_session,
        nq,
        build_filters(SearchRequest(query="white beans", category="recipes")),
        top_k=50,
    )
    assert [c.chunk_id for c in results] == [chunk]


async def test_query_with_sql_metacharacters_is_treated_as_text(
    db_session: AsyncSession,
) -> None:
    doc = await _make_document(
        db_session, category="recipes", subcategory=None, active_source_version=1
    )
    await _ready_item_with_chunk(
        db_session, doc, title="Beans", text="white beans recipe"
    )
    # A query carrying SQL metacharacters must not error or inject — it is bound text.
    nq = normalize_query("beans'); DROP TABLE chunks;--")
    results = await keyword_search(
        db_session,
        nq,
        build_filters(SearchRequest(query="x", category="recipes")),
        top_k=50,
    )
    # It matches the "beans" lexeme; the point is it ran safely as search text.
    assert all(c.retrieval_source == "keyword" for c in results)
    # The chunks table is intact (nothing was dropped/injected).
    surviving = await db_session.scalar(select(func.count()).select_from(Chunk))
    assert surviving is not None and surviving >= 1
