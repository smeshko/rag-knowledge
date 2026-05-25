"""Integration tests for initial-schema semantics.

Exercises the full SourceAsset -> Document -> SourceSpan -> ExtractionRun ->
KnowledgeItem -> Chunk -> ChunkEmbedding chain through the real ORM models, and
confirms Postgres rejects each early uniqueness constraint and a representative
foreign-key violation at flush time.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.models import (
    Chunk,
    ChunkEmbedding,
    Document,
    ExtractionRun,
    KnowledgeItem,
    SourceAsset,
    SourceSpan,
)


async def _build_chain_through_chunk(
    session: AsyncSession,
) -> tuple[SourceAsset, Document, SourceSpan, ExtractionRun, KnowledgeItem, Chunk]:
    """Insert one row of each model down to Chunk; flush after each step."""
    asset = SourceAsset(
        source_type="pdf",
        original_filename="x.pdf",
        storage_provider="local",
        storage_key="source-assets/x/original.pdf",
        content_hash="hash_x",
        upload_status="uploaded",
    )
    session.add(asset)
    await session.flush()

    document = Document(
        asset_id=asset.id,
        category="recipes",
        title="t",
        author="a",
        source_type="pdf",
        status="ready",
    )
    session.add(document)
    await session.flush()

    span = SourceSpan(
        document_id=document.id,
        source_version=1,
        source_type="pdf",
        locator={"type": "pdf_page_range", "page_start": 1, "page_end": 1},
        locator_hash="loc_hash",
        text="hello",
        text_hash="text_hash",
    )
    session.add(span)
    await session.flush()

    run = ExtractionRun(
        document_id=document.id,
        source_version=1,
        provider="openai",
        model="m",
        prompt_version="v1",
        schema_version="recipe.v1",
        input_source_span_ids=[span.id],
        input_hash="in_hash",
        status="success",
        output_json={"items": []},
    )
    session.add(run)
    await session.flush()

    item = KnowledgeItem(
        document_id=document.id,
        extraction_run_id=run.id,
        source_version=1,
        item_type="recipe",
        title="T",
        normalized_title="t",
        body_text="...",
        source_span_ids=[span.id],
        structured_data={"schema": "recipe.v1"},
        status="ready",
    )
    session.add(item)
    await session.flush()

    chunk = Chunk(
        document_id=document.id,
        parent_type="knowledge_item",
        parent_id=item.id,
        chunk_type="recipe_full",
        text="...",
        text_hash="c_hash",
        source_span_ids=[span.id],
        chunk_metadata={"category": "recipes"},
    )
    session.add(chunk)
    await session.flush()

    return asset, document, span, run, item, chunk


async def _document_with_asset(session: AsyncSession, suffix: str) -> Document:
    """Insert a SourceAsset + Document pair; return the flushed Document."""
    asset = SourceAsset(
        source_type="pdf",
        original_filename="x.pdf",
        storage_provider="local",
        storage_key=f"source-assets/{suffix}/original.pdf",
        content_hash=f"hash_{suffix}",
        upload_status="uploaded",
    )
    session.add(asset)
    await session.flush()
    document = Document(
        asset_id=asset.id,
        category="recipes",
        title="t",
        author="a",
        source_type="pdf",
        status="ready",
    )
    session.add(document)
    await session.flush()
    return document


async def test_full_fk_chain_persists(db_session: AsyncSession) -> None:
    asset, document, span, run, item, chunk = await _build_chain_through_chunk(db_session)

    embedding = ChunkEmbedding(
        chunk_id=chunk.id,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        embedding_vector=[0.0] * 1536,
    )
    db_session.add(embedding)
    await db_session.flush()

    await db_session.refresh(document, ["asset", "source_spans", "extraction_runs"])
    await db_session.refresh(item, ["document", "extraction_run"])
    await db_session.refresh(chunk, ["knowledge_item", "embeddings"])

    assert document.asset.id == asset.id
    assert len(document.source_spans) == 1
    assert document.source_spans[0].id == span.id
    assert len(document.extraction_runs) == 1
    assert document.extraction_runs[0].id == run.id
    assert item.document.id == document.id
    assert item.extraction_run.id == run.id
    assert chunk.knowledge_item.id == item.id
    assert chunk.document_id == document.id
    assert len(chunk.embeddings) == 1
    assert chunk.embeddings[0].embedding_dimensions == 1536
    assert len(chunk.embeddings[0].embedding_vector) == 1536
    assert chunk.embeddings[0].embedding_vector[0] == 0.0


async def test_chunk_embedding_uniqueness_enforced(db_session: AsyncSession) -> None:
    _, _, _, _, _, chunk = await _build_chain_through_chunk(db_session)
    kwargs = {
        "chunk_id": chunk.id,
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "embedding_vector": [0.0] * 1536,
    }
    db_session.add(ChunkEmbedding(**kwargs))
    await db_session.flush()
    db_session.add(ChunkEmbedding(**kwargs))
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


async def test_source_assets_content_hash_unique(db_session: AsyncSession) -> None:
    def _asset() -> SourceAsset:
        return SourceAsset(
            source_type="pdf",
            original_filename="x.pdf",
            storage_provider="local",
            storage_key="source-assets/x/original.pdf",
            content_hash="dup_hash",
            upload_status="uploaded",
        )

    db_session.add(_asset())
    await db_session.flush()
    db_session.add(_asset())
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


async def test_documents_asset_id_unique(db_session: AsyncSession) -> None:
    asset = SourceAsset(
        source_type="pdf",
        original_filename="x.pdf",
        storage_provider="local",
        storage_key="source-assets/x/original.pdf",
        content_hash="hash_doc_unique",
        upload_status="uploaded",
    )
    db_session.add(asset)
    await db_session.flush()

    def _document() -> Document:
        return Document(
            asset_id=asset.id,
            category="recipes",
            title="t",
            author="a",
            source_type="pdf",
            status="ready",
        )

    db_session.add(_document())
    await db_session.flush()
    db_session.add(_document())
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


async def test_source_spans_composite_unique(db_session: AsyncSession) -> None:
    asset = SourceAsset(
        source_type="pdf",
        original_filename="x.pdf",
        storage_provider="local",
        storage_key="source-assets/x/original.pdf",
        content_hash="hash_span_unique",
        upload_status="uploaded",
    )
    db_session.add(asset)
    await db_session.flush()
    document = Document(
        asset_id=asset.id,
        category="recipes",
        title="t",
        author="a",
        source_type="pdf",
        status="ready",
    )
    db_session.add(document)
    await db_session.flush()

    def _span() -> SourceSpan:
        return SourceSpan(
            document_id=document.id,
            source_version=1,
            source_type="pdf",
            locator={"type": "pdf_page_range", "page_start": 1, "page_end": 1},
            locator_hash="same_loc",
            text="hello",
            text_hash="text_hash",
        )

    db_session.add(_span())
    await db_session.flush()
    db_session.add(_span())
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


async def test_fk_violation_on_chunk_without_knowledge_item(db_session: AsyncSession) -> None:
    asset = SourceAsset(
        source_type="pdf",
        original_filename="x.pdf",
        storage_provider="local",
        storage_key="source-assets/x/original.pdf",
        content_hash="hash_fk_violation",
        upload_status="uploaded",
    )
    db_session.add(asset)
    await db_session.flush()
    document = Document(
        asset_id=asset.id,
        category="recipes",
        title="t",
        author="a",
        source_type="pdf",
        status="ready",
    )
    db_session.add(document)
    await db_session.flush()

    chunk = Chunk(
        document_id=document.id,
        parent_type="knowledge_item",
        parent_id="item_nonexistent",
        chunk_type="recipe_full",
        text="...",
        text_hash="c_hash",
        source_span_ids=[],
        chunk_metadata={},
    )
    db_session.add(chunk)
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


async def test_knowledge_item_document_id_mismatch_rejected(db_session: AsyncSession) -> None:
    """A raw-ID write where a KnowledgeItem's document_id disagrees with its
    ExtractionRun's document_id is rejected by the composite FK, even though the
    @validates relationship check never fires (no relationship is assigned)."""
    doc_a = await _document_with_asset(db_session, "ki_mismatch_a")
    doc_b = await _document_with_asset(db_session, "ki_mismatch_b")

    run = ExtractionRun(
        document_id=doc_a.id,
        source_version=1,
        provider="openai",
        model="m",
        prompt_version="v1",
        schema_version="recipe.v1",
        input_source_span_ids=[],
        input_hash="in_hash",
        status="success",
        output_json={"items": []},
    )
    db_session.add(run)
    await db_session.flush()

    item = KnowledgeItem(
        document_id=doc_b.id,  # disagrees with run.document_id (doc_a)
        extraction_run_id=run.id,
        source_version=1,
        item_type="recipe",
        title="T",
        normalized_title="t",
        body_text="...",
        source_span_ids=[],
        structured_data={"schema": "recipe.v1"},
        status="ready",
    )
    db_session.add(item)
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


async def test_chunk_document_id_mismatch_rejected(db_session: AsyncSession) -> None:
    """A raw-ID write where a Chunk's document_id disagrees with its parent
    KnowledgeItem's document_id is rejected by the composite FK."""
    doc_a = await _document_with_asset(db_session, "chunk_mismatch_a")
    doc_b = await _document_with_asset(db_session, "chunk_mismatch_b")

    run = ExtractionRun(
        document_id=doc_a.id,
        source_version=1,
        provider="openai",
        model="m",
        prompt_version="v1",
        schema_version="recipe.v1",
        input_source_span_ids=[],
        input_hash="in_hash",
        status="success",
        output_json={"items": []},
    )
    db_session.add(run)
    await db_session.flush()

    item = KnowledgeItem(
        document_id=doc_a.id,
        extraction_run_id=run.id,
        source_version=1,
        item_type="recipe",
        title="T",
        normalized_title="t",
        body_text="...",
        source_span_ids=[],
        structured_data={"schema": "recipe.v1"},
        status="ready",
    )
    db_session.add(item)
    await db_session.flush()

    chunk = Chunk(
        document_id=doc_b.id,  # disagrees with item.document_id (doc_a)
        parent_type="knowledge_item",
        parent_id=item.id,
        chunk_type="recipe_full",
        text="...",
        text_hash="c_hash",
        source_span_ids=[],
        chunk_metadata={},
    )
    db_session.add(chunk)
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()
