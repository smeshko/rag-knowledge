"""Integration tests for ingestion.pipeline.chunking.persist_chunks_for_ready_items.

Real Postgres (``test_engine`` / ``db_session``). Seeds one document carrying a
``READY`` item plus a ``NEEDS_REVIEW`` and a ``SUPERSEDED`` item, then proves the
helper persists chunks for the ready item only, satisfies the composite FK /
``document_id`` validator, and round-trips the ``metadata`` JSONB column.
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.chunking import persist_chunks_for_ready_items
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
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

SOURCE_VERSION = 1
CATEGORY = "recipes"


async def _seed_document_and_run(session: AsyncSession) -> tuple[str, str]:
    """Insert a Document + a SUCCESS ExtractionRun; return their ids."""
    repo = DocumentRepository(session)
    pdf_bytes = b"%PDF-1.4 chunking fixture"
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
        category=CATEGORY,
        subcategory=None,
        title="My Recipe Book",
        author="Alice",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.CREATING_CHUNKS,
    )
    run = ExtractionRun(
        document_id=document.id,
        source_version=SOURCE_VERSION,
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=["span_1"],
        input_hash=hashlib.sha256(b"window").hexdigest(),
        status=ExtractionRunStatus.SUCCESS,
        output_json=None,
    )
    session.add(run)
    await session.flush()
    return document.id, run.id


def _make_item(
    *,
    document_id: str,
    run_id: str,
    title: str,
    status: KnowledgeItemStatus,
) -> KnowledgeItem:
    return KnowledgeItem(
        document_id=document_id,
        extraction_run_id=run_id,
        source_version=SOURCE_VERSION,
        item_type="recipe",
        title=title,
        normalized_title=title.lower(),
        summary="A hearty weeknight soup.",
        body_text=f"{title}\n\nA hearty weeknight soup.",
        source_span_ids=["span_1"],
        structured_data={
            "ingredients_text": "2 tbsp olive oil\n1 onion",
            "ingredients": [{"raw_text": "2 tbsp olive oil"}],
            "steps_text": "1. Heat the oil.",
            "steps": [{"text": "Heat the oil."}],
        },
        status=status,
    )


async def test_persist_chunks_for_ready_items_only(db_session: AsyncSession) -> None:
    document_id, run_id = await _seed_document_and_run(db_session)
    ready = _make_item(
        document_id=document_id,
        run_id=run_id,
        title="Tomato and White Bean Soup",
        status=KnowledgeItemStatus.READY,
    )
    review = _make_item(
        document_id=document_id,
        run_id=run_id,
        title="Half-baked Recipe",
        status=KnowledgeItemStatus.NEEDS_REVIEW,
    )
    superseded = _make_item(
        document_id=document_id,
        run_id=run_id,
        title="Old Recipe",
        status=KnowledgeItemStatus.SUPERSEDED,
    )
    db_session.add_all([ready, review, superseded])
    await db_session.flush()

    count = await persist_chunks_for_ready_items(
        db_session, document_id=document_id, source_version=1, category=CATEGORY
    )
    # All five canonical types for the single ready item; none for the others.
    assert count == 5

    chunks = list(
        (
            await db_session.execute(
                select(Chunk).where(Chunk.document_id == document_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(chunks) == 5
    # Every chunk belongs to the ready item and agrees on document_id.
    assert {c.parent_id for c in chunks} == {ready.id}
    assert all(c.document_id == document_id for c in chunks)
    assert all(c.parent_type == ChunkParentType.KNOWLEDGE_ITEM for c in chunks)
    assert {c.chunk_type for c in chunks} == {
        ChunkType.RECIPE_TITLE,
        ChunkType.RECIPE_SUMMARY,
        ChunkType.RECIPE_INGREDIENTS,
        ChunkType.RECIPE_STEPS,
        ChunkType.RECIPE_FULL,
    }

    # The DB "metadata" column (mapped attribute chunk_metadata) round-trips.
    title_chunk = next(c for c in chunks if c.chunk_type == ChunkType.RECIPE_TITLE)
    assert title_chunk.chunk_metadata == {
        "category": CATEGORY,
        "item_type": "recipe",
        "title": "Tomato and White Bean Soup",
    }
    title_chunk_id = title_chunk.id

    # Re-read after a real commit to prove the JSONB write persisted. Capture the
    # id before expiring so the lookup arg is not itself a lazy load.
    await db_session.commit()
    db_session.expire_all()
    reloaded = await db_session.get(Chunk, title_chunk_id)
    assert reloaded is not None
    assert reloaded.chunk_metadata["category"] == CATEGORY


async def test_persist_chunks_no_ready_items_writes_nothing(
    db_session: AsyncSession,
) -> None:
    document_id, run_id = await _seed_document_and_run(db_session)
    review = _make_item(
        document_id=document_id,
        run_id=run_id,
        title="Half-baked Recipe",
        status=KnowledgeItemStatus.NEEDS_REVIEW,
    )
    db_session.add(review)
    await db_session.flush()

    count = await persist_chunks_for_ready_items(
        db_session, document_id=document_id, source_version=1, category=CATEGORY
    )
    assert count == 0
    total = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )
    assert total == 0
