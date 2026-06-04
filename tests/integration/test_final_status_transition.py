"""Integration test for the Phase 10.3 terminal-status transition.

Seeds a document at ``EMBEDDING_CHUNKS`` (with chunks for the ``>= 1`` case) and
drives ``_index_and_finalize``: it lands ``READY`` when the document has at least
one chunk and ``NEEDS_REVIEW`` when it has none. The predicate is the document's
chunk count (DECISIONS #3).
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.jobs import _index_and_finalize
from rag_recipes.ingestion.pipeline.chunking import persist_chunks_for_ready_items
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio

SOURCE_VERSION = 1
CATEGORY = "recipes"


def _single_session_factory(session: AsyncSession):  # type: ignore[no-untyped-def]
    """Yield the savepoint ``db_session`` to the stage and suppress its commit."""

    @asynccontextmanager
    async def _factory() -> AsyncIterator[AsyncSession]:
        original = session.commit
        session.commit = _noop  # type: ignore[method-assign]
        try:
            yield session
        finally:
            session.commit = original  # type: ignore[method-assign]

    async def _noop() -> None:
        return None

    return _factory


async def _seed_document(session: AsyncSession, *, with_chunks: bool) -> str:
    """Seed a document at EMBEDDING_CHUNKS, optionally with a ready item + chunks."""
    repo = DocumentRepository(session)
    pdf_bytes = b"%PDF-1.4 finalize fixture"
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
        status=DocumentStatus.EMBEDDING_CHUNKS,
    )
    if with_chunks:
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
        item = KnowledgeItem(
            document_id=document.id,
            extraction_run_id=run.id,
            source_version=SOURCE_VERSION,
            item_type="recipe",
            title="Tomato and White Bean Soup",
            normalized_title="tomato and white bean soup",
            summary="A hearty weeknight soup.",
            body_text="Tomato and White Bean Soup\n\nA hearty soup.",
            source_span_ids=["span_1"],
            structured_data={
                "ingredients_text": "2 tbsp olive oil",
                "ingredients": [{"raw_text": "2 tbsp olive oil"}],
                "steps_text": "1. Heat the oil.",
                "steps": [{"text": "Heat the oil."}],
            },
            status=KnowledgeItemStatus.READY,
        )
        session.add(item)
        await session.flush()
        await persist_chunks_for_ready_items(
            session, document_id=document.id, category=CATEGORY
        )
    await session.flush()
    return document.id


async def test_index_and_finalize_ready_with_chunks(db_session: AsyncSession) -> None:
    document_id = await _seed_document(db_session, with_chunks=True)
    chunk_count = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )
    assert chunk_count == 5

    terminal = await _index_and_finalize(
        _single_session_factory(db_session), document_id=document_id
    )

    assert terminal == DocumentStatus.READY
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.READY


async def test_index_and_finalize_needs_review_without_chunks(
    db_session: AsyncSession,
) -> None:
    document_id = await _seed_document(db_session, with_chunks=False)
    chunk_count = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )
    assert chunk_count == 0

    terminal = await _index_and_finalize(
        _single_session_factory(db_session), document_id=document_id
    )

    assert terminal == DocumentStatus.NEEDS_REVIEW
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.NEEDS_REVIEW
