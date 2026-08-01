"""Integration tests for DocumentRepository.delete_document_cascade (Phase 21.2).

Real Postgres (``db_session``). The repository method only flushes — the
transactional fixture's rollback isolation holds, so the plain ``db_session``
fixture is used (no savepoint-restart listener needed; that is the route
test's problem).

Seeding uses a local parameterized ``_seed_chain`` rather than the existing
chain builders: ``test_schema_integrity._build_chain_through_chunk`` and
``test_dedup_supersede._seed`` both pin a constant ``content_hash``, and
``source_assets.content_hash`` is UNIQUE — a second call raises
``IntegrityError``, so the two-document survival case (D1) is not
constructible with them. ``_seed_chain`` and ``assert_document_rows_gone``
are module-level on purpose: the route test (TASK-002) imports them from
here (settled in TASK-001's plan — import, don't duplicate).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.config import get_settings
from rag_recipes.ingestion.batch import ingest_batch_result
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.providers.llm.anthropic_batch import BatchResult
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionBatchItemStatus,
    ExtractionBatchStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_batch import ExtractionBatch
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.ingestion_failure import IngestionFailure
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.repositories.failures import FailuresRepository

pytestmark = pytest.mark.asyncio

SOURCE_VERSION = 1


@dataclass(frozen=True)
class SeededChain:
    """Ids of every row seeded for one document, for gone/intact assertions."""

    document_id: str
    asset_id: str
    storage_key: str
    span_ids: list[str]
    run_id: str
    item_ids: list[str]
    chunk_ids: list[str]
    embedding_ids: list[str]
    batch_id: str | None
    batch_item_id: str | None
    failure_id: str | None


async def _seed_chain(
    session: AsyncSession,
    suffix: str,
    *,
    status: DocumentStatus = DocumentStatus.READY,
    item_statuses: tuple[KnowledgeItemStatus, ...] = (KnowledgeItemStatus.READY,),
    normalized_title: str = "shared recipe",
    with_failure: bool = False,
    batch_item_status: ExtractionBatchItemStatus | None = None,
) -> SeededChain:
    """Seed a full asset→document→spans→run→items→chunks→embeddings chain.

    ``suffix`` varies ``content_hash`` / ``storage_key`` so two chains can
    coexist despite the UNIQUE ``source_assets.content_hash`` constraint.
    ``locator_hash`` needs no suffix — the span constraint is scoped per
    document. Each READY item gets one chunk with one 1536-dim embedding; all
    four unenforced span-id JSONB columns are populated so the survival test
    (D1) can sweep them.
    """
    storage_key = f"source-assets/{suffix}/original.pdf"
    asset = SourceAsset(
        source_type=SourceType.PDF,
        original_filename=f"{suffix}.pdf",
        storage_provider="local",
        storage_key=storage_key,
        content_hash=f"hash_{suffix}",
        upload_status=UploadStatus.UPLOADED,
    )
    session.add(asset)
    await session.flush()

    document = Document(
        asset_id=asset.id,
        category="recipes",
        title=f"Doc {suffix}",
        author="a",
        source_type=SourceType.PDF,
        active_source_version=SOURCE_VERSION,
        status=status,
    )
    session.add(document)
    await session.flush()

    spans: list[SourceSpan] = []
    for page in (1, 2):
        span = SourceSpan(
            document_id=document.id,
            source_version=SOURCE_VERSION,
            source_type=SourceType.PDF,
            locator={"type": "pdf_page_range", "page_start": page, "page_end": page},
            locator_hash=f"loc_{page}",
            text=f"page {page} text",
            text_hash=f"text_{suffix}_{page}",
        )
        session.add(span)
        spans.append(span)
    await session.flush()
    span_ids = [s.id for s in spans]

    run = ExtractionRun(
        document_id=document.id,
        source_version=SOURCE_VERSION,
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=span_ids,
        input_hash=f"in_hash_{suffix}",
        status=ExtractionRunStatus.SUCCESS,
        output_json={"items": []},
    )
    session.add(run)
    await session.flush()

    items: list[KnowledgeItem] = []
    for index, item_status in enumerate(item_statuses):
        item = KnowledgeItem(
            document_id=document.id,
            extraction_run_id=run.id,
            source_version=SOURCE_VERSION,
            item_type="recipe",
            title=f"Recipe {suffix} {index}",
            normalized_title=normalized_title,
            body_text="body " * 20,
            source_span_ids=span_ids,
            structured_data={"schema": "recipe.v1"},
            status=item_status,
        )
        session.add(item)
        items.append(item)
    await session.flush()

    chunks: list[Chunk] = []
    embeddings: list[ChunkEmbedding] = []
    for item in items:
        if item.status is not KnowledgeItemStatus.READY:
            continue
        chunk = Chunk(
            document_id=document.id,
            parent_type="knowledge_item",
            parent_id=item.id,
            chunk_type="recipe_full",
            text="chunk text",
            text_hash=f"c_hash_{suffix}_{item.id}",
            source_span_ids=span_ids,
            chunk_metadata={"category": "recipes"},
        )
        session.add(chunk)
        await session.flush()
        embedding = ChunkEmbedding(
            chunk_id=chunk.id,
            embedding_provider="fake",
            embedding_model="fake-embedding",
            embedding_dimensions=1536,
            embedding_vector=[0.0] * 1536,
        )
        session.add(embedding)
        chunks.append(chunk)
        embeddings.append(embedding)
    await session.flush()

    batch_id: str | None = None
    batch_item_id: str | None = None
    if batch_item_status is not None:
        batch = ExtractionBatch(
            provider="anthropic",
            provider_batch_id=f"msgbatch_{suffix}",
            model="claude-sonnet-4-6",
            processing_status=ExtractionBatchStatus.SUBMITTED,
            request_count=1,
        )
        session.add(batch)
        await session.flush()
        batch_item = ExtractionBatchItem(
            document_id=document.id,
            source_version=SOURCE_VERSION,
            input_hash=f"batch_hash_{suffix}",
            input_source_span_ids=span_ids,
            request_input="window",
            request_schema={"type": "object"},
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            status=batch_item_status,
            batch_id=batch.id,
        )
        session.add(batch_item)
        await session.flush()
        batch_id = batch.id
        batch_item_id = batch_item.id

    failure_id: str | None = None
    if with_failure:
        failure = await FailuresRepository(session).add_failure(
            document_id=document.id,
            last_status=DocumentStatus.EXTRACTING_ITEMS,
            reason="stuck_job_timeout",
            error_message="seeded failure",
        )
        failure_id = failure.id

    return SeededChain(
        document_id=document.id,
        asset_id=asset.id,
        storage_key=storage_key,
        span_ids=span_ids,
        run_id=run.id,
        item_ids=[item.id for item in items],
        chunk_ids=[chunk.id for chunk in chunks],
        embedding_ids=[embedding.id for embedding in embeddings],
        batch_id=batch_id,
        batch_item_id=batch_item_id,
        failure_id=failure_id,
    )


async def _count(session: AsyncSession, model: type, *where: object) -> int:
    result = await session.execute(
        select(func.count()).select_from(model).where(*where)  # type: ignore[arg-type]
    )
    return result.scalar_one()


async def assert_document_rows_gone(session: AsyncSession, chain: SeededChain) -> None:
    """Assert zero surviving rows for the chain across all nine affected tables."""
    session.expire_all()
    assert await _count(session, Document, Document.id == chain.document_id) == 0
    assert await _count(session, SourceAsset, SourceAsset.id == chain.asset_id) == 0
    assert (
        await _count(session, SourceSpan, SourceSpan.document_id == chain.document_id)
        == 0
    )
    assert (
        await _count(
            session, ExtractionRun, ExtractionRun.document_id == chain.document_id
        )
        == 0
    )
    assert (
        await _count(
            session, KnowledgeItem, KnowledgeItem.document_id == chain.document_id
        )
        == 0
    )
    assert await _count(session, Chunk, Chunk.document_id == chain.document_id) == 0
    if chain.embedding_ids:
        assert (
            await _count(
                session, ChunkEmbedding, ChunkEmbedding.id.in_(chain.embedding_ids)
            )
            == 0
        )
    assert (
        await _count(
            session,
            ExtractionBatchItem,
            ExtractionBatchItem.document_id == chain.document_id,
        )
        == 0
    )
    assert (
        await _count(
            session,
            IngestionFailure,
            IngestionFailure.document_id == chain.document_id,
        )
        == 0
    )


async def test_sole_document_delete_full_cascade(db_session: AsyncSession) -> None:
    """All nine tables emptied for the document: the eight explicit deletes plus
    ingestion_failures via the DB-level CASCADE (D5); superseded and extracting
    items die with the document (D1/D2)."""
    chain = await _seed_chain(
        db_session,
        "sole",
        item_statuses=(
            KnowledgeItemStatus.READY,
            KnowledgeItemStatus.SUPERSEDED,
            KnowledgeItemStatus.EXTRACTING,
        ),
        with_failure=True,
        batch_item_status=ExtractionBatchItemStatus.SUCCEEDED,
    )

    deletion = await DocumentRepository(db_session).delete_document_cascade(
        chain.document_id
    )

    assert deletion is not None
    await assert_document_rows_gone(db_session, chain)


async def test_delete_returns_storage_key_and_counts(db_session: AsyncSession) -> None:
    """The result carries the asset's storage_key (for the post-commit file
    delete) and the per-table deleted-row counts (the route's audit log; D5)."""
    chain = await _seed_chain(
        db_session,
        "counts",
        item_statuses=(KnowledgeItemStatus.READY, KnowledgeItemStatus.NEEDS_REVIEW),
        batch_item_status=ExtractionBatchItemStatus.SUCCEEDED,
    )

    deletion = await DocumentRepository(db_session).delete_document_cascade(
        chain.document_id
    )

    assert deletion is not None
    assert deletion.storage_key == chain.storage_key
    assert deletion.counts == {
        "chunk_embeddings": 1,
        "chunks": 1,
        "knowledge_items": 2,
        "extraction_runs": 1,
        "extraction_batch_items": 1,
        "source_spans": 2,
        "documents": 1,
        "source_assets": 1,
    }


async def test_two_document_dedup_case_survival(db_session: AsyncSession) -> None:
    """Deleting document A leaves document B — whose items share a
    normalized_title, the dedup grouping key — fully intact, and no surviving
    row's span-id JSONB column references a deleted span id (D1)."""
    chain_a = await _seed_chain(
        db_session, "dedup_a", batch_item_status=ExtractionBatchItemStatus.SUCCEEDED
    )
    chain_b = await _seed_chain(
        db_session, "dedup_b", batch_item_status=ExtractionBatchItemStatus.SUCCEEDED
    )

    deletion = await DocumentRepository(db_session).delete_document_cascade(
        chain_a.document_id
    )
    assert deletion is not None
    await assert_document_rows_gone(db_session, chain_a)

    # B's rows all survive.
    db_session.expire_all()
    assert await _count(db_session, Document, Document.id == chain_b.document_id) == 1
    assert await _count(db_session, SourceAsset, SourceAsset.id == chain_b.asset_id) == 1
    assert (
        await _count(db_session, SourceSpan, SourceSpan.id.in_(chain_b.span_ids)) == 2
    )
    assert (
        await _count(db_session, ExtractionRun, ExtractionRun.id == chain_b.run_id) == 1
    )
    assert (
        await _count(db_session, KnowledgeItem, KnowledgeItem.id.in_(chain_b.item_ids))
        == len(chain_b.item_ids)
    )
    assert await _count(db_session, Chunk, Chunk.id.in_(chain_b.chunk_ids)) == len(
        chain_b.chunk_ids
    )
    assert (
        await _count(
            db_session, ChunkEmbedding, ChunkEmbedding.id.in_(chain_b.embedding_ids)
        )
        == len(chain_b.embedding_ids)
    )
    assert (
        await _count(
            db_session, ExtractionBatchItem, ExtractionBatchItem.id == chain_b.batch_item_id
        )
        == 1
    )

    # No surviving row references a deleted span id, across all four
    # unenforced JSONB span-id columns (D1).
    deleted_span_ids = set(chain_a.span_ids)
    surviving_span_refs: list[list[str]] = []
    for column in (
        select(KnowledgeItem.source_span_ids),
        select(Chunk.source_span_ids),
        select(ExtractionRun.input_source_span_ids),
        select(ExtractionBatchItem.input_source_span_ids),
    ):
        result = await db_session.execute(column)
        surviving_span_refs.extend(result.scalars().all())
    for span_ids in surviving_span_refs:
        assert not deleted_span_ids.intersection(span_ids)


async def test_failed_document_with_submitted_batch_item(
    db_session: AsyncSession,
) -> None:
    """A failed document holding a submitted batch item: the item row is
    deleted, its ExtractionBatch survives with request_count unchanged, and a
    late ingest_batch_result for the vanished item no-ops (D4)."""
    chain = await _seed_chain(
        db_session,
        "batch_debris",
        status=DocumentStatus.FAILED,
        batch_item_status=ExtractionBatchItemStatus.SUBMITTED,
    )
    assert chain.batch_item_id is not None and chain.batch_id is not None
    batch_item_id = chain.batch_item_id

    deletion = await DocumentRepository(db_session).delete_document_cascade(
        chain.document_id
    )
    assert deletion is not None
    await assert_document_rows_gone(db_session, chain)

    batch = await db_session.get(ExtractionBatch, chain.batch_id)
    assert batch is not None
    assert batch.request_count == 1

    # A late provider result for the vanished item is a silent no-op: the
    # ingest guard re-fetches the row and returns when it is gone.
    db_session.expunge_all()
    vanished = ExtractionBatchItem(
        id=batch_item_id,
        document_id=chain.document_id,
        source_version=SOURCE_VERSION,
        input_hash="batch_hash_batch_debris",
        input_source_span_ids=[],
        request_input="window",
        request_schema={"type": "object"},
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        status=ExtractionBatchItemStatus.SUBMITTED,
    )
    await ingest_batch_result(
        db_session,
        vanished,
        BatchResult(custom_id=batch_item_id, result_type="succeeded", message=None),
        settings=get_settings(),
    )
    runs = await _count(
        db_session, ExtractionRun, ExtractionRun.document_id == chain.document_id
    )
    assert runs == 0


async def test_unknown_document_returns_none_and_deletes_nothing(
    db_session: AsyncSession,
) -> None:
    chain = await _seed_chain(db_session, "untouched")

    deletion = await DocumentRepository(db_session).delete_document_cascade(
        "doc_does_not_exist"
    )

    assert deletion is None
    db_session.expire_all()
    assert await _count(db_session, Document, Document.id == chain.document_id) == 1
    assert await _count(db_session, SourceAsset, SourceAsset.id == chain.asset_id) == 1
    assert (
        await _count(db_session, KnowledgeItem, KnowledgeItem.id.in_(chain.item_ids))
        == 1
    )
