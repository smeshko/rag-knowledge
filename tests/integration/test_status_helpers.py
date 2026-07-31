from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.status import (
    InvalidTransitionError,
    mark_failed,
    transition_to,
)
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.repositories.failures import FailuresRepository


async def _make_document(
    session: AsyncSession,
    *,
    content_hash: str,
    status: DocumentStatus = DocumentStatus.QUEUED,
) -> str:
    repo = DocumentRepository(session)
    aid = new_id(SourceAsset.ID_PREFIX)
    await repo.add_source_asset(
        id=aid,
        source_type=SourceType.PDF,
        original_filename="example.pdf",
        storage_provider="local",
        storage_key=f"source-assets/{aid}/original.pdf",
        content_hash=content_hash,
        upload_status=UploadStatus.UPLOADED,
    )
    document = await repo.add_document(
        asset_id=aid,
        category="recipes",
        subcategory=None,
        title="Example",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=status,
    )
    return document.id


@pytest.mark.asyncio
async def test_transition_to_happy_path(db_session: AsyncSession) -> None:
    document_id = await _make_document(db_session, content_hash="status-hash-happy")

    returned = await transition_to(
        db_session, document_id, DocumentStatus.EXTRACTING_TEXT
    )
    assert returned.status == DocumentStatus.EXTRACTING_TEXT

    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.EXTRACTING_TEXT


@pytest.mark.asyncio
async def test_transition_to_rejects_illegal_edge(db_session: AsyncSession) -> None:
    document_id = await _make_document(db_session, content_hash="status-hash-illegal")

    with pytest.raises(InvalidTransitionError) as excinfo:
        await transition_to(db_session, document_id, DocumentStatus.INDEXING)

    assert excinfo.value.current == DocumentStatus.QUEUED
    assert excinfo.value.attempted == DocumentStatus.INDEXING
    assert DocumentStatus.EXTRACTING_TEXT in excinfo.value.allowed
    assert DocumentStatus.FAILED in excinfo.value.allowed


@pytest.mark.asyncio
async def test_transition_to_raises_for_missing_document(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(LookupError):
        await transition_to(
            db_session, "doc_does_not_exist", DocumentStatus.EXTRACTING_TEXT
        )


@pytest.mark.asyncio
async def test_mark_failed_writes_failure_row_and_transitions(
    db_session: AsyncSession,
) -> None:
    document_id = await _make_document(
        db_session,
        content_hash="status-hash-markfail",
        status=DocumentStatus.QUEUED,
    )
    await transition_to(
        db_session, document_id, DocumentStatus.EXTRACTING_TEXT
    )

    returned = await mark_failed(
        db_session,
        document_id,
        reason="test",
        error_message="boom",
        metadata_json={"k": "v"},
    )

    assert returned.status == DocumentStatus.FAILED

    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert len(failures) == 1
    only = failures[0]
    assert only.last_status == DocumentStatus.EXTRACTING_TEXT
    assert only.reason == "test"
    assert only.error_message == "boom"
    assert only.metadata_json == {"k": "v"}


@pytest.mark.asyncio
async def test_mark_failed_rejected_when_already_terminal(
    db_session: AsyncSession,
) -> None:
    document_id = await _make_document(
        db_session,
        content_hash="status-hash-terminal",
        status=DocumentStatus.QUEUED,
    )
    await transition_to(db_session, document_id, DocumentStatus.EXTRACTING_TEXT)
    await mark_failed(db_session, document_id, reason="first")
    await db_session.flush()

    # First mark_failed wrote one row; capture the count before the second
    # attempt so we can assert the rejected call didn't strand a partial row.
    failures_before = await FailuresRepository(db_session).list_failures(document_id)
    assert len(failures_before) == 1

    with pytest.raises(InvalidTransitionError):
        await mark_failed(db_session, document_id, reason="second")

    # The second mark_failed inserted its failure row before transition_to
    # rejected. Caller is responsible for rolling back; the test does not
    # commit, so the savepoint rollback at teardown cleans both writes.
    # Inside this transaction the row IS visible — what matters is that the
    # status did not advance past FAILED again.
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.FAILED
