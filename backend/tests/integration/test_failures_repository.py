from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.repositories.failures import FailuresRepository


async def _make_document(session: AsyncSession, *, content_hash: str) -> str:
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
        status=DocumentStatus.QUEUED,
    )
    return document.id


@pytest.mark.asyncio
async def test_add_failure_round_trip(db_session: AsyncSession) -> None:
    document_id = await _make_document(db_session, content_hash="failures-hash-1")
    failures = FailuresRepository(db_session)

    failure = await failures.add_failure(
        document_id=document_id,
        last_status=DocumentStatus.EXTRACTING_TEXT,
        reason="stuck_job_timeout",
        error_message="hit timeout",
        metadata_json={"timeout_minutes": 30},
    )

    assert failure.id.startswith("fail_")
    assert failure.document_id == document_id
    assert failure.last_status == DocumentStatus.EXTRACTING_TEXT
    assert failure.reason == "stuck_job_timeout"
    assert failure.error_message == "hit timeout"
    assert failure.metadata_json == {"timeout_minutes": 30}
    assert failure.failed_at is not None


@pytest.mark.asyncio
async def test_add_failure_defaults_metadata_to_empty_dict(
    db_session: AsyncSession,
) -> None:
    document_id = await _make_document(db_session, content_hash="failures-hash-default")
    failures = FailuresRepository(db_session)

    failure = await failures.add_failure(
        document_id=document_id,
        last_status=DocumentStatus.EXTRACTING_TEXT,
        reason="stuck_job_timeout",
    )

    assert failure.error_message is None
    assert failure.metadata_json == {}


@pytest.mark.asyncio
async def test_list_failures_orders_latest_first(db_session: AsyncSession) -> None:
    document_id = await _make_document(db_session, content_hash="failures-hash-order")
    failures = FailuresRepository(db_session)

    inserted: list[str] = []
    for label in ("first", "second", "third"):
        row = await failures.add_failure(
            document_id=document_id,
            last_status=DocumentStatus.EXTRACTING_TEXT,
            reason=label,
        )
        inserted.append(row.id)
        await asyncio.sleep(0.01)

    listed = await failures.list_failures(document_id, limit=10)
    assert [row.reason for row in listed] == ["third", "second", "first"]
    assert [row.id for row in listed] == list(reversed(inserted))


@pytest.mark.asyncio
async def test_list_failures_respects_limit(db_session: AsyncSession) -> None:
    document_id = await _make_document(db_session, content_hash="failures-hash-limit")
    failures = FailuresRepository(db_session)

    for label in ("a", "b", "c", "d", "e"):
        await failures.add_failure(
            document_id=document_id,
            last_status=DocumentStatus.EXTRACTING_TEXT,
            reason=label,
        )
        await asyncio.sleep(0.01)

    listed = await failures.list_failures(document_id, limit=2)
    assert len(listed) == 2
    assert [row.reason for row in listed] == ["e", "d"]


@pytest.mark.asyncio
async def test_list_failures_returns_empty_for_unknown_document(
    db_session: AsyncSession,
) -> None:
    failures = FailuresRepository(db_session)
    listed = await failures.list_failures("doc_does_not_exist", limit=10)
    assert listed == []
