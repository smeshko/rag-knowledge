from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.config import get_settings
from rag_recipes.ingestion.cron import sweep_stuck_jobs
from rag_recipes.ingestion.status import transition_to
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.repositories.failures import FailuresRepository

pytestmark = pytest.mark.asyncio


SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def _make_session_factory(session: AsyncSession) -> SessionFactory:
    """Wrap an existing session in a no-commit async context manager.

    The cron expects ``session_factory()`` to be an async context manager
    yielding an ``AsyncSession``. The test wants to reuse the rolled-back
    ``db_session`` fixture, so this factory yields the same session and
    suppresses the cron's ``await session.commit()`` (the conftest's
    outer transaction rollback owns cleanup).
    """

    @asynccontextmanager
    async def _factory() -> AsyncIterator[AsyncSession]:
        original = session.commit
        session.commit = lambda: _noop()  # type: ignore[method-assign]
        try:
            yield session
        finally:
            session.commit = original  # type: ignore[method-assign]

    async def _noop() -> None:
        return None

    return _factory


async def _backdate_updated_at(
    session: AsyncSession, document_id: str, *, minutes_ago: int
) -> None:
    """Bypass ``onupdate=func.now()`` by writing ``updated_at`` via raw SQL."""
    await session.execute(
        text(
            "UPDATE documents "
            "SET updated_at = now() - make_interval(mins => :minutes) "
            "WHERE id = :id"
        ),
        {"minutes": minutes_ago, "id": document_id},
    )


async def _set_last_progress_at(
    session: AsyncSession, document_id: str, *, minutes_ago: int
) -> None:
    """Set the extraction heartbeat explicitly (it has no server default)."""
    await session.execute(
        text(
            "UPDATE documents "
            "SET last_progress_at = now() - make_interval(mins => :minutes) "
            "WHERE id = :id"
        ),
        {"minutes": minutes_ago, "id": document_id},
    )


async def _make_document(
    session: AsyncSession,
    *,
    content_hash: str,
    status: DocumentStatus = DocumentStatus.EXTRACTING_TEXT,
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
        status=DocumentStatus.QUEUED,
    )
    if status is not DocumentStatus.QUEUED:
        # Walk the matrix to reach the requested non-terminal state via
        # transition_to so the stored history is realistic.
        path: list[DocumentStatus] = []
        if status in {
            DocumentStatus.EXTRACTING_TEXT,
            DocumentStatus.CREATING_SOURCE_SPANS,
            DocumentStatus.EXTRACTING_ITEMS,
            DocumentStatus.VALIDATING_ITEMS,
            DocumentStatus.CREATING_CHUNKS,
            DocumentStatus.EMBEDDING_CHUNKS,
            DocumentStatus.INDEXING,
            DocumentStatus.READY,
            DocumentStatus.NEEDS_REVIEW,
        }:
            ordered = [
                DocumentStatus.EXTRACTING_TEXT,
                DocumentStatus.CREATING_SOURCE_SPANS,
                DocumentStatus.EXTRACTING_ITEMS,
                DocumentStatus.VALIDATING_ITEMS,
                DocumentStatus.CREATING_CHUNKS,
                DocumentStatus.EMBEDDING_CHUNKS,
                DocumentStatus.INDEXING,
            ]
            for step in ordered:
                path.append(step)
                if step is status:
                    break
            if status in {DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW}:
                path.append(status)
        for next_status in path:
            await transition_to(session, document.id, next_status)
    return document.id


async def test_sweep_marks_stuck_document_failed_and_records_reason(
    db_session: AsyncSession,
) -> None:
    document_id = await _make_document(
        db_session,
        content_hash="sweep-stuck-1",
        status=DocumentStatus.EXTRACTING_TEXT,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert count == 1

    db_session.expire_all()
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.FAILED

    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert len(failures) == 1
    only = failures[0]
    assert only.reason == "stuck_job_timeout"
    assert only.last_status == DocumentStatus.EXTRACTING_TEXT
    assert only.metadata_json == {
        "timeout_minutes": get_settings().stuck_job_timeout_minutes,
        "last_seen_status": "extracting_text",
    }


async def test_sweep_skips_documents_within_timeout(
    db_session: AsyncSession,
) -> None:
    document_id = await _make_document(
        db_session,
        content_hash="sweep-fresh",
        status=DocumentStatus.EXTRACTING_TEXT,
    )
    # No backdate — updated_at is now().
    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert count == 0
    db_session.expire_all()
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.EXTRACTING_TEXT
    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert failures == []


async def test_sweep_skips_terminal_documents_even_if_old(
    db_session: AsyncSession,
) -> None:
    document_id = await _make_document(
        db_session,
        content_hash="sweep-terminal-ready",
        status=DocumentStatus.READY,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert count == 0
    db_session.expire_all()
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.READY
    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert failures == []


async def test_sweep_exempts_creating_source_spans_handoff_state(
    db_session: AsyncSession,
) -> None:
    # A document that finished Phase 8.1 rests at CREATING_SOURCE_SPANS with no
    # consumer to advance it until Epic 9. Even aged well past the timeout it
    # must NOT be swept to FAILED — a successful extraction is not "stuck".
    document_id = await _make_document(
        db_session,
        content_hash="sweep-handoff-css",
        status=DocumentStatus.CREATING_SOURCE_SPANS,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert count == 0
    db_session.expire_all()
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.CREATING_SOURCE_SPANS
    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert failures == []


async def test_sweep_reaps_embedding_chunks_once_indexing_consumes_it(
    db_session: AsyncSession,
) -> None:
    # Phase 10.3's indexing/terminal stage now consumes EMBEDDING_CHUNKS, so it is
    # no longer a resting state: a document wedged there past the timeout (crashed
    # after embedding, before the terminal transition) is genuinely stuck and must
    # be swept to FAILED so it is visible for reprocessing — not exempt forever.
    document_id = await _make_document(
        db_session,
        content_hash="sweep-stuck-embedding-chunks",
        status=DocumentStatus.EMBEDDING_CHUNKS,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert count == 1
    db_session.expire_all()
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.FAILED
    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert len(failures) == 1
    assert failures[0].last_status == DocumentStatus.EMBEDDING_CHUNKS


async def test_sweep_reaps_creating_chunks_once_embedding_consumes_it(
    db_session: AsyncSession,
) -> None:
    # Phase 10.2's embedding stage now consumes CREATING_CHUNKS, so it is no longer
    # a resting state: a document wedged there past the timeout (crashed after
    # chunking, before embedding) is genuinely stuck and must be swept to FAILED so
    # it is visible for reprocessing — not silently exempt forever.
    document_id = await _make_document(
        db_session,
        content_hash="sweep-stuck-creating-chunks",
        status=DocumentStatus.CREATING_CHUNKS,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert count == 1
    db_session.expire_all()
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.FAILED
    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert len(failures) == 1
    assert failures[0].reason == "stuck_job_timeout"
    assert failures[0].last_status == DocumentStatus.CREATING_CHUNKS


async def test_sweep_rechecks_heartbeat_under_lock_against_concurrent_progress(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Race (review round-3 #1): the stale SELECT snapshots a long-running
    # EXTRACTING_ITEMS document, but it commits a fresh extraction batch (advancing
    # last_progress_at) before the sweep takes the row lock. EXTRACTING_ITEMS ->
    # FAILED is legal, so without re-evaluating the heartbeat under the lock the
    # sweep would fail a still-healthy document. Simulate the heartbeat refresh by
    # hooking the sweep's first locking ``get``.
    document_id = await _make_document(
        db_session,
        content_hash="sweep-race-progress",
        status=DocumentStatus.EXTRACTING_ITEMS,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)

    real_get = db_session.get
    triggered = {"done": False}

    async def racing_get(entity: Any, ident: Any, **kwargs: Any) -> Any:
        if not triggered["done"] and ident == document_id:
            triggered["done"] = True
            # The worker commits another batch: a fresh heartbeat lands between the
            # sweep's stale SELECT and this row lock.
            await db_session.execute(
                text("UPDATE documents SET last_progress_at = now() WHERE id = :id"),
                {"id": document_id},
            )
        return await real_get(entity, ident, **kwargs)

    monkeypatch.setattr(db_session, "get", racing_get)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert triggered["done"]  # the race was actually injected
    assert count == 0
    db_session.expire_all()
    doc = await real_get(Document, document_id)
    assert doc is not None
    assert doc.status == DocumentStatus.EXTRACTING_ITEMS
    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert failures == []


async def test_sweep_skips_extracting_items_with_fresh_heartbeat(
    db_session: AsyncSession,
) -> None:
    # A healthy long extraction commits batches, advancing last_progress_at even
    # though updated_at is old. The progress-aware sweep must NOT reap it
    # (Phase 9.5; DECISIONS #6).
    document_id = await _make_document(
        db_session,
        content_hash="sweep-fresh-heartbeat",
        status=DocumentStatus.EXTRACTING_ITEMS,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)
    await _set_last_progress_at(db_session, document_id, minutes_ago=0)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert count == 0
    db_session.expire_all()
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.EXTRACTING_ITEMS
    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert failures == []


async def test_sweep_reaps_extracting_items_with_stale_heartbeat(
    db_session: AsyncSession,
) -> None:
    # A genuinely hung extraction stops advancing last_progress_at; once both the
    # heartbeat and updated_at are stale it ages out as before.
    document_id = await _make_document(
        db_session,
        content_hash="sweep-stale-heartbeat",
        status=DocumentStatus.EXTRACTING_ITEMS,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)
    await _set_last_progress_at(db_session, document_id, minutes_ago=120)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert count == 1
    db_session.expire_all()
    status = await db_session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    assert status == DocumentStatus.FAILED
    failures = await FailuresRepository(db_session).list_failures(document_id)
    assert len(failures) == 1
    assert failures[0].reason == "stuck_job_timeout"


async def test_sweep_continues_after_per_doc_error(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a_id = await _make_document(
        db_session,
        content_hash="sweep-skip-a",
        status=DocumentStatus.EXTRACTING_TEXT,
    )
    b_id = await _make_document(
        db_session,
        content_hash="sweep-skip-b",
        status=DocumentStatus.EXTRACTING_TEXT,
    )
    await _backdate_updated_at(db_session, a_id, minutes_ago=120)
    await _backdate_updated_at(db_session, b_id, minutes_ago=120)

    # Patch the symbol the cron imported so the first stuck doc raises
    # InvalidTransitionError and the second proceeds normally. Mirrors the
    # "race won by another writer" path without needing real concurrency.
    from rag_recipes.ingestion import cron as cron_module
    from rag_recipes.ingestion.status import InvalidTransitionError, mark_failed

    real_mark_failed: Callable[..., Awaitable[Any]] = mark_failed
    calls = {"n": 0}

    async def flaky_mark_failed(
        session: AsyncSession,
        document_id: str,
        **kwargs: Any,
    ) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise InvalidTransitionError(
                current=DocumentStatus.FAILED,
                attempted=DocumentStatus.FAILED,
                allowed=frozenset({DocumentStatus.QUEUED}),
            )
        return await real_mark_failed(session, document_id, **kwargs)

    monkeypatch.setattr(cron_module, "mark_failed", flaky_mark_failed)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)

    assert count == 1
    assert calls["n"] == 2

    db_session.expire_all()
    a_status = await db_session.scalar(
        select(Document.status).where(Document.id == a_id)
    )
    b_status = await db_session.scalar(
        select(Document.status).where(Document.id == b_id)
    )
    # The patched failure aborted the first attempt — its row was rolled
    # back by the savepoint. The second doc went through normally.
    statuses = {a_status, b_status}
    assert DocumentStatus.FAILED in statuses
    assert DocumentStatus.EXTRACTING_TEXT in statuses
