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
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionBatchItemStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
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


async def _add_batch_item(
    session: AsyncSession,
    document_id: str,
    *,
    status: ExtractionBatchItemStatus,
    input_hash: str,
) -> None:
    session.add(
        ExtractionBatchItem(
            document_id=document_id,
            source_version=1,
            input_hash=input_hash,
            input_source_span_ids=["span_a"],
            request_input="window",
            request_schema={"type": "object"},
            prompt_version="recipe-extraction-v1",
            schema_version="recipe.v1",
            status=status,
        )
    )
    await session.flush()


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


# --- batch-aware exemption (Epic 19.2) --------------------------------------


@pytest.mark.parametrize(
    "item_status",
    [ExtractionBatchItemStatus.SUBMITTING, ExtractionBatchItemStatus.SUBMITTED],
)
async def test_sweep_exempts_in_flight_batch_document(
    db_session: AsyncSession,
    item_status: ExtractionBatchItemStatus,
) -> None:
    # An EXTRACTING_ITEMS doc with an in-flight (SUBMITTING/SUBMITTED) batch item
    # may legitimately wait up to 24h — a stale heartbeat must NOT reap it.
    document_id = await _make_document(
        db_session,
        content_hash=f"sweep-batch-inflight-{item_status.value}",
        status=DocumentStatus.EXTRACTING_ITEMS,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)
    await _set_last_progress_at(db_session, document_id, minutes_ago=120)
    await _add_batch_item(
        db_session, document_id, status=item_status, input_hash="inflight-1"
    )

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


async def test_sweep_reaps_stale_pending_only_batch_document(
    db_session: AsyncSession,
) -> None:
    # Only PENDING items means the submitter never claimed them (disabled/broken).
    # Such a doc must be surfaced (reaped), never hidden forever (DECISIONS #3).
    document_id = await _make_document(
        db_session,
        content_hash="sweep-batch-pending-only",
        status=DocumentStatus.EXTRACTING_ITEMS,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)
    await _set_last_progress_at(db_session, document_id, minutes_ago=120)
    await _add_batch_item(
        db_session,
        document_id,
        status=ExtractionBatchItemStatus.PENDING,
        input_hash="pending-1",
    )

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


async def test_sweep_reaps_terminal_only_batch_document(
    db_session: AsyncSession,
) -> None:
    # All-terminal items (e.g. after 19.3 ingestion) no longer exempt the doc — it
    # should make progress on its own; if wedged + stale it is genuinely stuck.
    document_id = await _make_document(
        db_session,
        content_hash="sweep-batch-terminal-only",
        status=DocumentStatus.EXTRACTING_ITEMS,
    )
    await _backdate_updated_at(db_session, document_id, minutes_ago=120)
    await _set_last_progress_at(db_session, document_id, minutes_ago=120)
    await _add_batch_item(
        db_session,
        document_id,
        status=ExtractionBatchItemStatus.SUCCEEDED,
        input_hash="succeeded-1",
    )

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


# ---------------------------------------------------------------------------
# Phase 21.3 (D1b): item-level stuck-`indexing` sweep backstop
# ---------------------------------------------------------------------------


async def _backdate_item_updated_at(
    session: AsyncSession, item_id: str, *, minutes_ago: int
) -> None:
    """Bypass ``onupdate=func.now()`` by writing ``updated_at`` via raw SQL."""
    await session.execute(
        text(
            "UPDATE knowledge_items "
            "SET updated_at = now() - make_interval(mins => :minutes) "
            "WHERE id = :id"
        ),
        {"minutes": minutes_ago, "id": item_id},
    )


async def _add_review_item(
    session: AsyncSession,
    document_id: str,
    *,
    status: KnowledgeItemStatus,
    run: ExtractionRun | None = None,
) -> tuple[str, ExtractionRun]:
    if run is None:
        run = ExtractionRun(
            document_id=document_id,
            source_version=1,
            provider="fake",
            model="fake-model",
            prompt_version="test-prompt-v1",
            schema_version="test-schema-v1",
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
        title=f"Item {new_id('t')}",
        normalized_title="item",
        summary=None,
        body_text="body " * 20,
        source_span_ids=[],
        structured_data={"schema": "recipe.v1", "warnings": ["no_steps"]},
        confidence=None,
        status=status,
    )
    session.add(item)
    await session.flush()
    return item.id, run


def test_stuck_indexing_timeout_defaults_to_15_minutes() -> None:
    from rag_recipes.config import Settings

    assert Settings.model_fields["stuck_indexing_timeout_minutes"].default == 15


async def test_item_sweep_reverts_aged_indexing_item_and_spares_the_rest(
    db_session: AsyncSession,
    override_settings_with_token: None,
    auth_headers: dict[str, str],
) -> None:
    document_id = await _make_document(
        db_session,
        content_hash="sweep-item-1",
        status=DocumentStatus.NEEDS_REVIEW,
    )
    aged_id, run = await _add_review_item(
        db_session, document_id, status=KnowledgeItemStatus.INDEXING
    )
    await _backdate_item_updated_at(db_session, aged_id, minutes_ago=60)
    fresh_id, _ = await _add_review_item(
        db_session, document_id, status=KnowledgeItemStatus.INDEXING, run=run
    )
    controls: dict[str, KnowledgeItemStatus] = {}
    for status in (
        KnowledgeItemStatus.READY,
        KnowledgeItemStatus.NEEDS_REVIEW,
        KnowledgeItemStatus.REJECTED,
        KnowledgeItemStatus.SUPERSEDED,
        KnowledgeItemStatus.EXTRACTING,
    ):
        item_id, _ = await _add_review_item(
            db_session, document_id, status=status, run=run
        )
        # Age every control too: only INDEXING may ever match the predicate.
        await _backdate_item_updated_at(db_session, item_id, minutes_ago=60)
        controls[item_id] = status

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)
    # The document sweep's return value is untouched by the item pass: the
    # parent doc here is terminal, so zero documents were swept.
    assert count == 0

    db_session.expire_all()
    statuses = {
        item_id: status
        for item_id, status in (
            await db_session.execute(
                select(KnowledgeItem.id, KnowledgeItem.status).where(
                    KnowledgeItem.document_id == document_id
                )
            )
        ).all()
    }
    assert statuses[aged_id] is KnowledgeItemStatus.NEEDS_REVIEW
    assert statuses[fresh_id] is KnowledgeItemStatus.INDEXING
    for item_id, expected in controls.items():
        assert statuses[item_id] is expected

    # The reverted item reappears in the review queue.
    from rag_recipes.api.app import app
    from rag_recipes.api.dependencies import get_session

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    try:
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=auth_headers
        ) as client:
            resp = await client.get(
                "/api/v1/review-items", params={"document_id": document_id}
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
    assert resp.status_code == 200
    listed_ids = {entry["id"] for entry in resp.json()["review_items"]}
    assert aged_id in listed_ids


async def test_item_sweep_does_not_change_document_sweep_return_value(
    db_session: AsyncSession,
) -> None:
    stuck_doc_id = await _make_document(
        db_session,
        content_hash="sweep-item-2",
        status=DocumentStatus.EXTRACTING_TEXT,
    )
    await _backdate_updated_at(db_session, stuck_doc_id, minutes_ago=120)
    terminal_doc_id = await _make_document(
        db_session,
        content_hash="sweep-item-3",
        status=DocumentStatus.NEEDS_REVIEW,
    )
    aged_item_id, _ = await _add_review_item(
        db_session, terminal_doc_id, status=KnowledgeItemStatus.INDEXING
    )
    await _backdate_item_updated_at(db_session, aged_item_id, minutes_ago=60)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    count = await sweep_stuck_jobs(ctx)
    # Return value is the DOCUMENT count only, even though one item was swept.
    assert count == 1

    db_session.expire_all()
    item_status = await db_session.scalar(
        select(KnowledgeItem.status).where(KnowledgeItem.id == aged_item_id)
    )
    assert item_status is KnowledgeItemStatus.NEEDS_REVIEW


async def test_late_original_job_noops_on_a_swept_item(
    db_session: AsyncSession,
) -> None:
    """D1b safety: an original index job that finally arrives after the sweep
    reverted the row hits its `status is INDEXING` guard and no-ops — the row
    is not resurrected and no chunks are written."""
    from rag_recipes.ingestion.jobs import index_knowledge_item
    from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
    from rag_recipes.storage.models.chunk import Chunk

    document_id = await _make_document(
        db_session,
        content_hash="sweep-item-4",
        status=DocumentStatus.NEEDS_REVIEW,
    )
    aged_id, _ = await _add_review_item(
        db_session, document_id, status=KnowledgeItemStatus.INDEXING
    )
    await _backdate_item_updated_at(db_session, aged_id, minutes_ago=60)

    ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
    }
    await sweep_stuck_jobs(ctx)

    job_ctx: dict[str, Any] = {
        "settings": get_settings(),
        "session_factory": _make_session_factory(db_session),
        "embedding_provider": FakeEmbeddingProvider(),
    }
    result = await index_knowledge_item(job_ctx, aged_id)
    assert result == 0

    db_session.expire_all()
    status = await db_session.scalar(
        select(KnowledgeItem.status).where(KnowledgeItem.id == aged_id)
    )
    assert status is KnowledgeItemStatus.NEEDS_REVIEW
    chunks = (
        await db_session.execute(
            select(Chunk.id).where(Chunk.parent_id == aged_id)
        )
    ).all()
    assert chunks == []
