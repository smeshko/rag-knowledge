"""Integration tests for POST /api/v1/knowledge-items/{item_id}/review (Epic 21.3, contract §2)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.repositories.documents import DocumentRepository

_STRUCTURED: dict[str, Any] = {
    "schema": "recipe.v1",
    "yield": "Serves 4",
    "ingredients": [{"raw_text": "1 cup beans", "item_normalized": "beans"}],
    "warnings": ["no_steps"],
}


@pytest.fixture
def client(
    db_session: AsyncSession,
    override_settings_with_token: None,
    auth_headers: dict[str, str],
) -> Iterator[httpx.AsyncClient]:
    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        yield httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=auth_headers
        )
    finally:
        app.dependency_overrides.pop(get_session, None)


async def _seed_document(
    session: AsyncSession, *, status: DocumentStatus = DocumentStatus.NEEDS_REVIEW
) -> Document:
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
    return await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="Decisions Cookbook",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=status,
    )


async def _seed_run(
    session: AsyncSession, *, document_id: str, source_version: int = 1
) -> ExtractionRun:
    run = ExtractionRun(
        document_id=document_id,
        source_version=source_version,
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
    return run


async def _seed_item(
    session: AsyncSession,
    *,
    run: ExtractionRun,
    status: KnowledgeItemStatus = KnowledgeItemStatus.NEEDS_REVIEW,
) -> KnowledgeItem:
    item = KnowledgeItem(
        document_id=run.document_id,
        extraction_run_id=run.id,
        source_version=run.source_version,
        item_type="recipe",
        title=f"Item {new_id('t')}",
        normalized_title="item",
        summary=None,
        body_text="body " * 30,
        source_span_ids=[],
        structured_data=dict(_STRUCTURED),
        confidence={"overall": 0.5},
        status=status,
    )
    session.add(item)
    await session.flush()
    return item


async def _reload_status(
    session: AsyncSession, item_id: str
) -> KnowledgeItemStatus | None:
    session.expunge_all()
    item = await session.get(KnowledgeItem, item_id)
    return None if item is None else item.status


@pytest.mark.asyncio
async def test_approve_returns_indexing_and_enqueues_job(
    client: httpx.AsyncClient, db_session: AsyncSession, fake_arq_redis: AsyncMock
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.post(
            f"/api/v1/knowledge-items/{item.id}/review",
            json={"decision": "approved"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "knowledge_item": {
            "id": item.id,
            "document_id": doc.id,
            "status": "indexing",
        },
        "decision": "approved",
    }
    assert await _reload_status(db_session, item.id) is KnowledgeItemStatus.INDEXING
    assert fake_arq_redis.enqueue_job.await_count == 1
    call = fake_arq_redis.enqueue_job.await_args
    assert call.args == ("index_knowledge_item", item.id)
    assert call.kwargs["_session_id"] == doc.id


@pytest.mark.asyncio
async def test_reject_is_terminal_and_enqueues_nothing(
    client: httpx.AsyncClient, db_session: AsyncSession, fake_arq_redis: AsyncMock
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.post(
            f"/api/v1/knowledge-items/{item.id}/review",
            json={"decision": "rejected"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["knowledge_item"]["status"] == "rejected"
    assert resp.json()["decision"] == "rejected"
    assert await _reload_status(db_session, item.id) is KnowledgeItemStatus.REJECTED
    assert fake_arq_redis.enqueue_job.await_count == 0


@pytest.mark.asyncio
async def test_unknown_item_404s_knowledge_item_not_found(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    async with client:
        resp = await client.post(
            "/api/v1/knowledge-items/item_does_not_exist/review",
            json={"decision": "approved"},
        )
    assert resp.status_code == 404, resp.text
    body = resp.json()
    assert body["error"]["code"] == "knowledge_item_not_found"
    assert body["error"]["details"] == {"item_id": "item_does_not_exist"}


@pytest.mark.asyncio
async def test_decided_and_ready_items_404_review_not_pending_with_true_status(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    pending = await _seed_item(db_session, run=run)
    ready = await _seed_item(db_session, run=run, status=KnowledgeItemStatus.READY)

    async with client:
        first = await client.post(
            f"/api/v1/knowledge-items/{pending.id}/review",
            json={"decision": "rejected"},
        )
        second = await client.post(
            f"/api/v1/knowledge-items/{pending.id}/review",
            json={"decision": "rejected"},
        )
        on_ready = await client.post(
            f"/api/v1/knowledge-items/{ready.id}/review",
            json={"decision": "approved"},
        )
    assert first.status_code == 200
    assert second.status_code == 404, second.text
    body = second.json()
    assert body["error"]["code"] == "review_not_pending"
    # The *true* current status — proves the explicit re-select, not a stale
    # identity-map read.
    assert body["error"]["details"] == {"item_id": pending.id, "status": "rejected"}
    assert on_ready.status_code == 404
    assert on_ready.json()["error"]["details"] == {
        "item_id": ready.id,
        "status": "ready",
    }


@pytest.mark.asyncio
async def test_malformed_decision_422s_in_the_envelope(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.post(
            f"/api/v1/knowledge-items/{item.id}/review",
            json={"decision": "maybe"},
        )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "invalid_request"
    assert await _reload_status(db_session, item.id) is KnowledgeItemStatus.NEEDS_REVIEW


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approved", "rejected"])
async def test_non_terminal_document_409s_both_decisions(
    client: httpx.AsyncClient, db_session: AsyncSession, decision: str
) -> None:
    doc = await _seed_document(db_session, status=DocumentStatus.QUEUED)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.post(
            f"/api/v1/knowledge-items/{item.id}/review",
            json={"decision": decision},
        )
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["error"]["code"] == "ingestion_already_running"
    assert body["error"]["details"] == {"document_id": doc.id, "status": "queued"}
    assert await _reload_status(db_session, item.id) is KnowledgeItemStatus.NEEDS_REVIEW


@pytest.mark.asyncio
async def test_stale_generation_approve_409s_but_reject_succeeds(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """D9: approving a stale generation is refused; rejecting it clears the queue."""
    doc = await _seed_document(db_session)
    run_v1 = await _seed_run(db_session, document_id=doc.id, source_version=1)
    run_v2 = await _seed_run(db_session, document_id=doc.id, source_version=2)
    stale = await _seed_item(db_session, run=run_v1)
    fresh = await _seed_item(db_session, run=run_v2)

    async with client:
        approve = await client.post(
            f"/api/v1/knowledge-items/{stale.id}/review",
            json={"decision": "approved"},
        )
        assert approve.status_code == 409, approve.text
        body = approve.json()
        assert body["error"]["code"] == "review_item_stale"
        assert body["error"]["details"] == {
            "item_id": stale.id,
            "source_version": 1,
            "current_source_version": 2,
        }
        assert (
            await _reload_status(db_session, stale.id)
            is KnowledgeItemStatus.NEEDS_REVIEW
        )

        reject = await client.post(
            f"/api/v1/knowledge-items/{stale.id}/review",
            json={"decision": "rejected"},
        )
        assert reject.status_code == 200, reject.text
        assert (
            await _reload_status(db_session, stale.id) is KnowledgeItemStatus.REJECTED
        )

        # Gone from the listing and from the counts; the fresh item remains.
        listing = await client.get(
            "/api/v1/review-items", params={"document_id": doc.id}
        )
        assert [e["id"] for e in listing.json()["review_items"]] == [fresh.id]
        detail = await client.get(f"/api/v1/documents/{doc.id}")
        counts = detail.json()["counts"]
        assert counts["knowledge_items"] == 1
        assert counts["needs_review_items"] == 1


@pytest.mark.asyncio
async def test_enqueue_failure_reverts_to_needs_review_and_500s(
    client: httpx.AsyncClient, db_session: AsyncSession, fake_arq_redis: AsyncMock
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)
    fake_arq_redis.enqueue_job.side_effect = RuntimeError("redis down")

    async with client:
        resp = await client.post(
            f"/api/v1/knowledge-items/{item.id}/review",
            json={"decision": "approved"},
        )
    assert resp.status_code == 500, resp.text
    assert resp.json()["error"]["code"] == "internal_error"
    # The compensating revert is observable: the row is back in the queue.
    assert await _reload_status(db_session, item.id) is KnowledgeItemStatus.NEEDS_REVIEW


@pytest.mark.asyncio
async def test_counts_and_derived_status_track_decisions(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Phase 21.3 (D3/D5) consistency flow: counts and the derived pill move
    together on every decision, and the last decision flips the pill to ready
    via the POST (list + detail agree)."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    first = await _seed_item(db_session, run=run)
    second = await _seed_item(db_session, run=run)

    async with client:
        before = await client.get(f"/api/v1/documents/{doc.id}")
        counts = before.json()["counts"]
        assert before.json()["document"]["status"] == "needs_review"
        assert counts["needs_review_items"] == 2
        assert counts["knowledge_items"] == 2

        reject = await client.post(
            f"/api/v1/knowledge-items/{first.id}/review",
            json={"decision": "rejected"},
        )
        assert reject.status_code == 200
        mid = await client.get(f"/api/v1/documents/{doc.id}")
        counts = mid.json()["counts"]
        assert mid.json()["document"]["status"] == "needs_review"
        assert counts["needs_review_items"] == 1
        # Rejection also drops the row out of the total (D5).
        assert counts["knowledge_items"] == 1

        approve = await client.post(
            f"/api/v1/knowledge-items/{second.id}/review",
            json={"decision": "approved"},
        )
        assert approve.status_code == 200
        after = await client.get(f"/api/v1/documents/{doc.id}")
        counts = after.json()["counts"]
        # The last pending item is decided: the pill turns green even while the
        # approved item is still `indexing` (D3 recorded consequence).
        assert after.json()["document"]["status"] == "ready"
        assert counts["needs_review_items"] == 0
        assert counts["knowledge_items"] == 1
        assert counts["ready_items"] == 0

        listing = await client.get("/api/v1/documents")
        by_id = {d["id"]: d["status"] for d in listing.json()["documents"]}
        assert by_id[doc.id] == "ready"
