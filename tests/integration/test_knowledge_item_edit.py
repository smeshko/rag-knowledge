"""Integration tests for PATCH /api/v1/knowledge-items/{item_id} (Epic 22.2).

Two paths, and most of this file exercises the first:

- ``needs_review`` — the original verb: correct a flagged item in place instead
  of waving it through broken or rejecting it. Editing never decides; the item
  is still ``needs_review`` afterwards, with its content warnings re-derived.
- ``ready`` — a shelved recipe. The row rewrite is the same, but the item's
  chunks and embeddings describe the old text, so the handler drops them and
  re-indexes. See the re-indexing section near the end.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session
from rag_recipes.ingestion.pipeline.chunking import build_chunks
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository

pytestmark = pytest.mark.asyncio

_BODY_TEXT = "Bean Stew\n\n" + ("a slow-simmered pot of beans for a cold evening. " * 6)

_STRUCTURED: dict[str, Any] = {
    "schema": "recipe.v1",
    "yield": "Serves 4",
    "prep_time": None,
    "cook_time": None,
    "total_time": None,
    "ingredients_text": None,
    "ingredients": [
        {
            "position": 1,
            "raw_text": "1 cup dried beans",
            "quantity_text": "1",
            "quantity_value": 1.0,
            "unit_raw": "cup",
            "unit_normalized": "cup",
            "item_text": "dried beans",
            "item_normalized": "beans",
            "preparation": None,
            "notes": None,
            "confidence": {
                "overall": 0.9,
                "quantity": 0.9,
                "unit": 0.9,
                "item": 0.9,
                "normalization": 0.9,
            },
        }
    ],
    "steps_text": None,
    "steps": [],
    "warnings": ["no_steps"],
}

_CONFIDENCE: dict[str, Any] = {
    "overall": 0.9,
    "boundary": 0.9,
    "fields": {"title": 0.9, "summary": 0.9, "yield": 0.9, "ingredients": 0.9, "steps": 0.9},
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
        title="Edits Cookbook",
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


async def _seed_span(session: AsyncSession, *, document_id: str) -> SourceSpan:
    text = "page text for the bean stew"
    locator = {"type": "pdf_page_range", "page_start": 12, "page_end": 12}
    span = SourceSpan(
        id=new_id("span"),
        document_id=document_id,
        source_version=1,
        source_type=SourceType.PDF,
        locator=locator,
        locator_hash=hashlib.sha256(str(locator).encode()).hexdigest(),
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    session.add(span)
    await session.flush()
    return span


async def _seed_item(
    session: AsyncSession,
    *,
    run: ExtractionRun,
    status: KnowledgeItemStatus = KnowledgeItemStatus.NEEDS_REVIEW,
    span_ids: list[str] | None = None,
    structured: dict[str, Any] | None = None,
) -> KnowledgeItem:
    item = KnowledgeItem(
        document_id=run.document_id,
        extraction_run_id=run.id,
        source_version=run.source_version,
        item_type="recipe",
        title="Bean Stew",
        normalized_title="bean stew",
        summary="A hearty stew.",
        body_text=_BODY_TEXT,
        source_span_ids=span_ids or [],
        structured_data=structured if structured is not None else _deep_copy(_STRUCTURED),
        confidence=dict(_CONFIDENCE),
        status=status,
    )
    session.add(item)
    await session.flush()
    return item


def _deep_copy(value: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(value)


async def _reload(session: AsyncSession, item_id: str) -> KnowledgeItem:
    """Re-read the row through a cleared identity map.

    Every persistence assertion goes through this: it is what catches an
    in-place JSONB mutation that was never dirty-tracked and silently vanished
    at commit.
    """
    session.expunge_all()
    item = await session.get(KnowledgeItem, item_id)
    assert item is not None
    return item


# Long enough that the rebuilt body_text clears `extraction_min_recipe_chars`
# (200) — a rebuild composes title + ingredients + steps, so a terse correction
# genuinely trips `recipe_too_short`, and these tests are about the other rules.
_FIXED_STEPS = [
    "Soak the beans overnight in plenty of cold water, then drain them well.",
    "Simmer with the aromatics for two hours, topping up the water as needed.",
    "Season generously and rest the pot off the heat for a further ten minutes.",
]


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


async def test_partial_patch_changes_only_the_named_fields(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)
    before = _deep_copy(item.structured_data)

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}", json={"title": "Smoky Bean Stew"}
        )

    assert resp.status_code == 200, resp.text
    reloaded = await _reload(db_session, item.id)
    assert reloaded.title == "Smoky Bean Stew"
    assert reloaded.normalized_title == "smoky bean stew"
    # Untouched columns stay exactly as they were.
    assert reloaded.summary == "A hearty stew."
    assert reloaded.body_text == _BODY_TEXT
    assert reloaded.structured_data["ingredients"] == before["ingredients"]
    assert reloaded.confidence == _CONFIDENCE
    assert reloaded.source_span_ids == []


async def test_response_carries_recomputed_reasons_and_the_item_stays_pending(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Editing never decides — it only makes the item correct."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)
    assert item.structured_data["warnings"] == ["no_steps"]

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}", json={"steps": _FIXED_STEPS}
        )

    assert resp.status_code == 200, resp.text
    payload = resp.json()["knowledge_item"]
    assert payload["review_reasons"] == []
    assert payload["status"] == "needs_review"
    assert payload["edited_at"] is not None

    reloaded = await _reload(db_session, item.id)
    assert reloaded.status is KnowledgeItemStatus.NEEDS_REVIEW
    assert reloaded.structured_data["warnings"] == []
    assert reloaded.edited_at is not None


async def test_the_edit_survives_a_fresh_session_reload(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The JSONB dirty-tracking regression test.

    ``structured_data`` is unwrapped ``JSONB``: an in-place edit of a loaded
    row's dict is not dirty-tracked and is silently dropped at commit. Reading
    the row back through a cleared identity map is what proves the write landed.
    """
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}",
            json={
                "ingredients": ["1 cup dried beans", "2 smoked ham hocks"],
                "steps": _FIXED_STEPS,
            },
        )
    assert resp.status_code == 200, resp.text

    reloaded = await _reload(db_session, item.id)
    assert [i["raw_text"] for i in reloaded.structured_data["ingredients"]] == [
        "1 cup dried beans",
        "2 smoked ham hocks",
    ]
    assert [s["text"] for s in reloaded.structured_data["steps"]] == _FIXED_STEPS
    assert "2 smoked ham hocks" in reloaded.body_text
    assert reloaded.structured_data["warnings"] == []


async def test_untouched_ingredient_keeps_its_parse_and_the_edited_one_loses_it(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)
    original = _deep_copy(item.structured_data)["ingredients"][0]

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}",
            json={"ingredients": ["1 cup dried beans", "2 smoked ham hocks"]},
        )
    assert resp.status_code == 200, resp.text

    rows = (await _reload(db_session, item.id)).structured_data["ingredients"]
    assert rows[0] == original
    assert rows[1]["edited"] is True
    assert rows[1]["item_normalized"] is None
    assert rows[1]["confidence"]["normalization"] == 1.0


async def test_summary_is_cleared_by_an_explicit_null_and_left_alone_when_absent(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        absent = await client.patch(
            f"/api/v1/knowledge-items/{item.id}", json={"title": "Bean Stew II"}
        )
        assert absent.status_code == 200, absent.text
        assert (await _reload(db_session, item.id)).summary == "A hearty stew."

        cleared = await client.patch(f"/api/v1/knowledge-items/{item.id}", json={"summary": None})

    assert cleared.status_code == 200, cleared.text
    assert (await _reload(db_session, item.id)).summary is None


async def test_confidence_warnings_survive_a_content_edit(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    structured = _deep_copy(_STRUCTURED)
    structured["warnings"] = ["no_steps", "low_boundary_confidence"]
    item = await _seed_item(db_session, run=run, structured=structured)
    item.confidence = {**_CONFIDENCE, "boundary": 0.1}
    await db_session.flush()

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}", json={"steps": _FIXED_STEPS}
        )

    assert resp.status_code == 200, resp.text
    warnings = (await _reload(db_session, item.id)).structured_data["warnings"]
    assert warnings == ["low_boundary_confidence"]


# --------------------------------------------------------------------------- #
# The snapshot is the original extraction, not the previous revision
# --------------------------------------------------------------------------- #


async def test_a_second_edit_leaves_the_snapshot_at_the_original_extraction(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        first = await client.patch(
            f"/api/v1/knowledge-items/{item.id}", json={"title": "First Edit"}
        )
        assert first.status_code == 200, first.text
        second = await client.patch(
            f"/api/v1/knowledge-items/{item.id}", json={"title": "Second Edit"}
        )

    assert second.status_code == 200, second.text
    reloaded = await _reload(db_session, item.id)
    assert reloaded.title == "Second Edit"
    assert reloaded.pre_edit_snapshot is not None
    assert reloaded.pre_edit_snapshot["title"] == "Bean Stew"
    assert reloaded.pre_edit_snapshot["body_text"] == _BODY_TEXT
    assert reloaded.pre_edit_snapshot["structured_data"]["warnings"] == ["no_steps"]
    assert reloaded.pre_edit_snapshot["confidence"] == _CONFIDENCE


async def test_an_unedited_item_carries_no_snapshot(db_session: AsyncSession) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    reloaded = await _reload(db_session, item.id)
    assert reloaded.pre_edit_snapshot is None
    assert reloaded.edited_at is None


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


async def test_an_edit_creates_no_chunks_and_the_edited_text_is_what_would_be_indexed(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """``needs_review`` items carry no chunks, which is what makes an edit cheap.

    The corrected text only reaches the index on approve — so this asserts the
    edit wrote nothing, and that the chunks ``build_chunks`` *would* produce for
    the approved row carry the correction rather than the original.
    """
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}",
            json={
                "ingredients": ["1 cup dried beans", "2 smoked ham hocks"],
                "steps": _FIXED_STEPS,
            },
        )
    assert resp.status_code == 200, resp.text

    chunk_count = (
        await db_session.execute(select(Chunk).where(Chunk.parent_id == item.id))
    ).scalars().all()
    assert chunk_count == []

    reloaded = await _reload(db_session, item.id)
    reloaded.status = KnowledgeItemStatus.READY
    texts = [chunk.text for chunk in build_chunks(reloaded, category="recipes")]
    assert any("2 smoked ham hocks" in text for text in texts)
    assert any(_FIXED_STEPS[0] in text for text in texts)


# --------------------------------------------------------------------------- #
# Guards — each individually reachable, in the POST's order
# --------------------------------------------------------------------------- #


async def test_unknown_item_is_404_knowledge_item_not_found(client: httpx.AsyncClient) -> None:
    async with client:
        resp = await client.patch("/api/v1/knowledge-items/item_nope", json={"title": "X"})

    assert resp.status_code == 404
    body = resp.json()["error"]
    assert body["code"] == "knowledge_item_not_found"
    assert body["details"] == {"item_id": "item_nope"}


async def test_mid_reprocess_document_is_409_ingestion_already_running(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session, status=DocumentStatus.EXTRACTING_ITEMS)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.patch(f"/api/v1/knowledge-items/{item.id}", json={"title": "X"})

    assert resp.status_code == 409
    body = resp.json()["error"]
    assert body["code"] == "ingestion_already_running"
    assert body["details"] == {"document_id": doc.id, "status": "extracting_items"}


async def test_stale_generation_is_409_review_item_stale(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    old_run = await _seed_run(db_session, document_id=doc.id, source_version=1)
    new_run = await _seed_run(db_session, document_id=doc.id, source_version=2)
    stale = await _seed_item(db_session, run=old_run)
    await _seed_item(db_session, run=new_run)

    async with client:
        resp = await client.patch(f"/api/v1/knowledge-items/{stale.id}", json={"title": "X"})

    assert resp.status_code == 409
    body = resp.json()["error"]
    assert body["code"] == "review_item_stale"
    assert body["details"] == {
        "item_id": stale.id,
        "source_version": 1,
        "current_source_version": 2,
    }


@pytest.mark.parametrize(
    "status",
    [
        KnowledgeItemStatus.REJECTED,
        KnowledgeItemStatus.INDEXING,
        KnowledgeItemStatus.SUPERSEDED,
    ],
)
async def test_a_non_editable_item_is_404_review_not_pending(
    client: httpx.AsyncClient, db_session: AsyncSession, status: KnowledgeItemStatus
) -> None:
    """Only needs_review and ready are editable; the rest are dead or mid-flight.

    ``READY`` is deliberately absent from this list — it moved to the
    re-indexing path below when the delete-and-re-embed route landed. The code
    stays ``review_not_pending`` because the frontend keys its refusal copy on
    it.
    """
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run, status=status)

    async with client:
        resp = await client.patch(f"/api/v1/knowledge-items/{item.id}", json={"title": "X"})

    assert resp.status_code == 404
    body = resp.json()["error"]
    assert body["code"] == "review_not_pending"
    assert body["details"] == {"item_id": item.id, "status": status.value}


# --------------------------------------------------------------------------- #
# The ready path: drop the stale index, re-embed from the saved text
# --------------------------------------------------------------------------- #


async def _seed_indexed_item(
    session: AsyncSession, *, run: ExtractionRun
) -> tuple[KnowledgeItem, str]:
    """A ``ready`` item carrying one chunk and one embedding, as indexing leaves it."""
    item = await _seed_item(session, run=run, status=KnowledgeItemStatus.READY)
    chunk = Chunk(
        document_id=run.document_id,
        parent_type="knowledge_item",
        parent_id=item.id,
        chunk_type="recipe_full",
        text="the stale indexed text",
        text_hash=hashlib.sha256(f"stale-{item.id}".encode()).hexdigest(),
        source_span_ids=[],
        chunk_metadata={"category": "recipes"},
    )
    session.add(chunk)
    await session.flush()
    session.add(
        ChunkEmbedding(
            chunk_id=chunk.id,
            embedding_provider="fake",
            embedding_model="fake-embedding",
            embedding_dimensions=1536,
            embedding_vector=[0.0] * 1536,
        )
    )
    await session.flush()
    return item, chunk.id


async def _index_row_counts(session: AsyncSession, item_id: str) -> tuple[int, int]:
    chunk_ids = select(Chunk.id).where(Chunk.parent_id == item_id)
    chunks = await session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.parent_id == item_id)
    )
    embeddings = await session.scalar(
        select(func.count())
        .select_from(ChunkEmbedding)
        .where(ChunkEmbedding.chunk_id.in_(chunk_ids))
    )
    return int(chunks or 0), int(embeddings or 0)


async def test_editing_a_ready_item_drops_its_index_and_enqueues_a_reindex(
    client: httpx.AsyncClient, db_session: AsyncSession, fake_arq_redis: Any
) -> None:
    """The whole point of the ready path: the old chunks must not survive.

    If they did, ``index_knowledge_item``'s defensive "already has chunks" guard
    would skip rebuilding and re-embed the OLD text — the recipe would stay
    findable by words it no longer contains, and the request would report
    success.
    """
    doc = await _seed_document(db_session, status=DocumentStatus.READY)
    run = await _seed_run(db_session, document_id=doc.id)
    item, _ = await _seed_indexed_item(db_session, run=run)
    assert await _index_row_counts(db_session, item.id) == (1, 1)

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}",
            json={"title": "Corrected Bean Stew", "steps": _FIXED_STEPS},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["knowledge_item"]["title"] == "Corrected Bean Stew"
    # The response reports the transitional status, not a comforting "ready" —
    # the recipe genuinely is not searchable until the worker finishes.
    assert body["knowledge_item"]["status"] == "indexing"

    reloaded = await _reload(db_session, item.id)
    assert reloaded.status is KnowledgeItemStatus.INDEXING
    assert reloaded.title == "Corrected Bean Stew"
    assert reloaded.edited_at is not None
    assert await _index_row_counts(db_session, item.id) == (0, 0)

    assert fake_arq_redis.enqueue_job.await_count == 1
    call = fake_arq_redis.enqueue_job.await_args
    assert call.args == ("index_knowledge_item", item.id)
    assert call.kwargs["_session_id"] == doc.id


async def test_editing_a_needs_review_item_still_enqueues_nothing(
    client: httpx.AsyncClient, db_session: AsyncSession, fake_arq_redis: Any
) -> None:
    """The original path is untouched: a row rewrite, still awaiting a decision."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}", json={"title": "Corrected"}
        )

    assert resp.status_code == 200, resp.text
    assert resp.json()["knowledge_item"]["status"] == "needs_review"
    assert (await _reload(db_session, item.id)).status is KnowledgeItemStatus.NEEDS_REVIEW
    assert await _index_row_counts(db_session, item.id) == (0, 0)
    assert fake_arq_redis.enqueue_job.await_count == 0


async def test_a_failed_reindex_enqueue_reverts_the_item_to_needs_review(
    client: httpx.AsyncClient, db_session: AsyncSession, fake_arq_redis: Any
) -> None:
    """Compensation lands on needs_review, not back on ready.

    The chunks are already gone by then, so a row labelled ``ready`` would claim
    to be on the shelf while being unfindable. Chunk-free IS what needs_review
    means.
    """
    doc = await _seed_document(db_session, status=DocumentStatus.READY)
    run = await _seed_run(db_session, document_id=doc.id)
    item, _ = await _seed_indexed_item(db_session, run=run)
    fake_arq_redis.enqueue_job.side_effect = RuntimeError("redis down")

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}",
            json={"title": "Corrected Bean Stew", "steps": _FIXED_STEPS},
        )

    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "internal_error"
    reloaded = await _reload(db_session, item.id)
    assert reloaded.status is KnowledgeItemStatus.NEEDS_REVIEW
    # The edit itself committed — it is the indexing that failed.
    assert reloaded.title == "Corrected Bean Stew"
    assert await _index_row_counts(db_session, item.id) == (0, 0)


async def test_an_edit_after_a_decide_gets_review_not_pending(
    client: httpx.AsyncClient, db_session: AsyncSession, fake_arq_redis: Any
) -> None:
    """A decided item can no longer be edited, and the edit writes nothing.

    Sequential by construction — the genuinely concurrent case (two connections
    contending on the guarded UPDATE) needs committed rows and lives in
    ``test_knowledge_item_edit_race.py``.
    """
    from rag_recipes.api.dependencies import get_arq_redis

    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    app.dependency_overrides[get_arq_redis] = lambda: fake_arq_redis
    try:
        async with client:
            decided = await client.post(
                f"/api/v1/knowledge-items/{item.id}/review", json={"decision": "rejected"}
            )
            assert decided.status_code == 200, decided.text
            losing_edit = await client.patch(
                f"/api/v1/knowledge-items/{item.id}", json={"title": "Too late"}
            )
    finally:
        app.dependency_overrides.pop(get_arq_redis, None)

    assert losing_edit.status_code == 404
    assert losing_edit.json()["error"]["code"] == "review_not_pending"
    reloaded = await _reload(db_session, item.id)
    assert reloaded.title == "Bean Stew"
    assert reloaded.edited_at is None


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #


async def test_the_scalar_metadata_fields_round_trip_through_the_alias(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """``yield`` is a Python keyword, so the field is only reachable by alias."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.patch(
            f"/api/v1/knowledge-items/{item.id}",
            json={
                "yield": "Serves 6",
                "prep_time": "15 min",
                "cook_time": "1 hr",
                "total_time": None,
            },
        )

    assert resp.status_code == 200, resp.text
    structured = (await _reload(db_session, item.id)).structured_data
    assert structured["yield"] == "Serves 6"
    assert structured["prep_time"] == "15 min"
    assert structured["cook_time"] == "1 hr"
    assert structured["total_time"] is None
    # A scalar-only edit touches no lines, so the body is left alone.
    assert (await _reload(db_session, item.id)).body_text == _BODY_TEXT


async def test_an_unknown_id_reports_as_unknown_whatever_the_body_says(
    client: httpx.AsyncClient,
) -> None:
    """The empty-body 400 must not mask the 404 (guard-order regression)."""
    async with client:
        resp = await client.patch("/api/v1/knowledge-items/item_nope", json={})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_item_not_found"


async def test_an_empty_patch_is_rejected(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.patch(f"/api/v1/knowledge-items/{item.id}", json={})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"title": None}, id="null-title"),
        pytest.param({"title": "   "}, id="blank-title"),
        pytest.param({"ingredients": ["1 cup beans", "  "]}, id="blank-ingredient"),
        pytest.param({"steps": [""]}, id="blank-step"),
        pytest.param({"ingredients": None}, id="null-ingredients"),
        pytest.param({"steps": None}, id="null-steps"),
        pytest.param({"yield_": "Serves 6"}, id="yield-by-field-name-not-alias"),
        pytest.param({"confidence": {"overall": 1.0}}, id="machine-owned-field"),
        pytest.param({"warnings": []}, id="warnings-not-writable"),
        pytest.param({"source_span_ids": ["span_1"]}, id="provenance-not-writable"),
    ],
)
async def test_invalid_payloads_are_422(
    client: httpx.AsyncClient, db_session: AsyncSession, payload: dict[str, Any]
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run)

    async with client:
        resp = await client.patch(f"/api/v1/knowledge-items/{item.id}", json=payload)

    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "invalid_request"


# --------------------------------------------------------------------------- #
# Read-side exposure
# --------------------------------------------------------------------------- #


async def test_edited_at_is_exposed_by_the_detail_and_the_queue(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    span = await _seed_span(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run, span_ids=[span.id])

    async with client:
        before_detail = await client.get(f"/api/v1/knowledge-items/{item.id}")
        before_queue = await client.get("/api/v1/review-items", params={"document_id": doc.id})
        assert before_detail.json()["knowledge_item"]["edited_at"] is None
        assert before_queue.json()["review_items"][0]["edited_at"] is None

        patched = await client.patch(
            f"/api/v1/knowledge-items/{item.id}", json={"steps": _FIXED_STEPS}
        )
        assert patched.status_code == 200, patched.text

        after_detail = await client.get(f"/api/v1/knowledge-items/{item.id}")
        after_queue = await client.get("/api/v1/review-items", params={"document_id": doc.id})

    edited_at = patched.json()["knowledge_item"]["edited_at"]
    assert edited_at is not None
    assert after_detail.json()["knowledge_item"]["edited_at"] == edited_at
    assert after_queue.json()["review_items"][0]["edited_at"] == edited_at
    # The queue's flag list is derived from the rewritten warnings.
    assert after_queue.json()["review_items"][0]["flags"] == []


async def test_the_patch_response_matches_the_detail_endpoint(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """One envelope: an edit cannot answer with a shape the GET would not."""
    doc = await _seed_document(db_session)
    run = await _seed_run(db_session, document_id=doc.id)
    span = await _seed_span(db_session, document_id=doc.id)
    item = await _seed_item(db_session, run=run, span_ids=[span.id])

    async with client:
        patched = await client.patch(
            f"/api/v1/knowledge-items/{item.id}", json={"steps": _FIXED_STEPS}
        )
        detail = await client.get(f"/api/v1/knowledge-items/{item.id}")

    assert patched.status_code == 200, patched.text
    assert patched.json() == detail.json()
    assert patched.json()["source_citations"][0]["label"] == "page 12"
