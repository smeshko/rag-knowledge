"""A genuine edit-vs-decide race for PATCH /knowledge-items/{id} (Epic 22.2).

Separate from ``test_knowledge_item_edit.py`` because a real race needs
**committed** rows and one connection per request: the savepoint ``db_session``
fixture hands every request the same uncommitted transaction, where two
"concurrent" calls are just two sequential statements and the guarded UPDATE is
never actually contended.

Here each request gets its own session from the engine, the rows are committed,
and the PATCH and the decide are fired with ``asyncio.gather`` — so the
``WHERE status = 'needs_review'`` predicate is what picks the winner, exactly as
in production. Rows are cleaned up afterwards (the burst-worker pattern from
``test_index_knowledge_item_job.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from arq.connections import ArqRedis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_arq_redis, get_session
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
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.session import build_session_factory
from tests.integration.conftest import AUTH_HEADERS

pytestmark = pytest.mark.asyncio

_BODY_TEXT = "Raced Stew\n\n" + ("a slow-simmered pot of beans for a cold evening. " * 6)

_STRUCTURED: dict[str, Any] = {
    "schema": "recipe.v1",
    "yield": "Serves 4",
    "prep_time": None,
    "cook_time": None,
    "total_time": None,
    "ingredients_text": None,
    "ingredients": [],
    "steps_text": None,
    "steps": [],
    "warnings": ["no_ingredients", "no_steps"],
}


async def _seed_committed(engine: AsyncEngine) -> tuple[str, str, str]:
    """Commit a document + run + needs_review item; return their ids."""
    factory = build_session_factory(engine)
    async with factory() as session:
        asset_id = new_id("asset")
        session.add(
            SourceAsset(
                id=asset_id,
                source_type=SourceType.PDF,
                original_filename="raced.pdf",
                storage_provider="fake",
                storage_key=f"source-assets/{asset_id}/original.pdf",
                content_hash=hashlib.sha256(asset_id.encode()).hexdigest(),
                upload_status=UploadStatus.UPLOADED,
            )
        )
        document = Document(
            asset_id=asset_id,
            category="recipes",
            title="Race Cookbook",
            author="",
            source_type=SourceType.PDF,
            status=DocumentStatus.NEEDS_REVIEW,
        )
        session.add(document)
        await session.flush()
        run = ExtractionRun(
            document_id=document.id,
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
            document_id=document.id,
            extraction_run_id=run.id,
            source_version=1,
            item_type="recipe",
            title="Raced Stew",
            normalized_title="raced stew",
            summary=None,
            body_text=_BODY_TEXT,
            source_span_ids=[],
            structured_data=dict(_STRUCTURED),
            confidence={"overall": 0.9, "boundary": 0.9},
            status=KnowledgeItemStatus.NEEDS_REVIEW,
        )
        session.add(item)
        await session.flush()
        await session.commit()
        return document.id, asset_id, item.id


async def _cleanup(engine: AsyncEngine, document_id: str, asset_id: str) -> None:
    factory = build_session_factory(engine)
    async with factory() as session:
        await session.execute(delete(KnowledgeItem).where(KnowledgeItem.document_id == document_id))
        await session.execute(delete(ExtractionRun).where(ExtractionRun.document_id == document_id))
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.execute(delete(SourceAsset).where(SourceAsset.id == asset_id))
        await session.commit()


async def test_an_edit_and_a_decide_racing_have_exactly_one_winner(
    test_engine: AsyncEngine, override_settings_with_token: None
) -> None:
    """The guarded UPDATE, not luck, decides which of the two lands."""
    document_id, asset_id, item_id = await _seed_committed(test_engine)
    factory = build_session_factory(test_engine)

    async def _session_per_request() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    fake_redis = AsyncMock(spec=ArqRedis)
    fake_redis.enqueue_job = AsyncMock(return_value=AsyncMock())

    app.dependency_overrides[get_session] = _session_per_request
    app.dependency_overrides[get_arq_redis] = lambda: fake_redis
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=AUTH_HEADERS
        ) as client:
            edit, decide = await asyncio.gather(
                client.patch(
                    f"/api/v1/knowledge-items/{item_id}",
                    json={"ingredients": ["2 tbsp olive oil", "3 ripe tomatoes"]},
                ),
                client.post(
                    f"/api/v1/knowledge-items/{item_id}/review",
                    json={"decision": "rejected"},
                ),
            )

        statuses = sorted([edit.status_code, decide.status_code])
        assert statuses == [200, 404], f"edit={edit.text} decide={decide.text}"
        loser = edit if edit.status_code == 404 else decide
        assert loser.json()["error"]["code"] == "review_not_pending"

        async with factory() as session:
            row = (
                await session.execute(
                    select(KnowledgeItem.status, KnowledgeItem.edited_at).where(
                        KnowledgeItem.id == item_id
                    )
                )
            ).one()

        if edit.status_code == 200:
            # The edit won: the item is still pending review and now marked edited.
            assert row.status is KnowledgeItemStatus.NEEDS_REVIEW
            assert row.edited_at is not None
        else:
            # The decide won: the edit wrote nothing at all, not even edited_at.
            assert row.status is KnowledgeItemStatus.REJECTED
            assert row.edited_at is None
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_arq_redis, None)
        await _cleanup(test_engine, document_id, asset_id)
