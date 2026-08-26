"""Integration tests for ``POST /api/v1/knowledge-items`` — writing a recipe by hand.

Two halves, and they need different isolation:

- most tests run on the savepoint ``db_session`` fixture, which rolls back the
  fixed-id handwritten shelf along with everything else, so each one starts
  from a shelf that does not exist yet;
- the end-to-end test commits (a worker runs in its own session and cannot see
  a savepoint), so it cleans the shelf up by hand in a ``finally``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import (
    get_embedding_provider,
    get_reranker_provider,
    get_session,
)
from rag_recipes.api.dependencies import (
    get_settings as get_settings_dep,
)
from rag_recipes.config import get_settings
from rag_recipes.ingestion.jobs import index_knowledge_item
from rag_recipes.ingestion.manual import (
    MANUAL_SHELF_DOCUMENT_ID,
    MANUAL_SHELF_TITLE,
)
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.storage.enums import (
    DocumentStatus,
    KnowledgeItemStatus,
    SourceType,
)
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.session import build_session_factory
from tests.integration.conftest import AUTH_HEADERS, TEST_API_TOKEN

pytestmark = pytest.mark.asyncio

_FAKE_PROVIDER = "fake"
_FAKE_MODEL = "fake-embedding"

_RECIPE: dict[str, Any] = {
    "title": "Roast Tomato Soup",
    "summary": "A soup that tastes of the oven.",
    "yield": "Serves 4",
    "prep_time": "10 minutes",
    "cook_time": "40 minutes",
    "total_time": "50 minutes",
    "ingredients": [
        "500 g ripe tomatoes",
        "2 cloves garlic",
        "3 tbsp olive oil",
    ],
    "steps": [
        "Halve the tomatoes and salt them well.",
        "Roast at 200C with the garlic and oil until collapsing.",
        "Blitz smooth, then taste again for salt.",
    ],
}


@pytest.fixture
def client(
    db_session: AsyncSession,
    override_settings_with_token: None,
    auth_headers: dict[str, str],
) -> AsyncIterator[httpx.AsyncClient]:
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


async def _create(client: httpx.AsyncClient, **overrides: Any) -> httpx.Response:
    return await client.post("/api/v1/knowledge-items", json={**_RECIPE, **overrides})


async def test_a_typed_recipe_is_shelved_and_queued_for_indexing(
    client: httpx.AsyncClient, fake_arq_redis: AsyncMock
) -> None:
    response = await _create(client)

    assert response.status_code == 201, response.text
    item = response.json()["knowledge_item"]
    assert item["title"] == "Roast Tomato Soup"
    assert item["document_id"] == MANUAL_SHELF_DOCUMENT_ID
    # Straight past the review queue: authored, not extracted. `indexing` is
    # the approve path's second half, entered without the first.
    assert item["status"] == KnowledgeItemStatus.INDEXING.value
    assert item["review_reasons"] == []

    fake_arq_redis.enqueue_job.assert_awaited_once_with(
        "index_knowledge_item",
        item["id"],
        _job_id=None,
        _queue_name=None,
        _session_id=MANUAL_SHELF_DOCUMENT_ID,
    )


async def test_the_shelf_is_bootstrapped_with_a_usable_source_chain(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The three rows a ``KnowledgeItem``'s FKs demand, and the two values that
    decide whether the recipe is ever findable: the document's
    ``active_source_version`` and the run's agreement with it."""
    response = await _create(client)
    assert response.status_code == 201, response.text

    document = await db_session.get(Document, MANUAL_SHELF_DOCUMENT_ID)
    assert document is not None
    assert document.title == MANUAL_SHELF_TITLE
    assert document.source_type is SourceType.MANUAL
    assert document.status is DocumentStatus.READY
    assert document.active_source_version == 1

    asset = await db_session.get(SourceAsset, document.asset_id)
    assert asset is not None
    assert asset.source_type is SourceType.MANUAL

    item = await db_session.get(KnowledgeItem, response.json()["knowledge_item"]["id"])
    assert item is not None
    # Search requires source_version == active_source_version; the composite FK
    # requires the run to agree with both.
    assert item.source_version == document.active_source_version
    run = await db_session.get(ExtractionRun, item.extraction_run_id)
    assert run is not None
    assert run.document_id == document.id
    assert run.source_version == item.source_version
    assert run.provider == "manual"


async def test_the_shelf_is_created_once_and_shared(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Not one document per recipe: the library lists documents, so that would
    fill the shelf with one-recipe spines."""
    first = await _create(client)
    second = await _create(client, title="Braised Fennel")

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert (
        first.json()["knowledge_item"]["document_id"]
        == second.json()["knowledge_item"]["document_id"]
    )

    documents = (
        (
            await db_session.execute(
                select(Document.id).where(Document.source_type == SourceType.MANUAL)
            )
        )
        .scalars()
        .all()
    )
    assert list(documents) == [MANUAL_SHELF_DOCUMENT_ID]

    runs = (
        (
            await db_session.execute(
                select(ExtractionRun.id).where(
                    ExtractionRun.document_id == MANUAL_SHELF_DOCUMENT_ID
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(list(runs)) == 1


async def test_provenance_is_left_empty_rather_than_invented(
    client: httpx.AsyncClient,
) -> None:
    """No page was read, so there is no citation to make and no snapshot of an
    earlier extraction to keep."""
    body = (await _create(client)).json()

    assert body["knowledge_item"]["source_span_ids"] == []
    assert body["source_citations"] == []
    # No "· page 22" tail, because there is no page.
    assert body["display"]["subtitle"] == MANUAL_SHELF_TITLE
    assert body["knowledge_item"]["edited_at"] is None
    assert body["knowledge_item"]["favourited_at"] is None


async def test_the_created_payload_is_human_authored_recipe_v1(
    client: httpx.AsyncClient,
) -> None:
    structured = (await _create(client)).json()["knowledge_item"]["structured_data"]

    assert structured["schema"] == "recipe.v1"
    assert structured["yield"] == "Serves 4"
    assert [row["raw_text"] for row in structured["ingredients"]] == _RECIPE[
        "ingredients"
    ]
    assert [row["step_number"] for row in structured["steps"]] == [1, 2, 3]
    # The edit layer's human-authored row shape, reached by construction rather
    # than by a second implementation of it.
    assert structured["ingredients"][0]["edited"] is True
    assert structured["ingredients"][0]["confidence"]["overall"] == 1.0
    assert structured["warnings"] == []


async def test_the_201_envelope_is_the_detail_envelope(
    client: httpx.AsyncClient,
) -> None:
    """Byte-identical to ``GET``, for the same reason the edit ``PATCH`` is: a
    client that renders the created recipe must not need a second shape."""
    created = await _create(client)
    fetched = await client.get(
        f"/api/v1/knowledge-items/{created.json()['knowledge_item']['id']}"
    )

    assert fetched.status_code == 200, fetched.text
    assert created.json() == fetched.json()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"title": "   "}, "whitespace-only title"),
        ({"title": ""}, "empty title"),
        ({"ingredients": ["500 g tomatoes", "  "]}, "blank ingredient line"),
        ({"steps": [""]}, "blank step line"),
        ({"confidence": {"overall": 1.0}}, "machine-owned field"),
        ({"status": "ready"}, "unknown field"),
    ],
)
async def test_unusable_bodies_are_rejected(
    client: httpx.AsyncClient, overrides: dict[str, Any], reason: str
) -> None:
    response = await _create(client, **overrides)

    assert response.status_code == 422, f"{reason}: {response.text}"


async def test_a_title_alone_is_enough(client: httpx.AsyncClient) -> None:
    """``PATCH`` accepts emptying both lists, so a create rule the edit rule
    does not share would only be a trap on the way in."""
    response = await client.post(
        "/api/v1/knowledge-items", json={"title": "Something I'll finish later"}
    )

    assert response.status_code == 201, response.text
    structured = response.json()["knowledge_item"]["structured_data"]
    assert structured["ingredients"] == []
    assert structured["steps"] == []


async def test_reprocessing_the_handwritten_shelf_is_refused(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """There is no PDF to re-extract. Requeuing would run the pipeline over
    nothing and — on the new-source-version path — supersede every hand-typed
    recipe in favour of a generation with no items."""
    assert (await _create(client)).status_code == 201

    response = await client.post(
        f"/api/v1/documents/{MANUAL_SHELF_DOCUMENT_ID}/reprocess",
        json={"mode": "auto", "reason": "curiosity"},
    )

    # 400, not 409: waiting does not make it possible.
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_request"
    document = await db_session.get(Document, MANUAL_SHELF_DOCUMENT_ID)
    assert document is not None
    assert document.status is DocumentStatus.READY


async def test_deleting_the_handwritten_shelf_is_refused(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """One 204 on the shelf would drop every hand-typed recipe at once;
    recipes on it go through DELETE /knowledge-items/{item_id} one by one."""
    created = await _create(client)
    assert created.status_code == 201
    item_id = created.json()["knowledge_item"]["id"]

    response = await client.delete(f"/api/v1/documents/{MANUAL_SHELF_DOCUMENT_ID}")

    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_request"
    document = await db_session.get(Document, MANUAL_SHELF_DOCUMENT_ID)
    assert document is not None
    assert document.status is DocumentStatus.READY
    # The per-recipe path still works.
    assert (await client.delete(f"/api/v1/knowledge-items/{item_id}")).status_code == 204


async def _cleanup_shelf(test_engine: AsyncEngine) -> None:
    """Drop the committed shelf and everything on it, in FK-safe order."""
    session_factory = build_session_factory(test_engine)
    async with session_factory() as session:
        document = await session.get(Document, MANUAL_SHELF_DOCUMENT_ID)
        if document is None:
            return
        chunk_ids = select(Chunk.id).where(Chunk.document_id == MANUAL_SHELF_DOCUMENT_ID)
        await session.execute(
            delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids))
        )
        for stmt in (
            delete(Chunk).where(Chunk.document_id == MANUAL_SHELF_DOCUMENT_ID),
            delete(KnowledgeItem).where(
                KnowledgeItem.document_id == MANUAL_SHELF_DOCUMENT_ID
            ),
            delete(ExtractionRun).where(
                ExtractionRun.document_id == MANUAL_SHELF_DOCUMENT_ID
            ),
            delete(Document).where(Document.id == MANUAL_SHELF_DOCUMENT_ID),
            delete(SourceAsset).where(SourceAsset.id == document.asset_id),
        ):
            await session.execute(stmt)
        await session.commit()


async def test_a_typed_recipe_reaches_the_shelf_and_search(
    test_engine: AsyncEngine,
) -> None:
    """End-to-end: POST → indexing → ``index_knowledge_item`` → ready and findable.

    The assertion that matters is the last one. Everything upstream of it can be
    correct while the recipe stays invisible, because search additionally
    demands ``source_version == active_source_version`` — the value
    ``ensure_manual_shelf`` sets up front precisely so this holds.

    Committed rows (the job opens its own session), the job called directly
    rather than through a burst worker — the queue round trip is already
    covered by ``test_index_knowledge_item_job``.
    """
    session_factory = build_session_factory(test_engine)
    settings = get_settings().model_copy(
        update={
            "personal_api_token": TEST_API_TOKEN,
            "embedding_provider": _FAKE_PROVIDER,
            "embedding_model": _FAKE_MODEL,
        }
    )
    fake = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_settings_dep] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: fake
    app.dependency_overrides[get_reranker_provider] = lambda: None
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=AUTH_HEADERS
        ) as client:
            created = await _create(client)
            assert created.status_code == 201, created.text
            item_id = created.json()["knowledge_item"]["id"]

            indexed = await index_knowledge_item(
                {
                    "settings": settings,
                    "session_factory": session_factory,
                    "embedding_provider": fake,
                },
                item_id,
            )
            assert indexed >= 1

            async with session_factory() as session:
                item = await session.get(KnowledgeItem, item_id)
                assert item is not None
                assert item.status is KnowledgeItemStatus.READY
                chunk_types = sorted(
                    row[0]
                    for row in (
                        await session.execute(
                            select(Chunk.chunk_type).where(Chunk.parent_id == item_id)
                        )
                    ).all()
                )
                # The full set a complete recipe earns — the ingredients and
                # steps chunks are built from the text blocks the create path
                # derived, so an empty one here would mean they never landed.
                assert chunk_types == [
                    "recipe_full",
                    "recipe_ingredients",
                    "recipe_steps",
                    "recipe_summary",
                    "recipe_title",
                ]
                # The handoff had nothing to do: the shelf was already on the
                # item's version, and a second manual recipe must not supersede
                # the first.
                document = await session.get(Document, MANUAL_SHELF_DOCUMENT_ID)
                assert document is not None
                assert document.active_source_version == item.source_version

            search = await client.post(
                "/api/v1/search",
                json={"query": "roast tomato soup", "mode": "keyword"},
            )
            assert search.status_code == 200, search.text
            assert item_id in [r["item"]["id"] for r in search.json()["results"]]
    finally:
        for dep in (
            get_session,
            get_settings_dep,
            get_embedding_provider,
            get_reranker_provider,
        ):
            app.dependency_overrides.pop(dep, None)
        await _cleanup_shelf(test_engine)
