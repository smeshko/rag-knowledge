"""Integration tests for GET /api/v1/knowledge-items/{item_id} (doc 6 § 8)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository

_FULL_STRUCTURED: dict[str, Any] = {
    "schema": "recipe.v1",
    "yield": "Serves 4",
    "cook_time": "35 minutes",
    "ingredients": [
        {"raw_text": "1 cup white beans", "item_normalized": "white beans"},
        {"raw_text": "2 cups stock", "item_normalized": "vegetable stock"},
        {"raw_text": "1 onion", "item_normalized": "onion"},
    ],
    "steps": [
        {"text": "Soak the beans.", "source_span_ids": ["span_a"]},
        {"text": "Simmer with stock.", "source_span_ids": ["span_b"]},
    ],
    "ingredients_text": "white beans, stock, onion",
    "an_unknown_future_key": {"nested": [1, 2, 3]},
    "warnings": [],
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


async def _seed_item(
    session: AsyncSession,
    *,
    status: KnowledgeItemStatus = KnowledgeItemStatus.READY,
    with_spans: bool = True,
    structured_data: dict[str, Any] | None = None,
) -> str:
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
    doc = await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="Simple Thai Food",
        author="Leela",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=1,
        status=DocumentStatus.READY,
    )
    span_ids: list[str] = []
    if with_spans:
        for page in (42, 43):
            locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
            span = SourceSpan(
                id=new_id("span"),
                document_id=doc.id,
                source_version=1,
                source_type=SourceType.PDF,
                locator=locator,
                locator_hash=hashlib.sha256(f"{doc.id}-{page}".encode()).hexdigest(),
                text="recipe text",
                text_hash=hashlib.sha256(f"t{page}".encode()).hexdigest(),
            )
            session.add(span)
            await session.flush()
            span_ids.append(span.id)
    run = ExtractionRun(
        document_id=doc.id,
        source_version=1,
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_source_span_ids=[],
        input_hash=new_id("hash"),
        status=ExtractionRunStatus.SUCCESS,
        output_json=None,
    )
    session.add(run)
    await session.flush()
    item = KnowledgeItem(
        document_id=doc.id,
        extraction_run_id=run.id,
        source_version=1,
        item_type="recipe",
        title="Cozy White Bean Soup",
        normalized_title="cozy white bean soup",
        summary="A cozy bowl of white beans.",
        body_text="x" * 100,
        source_span_ids=span_ids,
        structured_data=structured_data if structured_data is not None else _FULL_STRUCTURED,
        confidence={"overall": 0.88},
        status=status,
    )
    session.add(item)
    await session.flush()
    return item.id


@pytest.mark.asyncio
async def test_returns_full_structured_data_display_and_citations(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    item_id = await _seed_item(db_session)
    async with client:
        resp = await client.get(f"/api/v1/knowledge-items/{item_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ki = body["knowledge_item"]
    assert ki["id"] == item_id
    assert ki["status"] == "ready"
    assert ki["confidence"] == {"overall": 0.88}
    # Full structured_data round-trips: every ingredient/step + unknown keys survive.
    sd = ki["structured_data"]
    assert len(sd["ingredients"]) == 3
    assert len(sd["steps"]) == 2
    assert sd["steps"][1]["source_span_ids"] == ["span_b"]
    assert sd["an_unknown_future_key"] == {"nested": [1, 2, 3]}
    assert sd["schema"] == "recipe.v1"
    # display + citations.
    assert body["display"]["title"] == "Cozy White Bean Soup"
    assert body["display"]["subtitle"] == "Simple Thai Food · page 42"
    labels = [c["label"] for c in body["source_citations"]]
    assert labels == ["page 42", "page 43"]
    assert body["source_citations"][0]["locator"]["page_start"] == 42
    # Ready items carry an empty review_reasons list (Epic 21.1, additive).
    assert ki["review_reasons"] == []


@pytest.mark.asyncio
async def test_needs_review_item_carries_review_reasons(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """A needs_review item projects its persisted warning codes as
    {code, message} review_reasons (Epic 21.1, D3)."""
    structured = dict(_FULL_STRUCTURED)
    structured["warnings"] = ["no_ingredients", "low_overall_confidence"]
    item_id = await _seed_item(
        db_session,
        status=KnowledgeItemStatus.NEEDS_REVIEW,
        structured_data=structured,
    )
    async with client:
        resp = await client.get(f"/api/v1/knowledge-items/{item_id}")
    assert resp.status_code == 200, resp.text
    ki = resp.json()["knowledge_item"]
    assert ki["status"] == "needs_review"
    reasons = ki["review_reasons"]
    # Codes match the persisted warnings verbatim, with non-empty messages.
    assert [r["code"] for r in reasons] == ["no_ingredients", "low_overall_confidence"]
    assert all(r["message"] for r in reasons)
    # The raw codes still round-trip verbatim in structured_data.
    assert ki["structured_data"]["warnings"] == [
        "no_ingredients",
        "low_overall_confidence",
    ]


@pytest.mark.asyncio
async def test_needs_review_unknown_warning_projects_llm_warning_envelope(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    structured = dict(_FULL_STRUCTURED)
    structured["warnings"] = ["model said something odd"]
    item_id = await _seed_item(
        db_session,
        status=KnowledgeItemStatus.NEEDS_REVIEW,
        structured_data=structured,
    )
    async with client:
        resp = await client.get(f"/api/v1/knowledge-items/{item_id}")
    assert resp.status_code == 200, resp.text
    reasons = resp.json()["knowledge_item"]["review_reasons"]
    assert reasons == [
        {"code": "llm_warning", "message": "model said something odd"}
    ]


@pytest.mark.asyncio
async def test_unknown_id_returns_404_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        resp = await client.get("/api/v1/knowledge-items/ki_does_not_exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "knowledge_item_not_found"
    assert body["error"]["details"]["item_id"] == "ki_does_not_exist"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [KnowledgeItemStatus.SUPERSEDED, KnowledgeItemStatus.NEEDS_REVIEW],
)
async def test_returns_item_regardless_of_status(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    status: KnowledgeItemStatus,
) -> None:
    item_id = await _seed_item(db_session, status=status)
    async with client:
        resp = await client.get(f"/api/v1/knowledge-items/{item_id}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["knowledge_item"]["status"] == status.value


@pytest.mark.asyncio
async def test_item_without_spans_has_no_citations(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    item_id = await _seed_item(db_session, with_spans=False)
    async with client:
        resp = await client.get(f"/api/v1/knowledge-items/{item_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source_citations"] == []
    # subtitle degrades to the document title alone.
    assert body["display"]["subtitle"] == "Simple Thai Food"


@pytest.mark.asyncio
async def test_missing_token_returns_401(
    db_session: AsyncSession, override_settings_with_token: None
) -> None:
    item_id = await _seed_item(db_session)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            resp = await c.get(f"/api/v1/knowledge-items/{item_id}")
    finally:
        app.dependency_overrides.pop(get_session, None)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
