"""Integration tests for POST /api/v1/menus.

Real end-to-end against the Epic 12 retrieval facade via ``db_session``, with the
embedding provider and settings overridden as in ``test_search.py`` and the LLM
replaced by a scripted double that answers the planner and the selection calls
differently (the menu layer makes two LLM calls with different schemas, which the
single-response ``FakeLLMProvider`` cannot express). Recipe seeding reuses the
``test_search.py`` helpers so retrieval parity is exact.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import (
    get_embedding_provider,
    get_llm_provider,
    get_reranker_provider,
    get_session,
    get_settings,
)
from rag_recipes.menus.schema import MENU_PLAN_SCHEMA_VERSION, MENU_SELECTION_SCHEMA_VERSION
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputResponse
from tests.integration.conftest import AUTH_HEADERS, TEST_API_TOKEN
from tests.integration.test_search import (
    _FAKE_MODEL,
    _FAKE_PROVIDER,
    _make_document,
    _make_recipe,
    _make_span,
)

pytestmark = pytest.mark.asyncio

_REQUEST = "a light salad, a hearty casserole and a chocolate no-bake dessert"

_SALAD_QUERY = "light green salad lemon vinaigrette"
_MAIN_QUERY = "hearty baked vegetable casserole"
_DESSERT_QUERY = "no-bake chocolate mousse"


def _ScriptedLLM(  # noqa: N802 — reads as the double it replaces
    *,
    plan: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
    fail_plan: bool = False,
    fail_selection: bool = False,
) -> FakeLLMProvider:
    """The plan and selection calls answered separately, keyed on ``schema_version``.

    ``fail_plan`` / ``fail_selection`` raise ``LLMTechnicalError`` for that one call
    so each safe-fallback path can be exercised independently. Built on the shared
    ``FakeLLMProvider`` (tests/AGENTS.md: extend the fake, don't fork it).
    """
    responses: dict[str, dict[str, Any] | StructuredOutputResponse | LLMTechnicalError] = {}
    if fail_plan:
        responses[MENU_PLAN_SCHEMA_VERSION] = LLMTechnicalError("scripted plan failure")
    elif plan is not None:
        responses[MENU_PLAN_SCHEMA_VERSION] = plan
    if fail_selection:
        responses[MENU_SELECTION_SCHEMA_VERSION] = LLMTechnicalError("scripted selection failure")
    elif selection is not None:
        responses[MENU_SELECTION_SCHEMA_VERSION] = selection
    return FakeLLMProvider(responses_by_schema_version=responses)


_PLAN = {
    "theme": "A light-to-rich autumn dinner",
    "courses": [
        {"slot": "salad", "query": _SALAD_QUERY, "note": "light"},
        {"slot": "main", "query": _MAIN_QUERY, "note": "hearty"},
        {"slot": "dessert", "query": _DESSERT_QUERY, "note": "no-bake, chocolate"},
    ],
}


@asynccontextmanager
async def _client(
    db_session: AsyncSession,
    llm: FakeLLMProvider,
    *,
    debug_enabled: bool = False,
    with_auth: bool = True,
) -> AsyncIterator[httpx.AsyncClient]:
    settings = get_settings().model_copy(
        update={
            "personal_api_token": TEST_API_TOKEN,
            "embedding_provider": _FAKE_PROVIDER,
            "embedding_model": _FAKE_MODEL,
            "debug_endpoints_enabled": debug_enabled,
        }
    )
    fake_embed = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: fake_embed
    app.dependency_overrides[get_llm_provider] = lambda: llm
    app.dependency_overrides[get_reranker_provider] = lambda: None
    transport = httpx.ASGITransport(app=app)
    headers = AUTH_HEADERS if with_auth else {}
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=headers
        ) as client:
            yield client
    finally:
        for dep in (
            get_session,
            get_settings,
            get_embedding_provider,
            get_llm_provider,
            get_reranker_provider,
        ):
            app.dependency_overrides.pop(dep, None)


async def _seed_three_courses(session: AsyncSession) -> dict[str, str]:
    """Seed one recipe per course, each embedded on its own course query."""
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    doc = await _make_document(session)
    ids: dict[str, str] = {}
    for slot, title, chunk_text, query, page in (
        (
            "salad",
            "Butter Lettuce Salad",
            "light green salad with lemon vinaigrette",
            _SALAD_QUERY,
            10,
        ),
        (
            "main",
            "Baked Vegetable Casserole",
            "a hearty baked vegetable casserole",
            _MAIN_QUERY,
            20,
        ),
        ("dessert", "Chocolate Chia Mousse", "a no-bake chocolate mousse", _DESSERT_QUERY, 30),
    ):
        span = await _make_span(session, doc, page=page)
        ids[slot] = await _make_recipe(
            session,
            doc,
            title=title,
            chunk_text=chunk_text,
            embed_text=query,
            provider=provider,
            span_ids=[span],
        )
    return ids


def _selection_for(ids: dict[str, str]) -> dict[str, Any]:
    """A valid selection. cite ids follow pack order: salad, main, dessert."""
    return {
        "menu": {
            "title": "Autumn Dinner",
            "text": "The salad stays light before the casserole; the mousse needs no oven.",
            "citations": ["cite_1", "cite_2"],
        },
        "courses": [
            {
                "slot": "salad",
                "knowledge_item_id": ids["salad"],
                "reason": "Keeps it light",
                "citation_ids": ["cite_1"],
            },
            {
                "slot": "main",
                "knowledge_item_id": ids["main"],
                "reason": "The hearty centre",
                "citation_ids": ["cite_2"],
            },
            {
                "slot": "dessert",
                "knowledge_item_id": ids["dessert"],
                "reason": "No oven needed",
                "citation_ids": ["cite_3"],
            },
        ],
    }


async def test_menu_happy_path_fills_every_course(db_session: AsyncSession) -> None:
    ids = await _seed_three_courses(db_session)
    llm = _ScriptedLLM(plan=_PLAN, selection=_selection_for(ids))
    async with _client(db_session, llm) as client:
        resp = await client.post("/api/v1/menus", json={"query": _REQUEST})

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == _REQUEST
    assert body["theme"] == "A light-to-rich autumn dinner"
    assert body["warnings"] == []
    assert body["menu"]["title"] == "Autumn Dinner"

    assert [c["slot"] for c in body["courses"]] == ["salad", "main", "dessert"]
    picked = {c["slot"]: c["selection"]["knowledge_item_id"] for c in body["courses"]}
    assert picked == ids
    # Titles are reconstructed from the pack, not taken from the model.
    titles = {c["slot"]: c["selection"]["title"] for c in body["courses"]}
    assert titles["dessert"] == "Chocolate Chia Mousse"
    # Every course is filled by a different recipe.
    assert len(set(picked.values())) == 3


async def test_each_course_is_searched_with_its_own_planned_query(
    db_session: AsyncSession,
) -> None:
    """The whole point of the layer: three retrievals, not one blurred query."""
    ids = await _seed_three_courses(db_session)
    llm = _ScriptedLLM(plan=_PLAN, selection=_selection_for(ids))
    async with _client(db_session, llm, debug_enabled=True) as client:
        resp = await client.post(
            "/api/v1/menus", json={"query": _REQUEST, "menu": {"include_debug": True}}
        )

    body = resp.json()
    assert [c["query"] for c in body["courses"]] == [_SALAD_QUERY, _MAIN_QUERY, _DESSERT_QUERY]
    debug = body["debug"]
    assert debug["course_count"] == 3
    assert debug["plan_is_fallback"] is False
    assert set(debug["retrieval_debug_by_slot"]) == {"salad", "main", "dessert"}
    assert debug["retrieval_debug_by_slot"]["main"]["normalized_query"] == _MAIN_QUERY


async def test_citations_are_reconstructed_from_the_pack(db_session: AsyncSession) -> None:
    ids = await _seed_three_courses(db_session)
    llm = _ScriptedLLM(plan=_PLAN, selection=_selection_for(ids))
    async with _client(db_session, llm) as client:
        resp = await client.post("/api/v1/menus", json={"query": _REQUEST})

    citations = resp.json()["citations"]
    assert citations, "a grounded menu must expose its citation detail"
    by_id = {c["citation_id"]: c for c in citations}
    # Every cited id resolves to a real span with a page label, bound to one item.
    for cite in citations:
        assert cite["source_span_id"]
        assert cite["label"].startswith("page ")
    assert by_id["cite_1"]["knowledge_item_id"] == ids["salad"]


async def test_misattributed_citation_falls_back_to_best_per_course(
    db_session: AsyncSession,
) -> None:
    """A cross-item citation is a grounding failure — never surfaced as an answer."""
    ids = await _seed_three_courses(db_session)
    selection = _selection_for(ids)
    selection["courses"][0]["citation_ids"] = ["cite_3"]  # dessert's span for the salad
    llm = _ScriptedLLM(plan=_PLAN, selection=selection)
    async with _client(db_session, llm) as client:
        resp = await client.post("/api/v1/menus", json={"query": _REQUEST})

    assert resp.status_code == 200
    body = resp.json()
    assert body["warnings"], "a fallback must say so"
    # The menu shape survives — each course still names its best candidate — but
    # nothing is presented as grounded.
    assert [c["slot"] for c in body["courses"]] == ["salad", "main", "dessert"]
    assert all(c["selection"] is not None for c in body["courses"])
    assert all(c["selection"]["citation_ids"] == [] for c in body["courses"])
    assert all(c["selection"]["reason"] == "" for c in body["courses"])
    assert body["citations"] == []


async def test_serving_one_dish_in_two_courses_falls_back(db_session: AsyncSession) -> None:
    ids = await _seed_three_courses(db_session)
    selection = _selection_for(ids)
    selection["courses"][1]["knowledge_item_id"] = ids["salad"]
    selection["courses"][1]["citation_ids"] = ["cite_1"]
    llm = _ScriptedLLM(plan=_PLAN, selection=selection)
    async with _client(db_session, llm) as client:
        resp = await client.post("/api/v1/menus", json={"query": _REQUEST})

    assert resp.json()["warnings"]


async def test_selection_failure_is_a_fallback_not_a_502(db_session: AsyncSession) -> None:
    ids = await _seed_three_courses(db_session)
    llm = _ScriptedLLM(plan=_PLAN, fail_selection=True)
    async with _client(db_session, llm) as client:
        resp = await client.post("/api/v1/menus", json={"query": _REQUEST})

    assert resp.status_code == 200
    body = resp.json()
    assert body["warnings"]
    assert {c["selection"]["knowledge_item_id"] for c in body["courses"]} == set(ids.values())


async def test_planner_failure_degrades_to_a_single_course_search(
    db_session: AsyncSession,
) -> None:
    """With no plan, a menu request must behave exactly like POST /search."""
    await _seed_three_courses(db_session)
    llm = _ScriptedLLM(fail_plan=True, fail_selection=True)
    async with _client(db_session, llm, debug_enabled=True) as client:
        resp = await client.post(
            "/api/v1/menus", json={"query": _REQUEST, "menu": {"include_debug": True}}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["courses"]) == 1
    assert body["courses"][0]["query"] == _REQUEST
    assert body["debug"]["plan_is_fallback"] is True
    assert any("could not be split into courses" in w for w in body["warnings"])


async def test_candidates_are_omitted_by_default_and_returned_on_request(
    db_session: AsyncSession,
) -> None:
    ids = await _seed_three_courses(db_session)
    llm = _ScriptedLLM(plan=_PLAN, selection=_selection_for(ids))
    async with _client(db_session, llm) as client:
        default = await client.post("/api/v1/menus", json={"query": _REQUEST})
        expanded = await client.post(
            "/api/v1/menus",
            json={"query": _REQUEST, "menu": {"include_candidates": True}},
        )

    assert all(c["candidates"] == [] for c in default.json()["courses"])
    expanded_courses = expanded.json()["courses"]
    assert any(c["candidates"] for c in expanded_courses)
    first = next(c for c in expanded_courses if c["candidates"])["candidates"][0]
    assert first["type"] == "knowledge_item_result"
    assert first["item"]["title"]


async def test_debug_is_absent_unless_opted_in_and_enabled(db_session: AsyncSession) -> None:
    ids = await _seed_three_courses(db_session)
    llm = _ScriptedLLM(plan=_PLAN, selection=_selection_for(ids))
    async with _client(db_session, llm, debug_enabled=False) as client:
        gated = await client.post(
            "/api/v1/menus", json={"query": _REQUEST, "menu": {"include_debug": True}}
        )
    assert "debug" not in gated.json()

    async with _client(db_session, llm, debug_enabled=True) as client:
        not_asked = await client.post("/api/v1/menus", json={"query": _REQUEST})
    assert "debug" not in not_asked.json()


async def test_no_results_reports_every_course_unfilled(db_session: AsyncSession) -> None:
    llm = _ScriptedLLM(plan=_PLAN, selection={})
    async with _client(db_session, llm) as client:
        resp = await client.post("/api/v1/menus", json={"query": _REQUEST})

    body = resp.json()
    assert [c["slot"] for c in body["courses"]] == ["salad", "main", "dessert"]
    assert all(c["selection"] is None for c in body["courses"])
    assert any("No relevant results" in w for w in body["warnings"])


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"query": "   "}, "query"),
        ({"query": "x", "retrieval": {"mode": "telepathy"}}, "mode"),
    ],
)
async def test_invalid_requests_use_the_doc6_error_envelope(
    db_session: AsyncSession, payload: dict[str, Any], field: str
) -> None:
    llm = _ScriptedLLM(plan=_PLAN, selection={})
    async with _client(db_session, llm) as client:
        resp = await client.post("/api/v1/menus", json=payload)

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["details"]["field"] == field


async def test_menus_requires_auth(db_session: AsyncSession) -> None:
    llm = _ScriptedLLM(plan=_PLAN, selection={})
    async with _client(db_session, llm, with_auth=False) as client:
        resp = await client.post("/api/v1/menus", json={"query": _REQUEST})
    assert resp.status_code == 401
