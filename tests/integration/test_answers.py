"""Integration tests for POST /api/v1/answers (doc 8).

Real end-to-end against the Epic 12 retrieval facade via ``db_session``, with the
embedding provider, settings, and (new here) the answer ``get_llm_provider``
overridden by a ``FakeLLMProvider`` returning a canned ``answer.v1`` payload. Recipe
seeding reuses the ``test_search.py`` helpers so retrieval parity is exact.
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
    get_session,
    get_settings,
)
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.llm.fake import FakeLLMProvider
from tests.integration.conftest import AUTH_HEADERS, TEST_API_TOKEN
from tests.integration.test_search import (
    _FAKE_MODEL,
    _FAKE_PROVIDER,
    _QUERY,
    _seed_one_recipe,
)

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _client(
    db_session: AsyncSession,
    *,
    llm_response: dict[str, Any] | None = None,
    fail_technically: bool = False,
    with_auth: bool = True,
    debug_enabled: bool = False,
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
    fake_llm = (
        FakeLLMProvider(fail_technically=True)
        if fail_technically
        else FakeLLMProvider(default_output=llm_response)
    )

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: fake_embed
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    transport = httpx.ASGITransport(app=app)
    headers = AUTH_HEADERS if with_auth else {}
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=headers
        ) as client:
            yield client
    finally:
        for dep in (get_session, get_settings, get_embedding_provider, get_llm_provider):
            app.dependency_overrides.pop(dep, None)


def _valid_payload(item_id: str) -> dict[str, Any]:
    # The pack assigns cite_1 to the single seeded span; the model emits deliberately
    # wrong citation detail to prove the backend reconstructs it from the pack.
    return {
        "answer": {
            "style": "recommendation",
            "text": "Try the Cozy White Bean Soup.",
            "citations": ["cite_1"],
        },
        "recommendations": [
            {
                "knowledge_item_id": item_id,
                "reason": "Uses white beans directly.",
                "citation_ids": ["cite_1"],
            }
        ],
        "citations": [{"citation_id": "cite_1", "knowledge_item_id": "WRONG", "label": "x"}],
    }


async def test_answers_happy_path(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    item_id = await _seed_one_recipe(db_session, provider)
    async with _client(db_session, llm_response=_valid_payload(item_id)) as client:
        resp = await client.post(
            "/api/v1/answers",
            json={"query": _QUERY, "answer": {"include_results": True}},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query"] == _QUERY
    assert body["answer"]["style"] == "recommendation"
    assert body["answer"]["text"] == "Try the Cozy White Bean Soup."
    assert body["warnings"] == []
    # Recommendation title reconstructed from the pack, not the LLM.
    rec = body["recommendations"][0]
    assert rec["knowledge_item_id"] == item_id
    assert rec["title"] == "Cozy White Bean Soup"
    # citations[] reconstructed from the pack — item id is the real one, not "WRONG".
    cite = body["citations"][0]
    assert cite["citation_id"] == "cite_1"
    assert cite["knowledge_item_id"] == item_id
    assert cite["label"] == "page 42"
    assert len(body["results"]) == 1


async def test_answers_results_envelope_matches_search(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    item_id = await _seed_one_recipe(db_session, provider)
    async with _client(db_session, llm_response=_valid_payload(item_id)) as client:
        answers_body = (
            await client.post(
                "/api/v1/answers",
                json={"query": _QUERY, "answer": {"include_results": True}},
            )
        ).json()
        search_body = (
            await client.post("/api/v1/search", json={"query": _QUERY, "mode": "hybrid"})
        ).json()
    # /answers.results must be byte-identical to /search.results (shared projector).
    assert answers_body["results"] == search_body["results"]


async def test_answers_invalid_citation_safe_fallback_keeps_results(
    db_session: AsyncSession,
) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    item_id = await _seed_one_recipe(db_session, provider)
    bad = _valid_payload(item_id)
    bad["recommendations"][0]["citation_ids"] = ["cite_999"]  # not in the pack
    # include_results=False, but a fallback must still carry the retrieved results.
    async with _client(db_session, llm_response=bad) as client:
        resp = await client.post(
            "/api/v1/answers",
            json={"query": _QUERY, "answer": {"include_results": False}},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["warnings"]  # non-empty warning
    assert body["recommendations"] == []
    assert len(body["results"]) == 1  # fallback keeps results despite include_results=False


async def test_answers_technical_failure_safe_fallback(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    await _seed_one_recipe(db_session, provider)
    async with _client(db_session, fail_technically=True) as client:
        resp = await client.post(
            "/api/v1/answers",
            json={"query": _QUERY, "answer": {"include_results": True}},
        )
    assert resp.status_code == 200, resp.text  # not a 500
    body = resp.json()
    assert body["warnings"]
    assert len(body["results"]) == 1


async def test_answers_include_results_toggle_keeps_title(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    item_id = await _seed_one_recipe(db_session, provider)
    async with _client(db_session, llm_response=_valid_payload(item_id)) as client:
        with_results = (
            await client.post(
                "/api/v1/answers",
                json={"query": _QUERY, "answer": {"include_results": True}},
            )
        ).json()
        without_results = (
            await client.post(
                "/api/v1/answers",
                json={"query": _QUERY, "answer": {"include_results": False}},
            )
        ).json()
    assert len(with_results["results"]) == 1
    assert without_results["results"] == []  # success path drops results
    # title is reconstructed from the pack and present regardless of include_results.
    assert without_results["recommendations"][0]["title"] == "Cozy White Bean Soup"


async def test_answers_negative_limit_behaves_like_default(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    item_id = await _seed_one_recipe(db_session, provider)
    async with _client(db_session, llm_response=_valid_payload(item_id)) as client:
        resp = await client.post(
            "/api/v1/answers",
            json={
                "query": _QUERY,
                "retrieval": {"limit": -5},
                "answer": {"include_results": True},
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # A negative limit normalizes to the default — no false fallback, results returned.
    assert body["warnings"] == []
    assert len(body["results"]) == 1
    assert body["recommendations"][0]["knowledge_item_id"] == item_id


async def test_answers_unsupported_style_returns_400(db_session: AsyncSession) -> None:
    async with _client(db_session, llm_response={}) as client:
        resp = await client.post(
            "/api/v1/answers",
            json={"query": _QUERY, "answer": {"style": "bogus_style"}},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["details"]["field"] == "style"


async def test_answers_empty_query_returns_400(db_session: AsyncSession) -> None:
    async with _client(db_session, llm_response={}) as client:
        resp = await client.post("/api/v1/answers", json={"query": "   "})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


async def test_answers_bad_mode_returns_400(db_session: AsyncSession) -> None:
    async with _client(db_session, llm_response={}) as client:
        resp = await client.post(
            "/api/v1/answers",
            json={"query": _QUERY, "retrieval": {"mode": "bogus"}},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["details"]["field"] == "mode"


async def test_answers_missing_token_returns_401(db_session: AsyncSession) -> None:
    async with _client(db_session, llm_response={}, with_auth=False) as client:
        resp = await client.post("/api/v1/answers", json={"query": _QUERY})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize(
    ("style", "version"),
    [
        ("recommendation", "answer-recommendation-v2"),
        ("summary", "answer-summary-v2"),
        ("comparison", "answer-comparison-v2"),
        ("direct_answer", "answer-direct-answer-v2"),
    ],
)
async def test_answers_each_style_routes_with_versioned_prompt(
    db_session: AsyncSession, style: str, version: str
) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    item_id = await _seed_one_recipe(db_session, provider)
    async with _client(
        db_session, llm_response=_valid_payload(item_id), debug_enabled=True
    ) as client:
        resp = await client.post(
            "/api/v1/answers",
            json={
                "query": _QUERY,
                "answer": {"style": style, "include_debug": True, "include_results": True},
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"]["style"] == style  # response style is the requested style
    assert len(body["results"]) == 1
    # The per-style versioned prompt flows into the debug payload.
    assert body["debug"]["prompt_version"] == version


async def test_answers_summary_empty_recommendations(db_session: AsyncSession) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    await _seed_one_recipe(db_session, provider)
    # summary may omit recommendations as long as the answer cites ≥1 valid id.
    canned: dict[str, Any] = {
        "answer": {"style": "summary", "text": "A summary of the soup.", "citations": ["cite_1"]},
        "recommendations": [],
        "citations": [],
    }
    async with _client(db_session, llm_response=canned) as client:
        resp = await client.post(
            "/api/v1/answers",
            json={"query": _QUERY, "answer": {"style": "summary"}},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"]["style"] == "summary"
    assert body["recommendations"] == []
    assert body["citations"][0]["citation_id"] == "cite_1"
    assert body["warnings"] == []


@pytest.mark.parametrize(
    ("include_debug", "debug_enabled", "present"),
    [
        (True, True, True),  # only T/T exposes debug
        (True, False, False),
        (False, True, False),  # guards a route gated only on debug_endpoints_enabled
        (False, False, False),
    ],
)
async def test_answers_debug_gate_matrix(
    db_session: AsyncSession, include_debug: bool, debug_enabled: bool, present: bool
) -> None:
    provider = FakeEmbeddingProvider(provider=_FAKE_PROVIDER, model=_FAKE_MODEL)
    item_id = await _seed_one_recipe(db_session, provider)
    async with _client(
        db_session, llm_response=_valid_payload(item_id), debug_enabled=debug_enabled
    ) as client:
        resp = await client.post(
            "/api/v1/answers",
            json={"query": _QUERY, "answer": {"include_debug": include_debug}},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    if present:
        assert "debug" in body
        debug = body["debug"]
        assert debug["retrieval_mode"] == "hybrid"
        assert debug["model"] == "fake-model"
        assert debug["prompt_version"] == "answer-recommendation-v2"
        assert debug["context_item_count"] == 1
        assert debug["citation_count"] == 1
        assert debug["retrieval_debug"]["retrieval_mode"] == "hybrid"
    else:
        assert "debug" not in body  # key absent, not null
