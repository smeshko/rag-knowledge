"""Integration assertions for the fail-closed bearer-token gate (Phase 6.3).

Every endpoint, including `/health`, requires a valid bearer token. With
``personal_api_token`` unset, every request 401s (fail-closed).
"""

from __future__ import annotations

import httpx
import pytest

from rag_recipes.api.app import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> httpx.AsyncClient:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def test_health_401s_without_token_when_token_unset(
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_documents_route_401s_without_token_when_token_unset(
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.get("/api/v1/documents")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_health_401s_without_token_when_token_configured(
    override_settings_with_token: None,
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_documents_route_401s_with_wrong_token_when_token_configured(
    override_settings_with_token: None,
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.get(
            "/api/v1/documents",
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_documents_route_401s_without_token_when_token_configured(
    override_settings_with_token: None,
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.get("/api/v1/documents")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


# The valid-token-passes path is covered by:
#   - tests/unit/api/test_auth_dependency.py (direct dependency call).
#   - tests/integration/test_health.py (sends a valid bearer header).
#   - the full Phase 6.1/6.2/6.3 documents integration suite (every test sends
#     ``AUTH_HEADERS`` and the routes return 2xx/4xx business statuses,
#     never 401).
