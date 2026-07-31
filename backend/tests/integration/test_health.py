import httpx
import pytest

from rag_recipes.api.app import app


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    override_settings_with_token: None,
    auth_headers: dict[str, str],
) -> httpx.AsyncClient:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(
        transport=transport, base_url="http://testserver", headers=auth_headers
    )


async def test_health_endpoint_returns_ok(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
