"""App-level exception handler envelopes (doc 6 §error model)."""

from __future__ import annotations

import httpx
import pytest

from rag_recipes.api.app import app
from rag_recipes.api.errors import ApiError, ErrorCode


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


@app.get("/__test__/raises-api-error")
async def _raise_api_error() -> None:
    raise ApiError(
        status_code=418,
        code=ErrorCode.INVALID_REQUEST,
        message="I am a teapot.",
        details={"field": "kettle"},
    )


@app.get("/__test__/raises-unexpected")
async def _raise_unexpected() -> None:
    raise RuntimeError("internal secret should not leak")


@app.post("/__test__/needs-json")
async def _needs_json(payload: dict[str, str]) -> dict[str, str]:
    return payload


async def test_api_error_renders_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/__test__/raises-api-error")
    assert response.status_code == 418
    assert response.json() == {
        "error": {
            "code": "invalid_request",
            "message": "I am a teapot.",
            "details": {"field": "kettle"},
        }
    }


async def test_unhandled_exception_returns_500_envelope(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", headers=auth_headers
    ) as c:
        response = await c.get("/__test__/raises-unexpected")
    assert response.status_code == 500
    body = response.json()
    assert body == {
        "error": {
            "code": "internal_error",
            "message": "Unexpected server error.",
            "details": {},
        }
    }
    assert "internal secret" not in response.text


async def test_request_validation_error_returns_422_envelope(
    client: httpx.AsyncClient,
) -> None:
    async with client:
        response = await client.post(
            "/__test__/needs-json",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["message"] == "Request validation failed."
    assert "errors" in body["error"]["details"]
    assert isinstance(body["error"]["details"]["errors"], list)
