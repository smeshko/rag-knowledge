"""Unit tests for AnthropicBatchProvider.submit_batch (Epic 19.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic
import httpx
import pytest
from anthropic._exceptions import OverloadedError

from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.anthropic_batch import (
    AnthropicBatchProvider,
    BatchExtractionRequest,
)

_REQUEST_OBJ = httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches")

_SCHEMA_WITH_CONSTRAINTS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"n": {"type": "integer", "minimum": 0, "maximum": 10}},
    "required": ["n"],
}


def _request(custom_id: str = "ebitem_1", *, input: str = "extract this") -> BatchExtractionRequest:
    return BatchExtractionRequest(
        custom_id=custom_id,
        input=input,
        model="claude-sonnet-4-6",
        max_tokens=8192,
        json_schema=_SCHEMA_WITH_CONSTRAINTS,
    )


@dataclass
class _FakeBatch:
    id: str = "msgbatch_123"


class _FakeBatchesResource:
    def __init__(
        self, response: _FakeBatch | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeBatch:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _FakeMessagesResource:
    def __init__(self, batches: _FakeBatchesResource) -> None:
        self.batches = batches


class _FakeBatchClient:
    def __init__(
        self, response: _FakeBatch | None = None, error: Exception | None = None
    ) -> None:
        self.batches = _FakeBatchesResource(response, error)
        self.messages = _FakeMessagesResource(self.batches)


def _provider(client: Any) -> AnthropicBatchProvider:
    return AnthropicBatchProvider(api_key="sk-ant-test", client=client)


async def test_submit_batch_maps_requests_and_returns_result() -> None:
    client = _FakeBatchClient(response=_FakeBatch(id="msgbatch_abc"))
    provider = _provider(client)
    result = await provider.submit_batch([_request("ebitem_a"), _request("ebitem_b")])

    assert result.provider_batch_id == "msgbatch_abc"
    assert result.request_count == 2
    # Exactly one create call carrying both requests.
    assert len(client.batches.calls) == 1
    sent = client.batches.calls[0]["requests"]
    assert [r["custom_id"] for r in sent] == ["ebitem_a", "ebitem_b"]
    params = sent[0]["params"]
    assert params["model"] == "claude-sonnet-4-6"
    assert params["max_tokens"] == 8192
    assert params["messages"] == [{"role": "user", "content": "extract this"}]
    fmt = params["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    # _sanitize_schema is reused: unsupported keywords are stripped from the sent schema.
    sent_props = fmt["schema"]["properties"]["n"]
    assert "minimum" not in sent_props and "maximum" not in sent_props


async def test_submit_batch_forwards_idempotency_key() -> None:
    client = _FakeBatchClient(response=_FakeBatch())
    provider = _provider(client)
    await provider.submit_batch([_request()], idempotency_key="ebatch_xyz")

    assert client.batches.calls[0]["extra_headers"] == {"Idempotency-Key": "ebatch_xyz"}


async def test_submit_batch_no_idempotency_key_sends_none() -> None:
    client = _FakeBatchClient(response=_FakeBatch())
    provider = _provider(client)
    await provider.submit_batch([_request()])

    assert client.batches.calls[0]["extra_headers"] is None


def test_default_client_disables_sdk_retries() -> None:
    provider = AnthropicBatchProvider(api_key="sk-ant-test")
    assert provider._client.max_retries == 0


# --- retry / error parity ---------------------------------------------------


class _SequenceBatchesResource:
    def __init__(self, actions: list[Any]) -> None:
        self._actions = list(actions)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeBatch:
        self.calls.append(kwargs)
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        assert isinstance(action, _FakeBatch)
        return action


class _SequenceBatchClient:
    def __init__(self, actions: list[Any]) -> None:
        self.batches = _SequenceBatchesResource(actions)
        self.messages = _FakeMessagesResource(self.batches)  # type: ignore[arg-type]


@pytest.fixture
def _recorded_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(
        "rag_recipes.providers.llm.anthropic_batch.asyncio.sleep", _fake_sleep
    )
    return delays


def _rate_limit() -> anthropic.RateLimitError:
    response = httpx.Response(429, request=_REQUEST_OBJ)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def _overloaded() -> OverloadedError:
    response = httpx.Response(529, request=_REQUEST_OBJ)
    return OverloadedError("overloaded", response=response, body=None)


async def test_rate_limit_retried_then_succeeds(_recorded_sleep: list[float]) -> None:
    client = _SequenceBatchClient([_rate_limit(), _FakeBatch(id="msgbatch_ok")])
    provider = AnthropicBatchProvider(
        api_key="sk-ant-test", client=client, max_rate_limit_retries=5
    )
    result = await provider.submit_batch([_request()])
    assert result.provider_batch_id == "msgbatch_ok"
    assert len(client.batches.calls) == 2
    assert len(_recorded_sleep) == 1


async def test_overloaded_retried_then_succeeds(_recorded_sleep: list[float]) -> None:
    client = _SequenceBatchClient([_overloaded(), _FakeBatch(id="msgbatch_ok")])
    provider = AnthropicBatchProvider(
        api_key="sk-ant-test", client=client, max_rate_limit_retries=5
    )
    result = await provider.submit_batch([_request()])
    assert result.provider_batch_id == "msgbatch_ok"
    assert len(client.batches.calls) == 2


async def test_rate_limit_exhausted_raises(_recorded_sleep: list[float]) -> None:
    client = _SequenceBatchClient([_rate_limit(), _rate_limit(), _rate_limit()])
    provider = AnthropicBatchProvider(
        api_key="sk-ant-test", client=client, max_rate_limit_retries=2
    )
    with pytest.raises(LLMTechnicalError):
        await provider.submit_batch([_request()])
    assert len(client.batches.calls) == 3


async def test_non_retryable_status_raises_immediately(
    _recorded_sleep: list[float],
) -> None:
    status_error = anthropic.APIStatusError(
        "server error",
        response=httpx.Response(500, request=_REQUEST_OBJ),
        body=None,
    )
    client = _SequenceBatchClient([status_error])
    provider = AnthropicBatchProvider(api_key="sk-ant-test", client=client)
    with pytest.raises(LLMTechnicalError):
        await provider.submit_batch([_request()])
    assert len(client.batches.calls) == 1
    assert _recorded_sleep == []


async def test_other_api_error_wrapped() -> None:
    client = _FakeBatchClient(error=anthropic.APITimeoutError(request=_REQUEST_OBJ))
    provider = _provider(client)
    with pytest.raises(LLMTechnicalError):
        await provider.submit_batch([_request()])
