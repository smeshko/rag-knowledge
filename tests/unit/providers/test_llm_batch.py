"""Unit tests for AnthropicBatchProvider — submit (19.2) + retrieve/results (19.3)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx
import pytest
from anthropic._exceptions import OverloadedError

from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.anthropic import _OUTPUT_TOOL_NAME
from rag_recipes.providers.llm.anthropic_batch import (
    AnthropicBatchProvider,
    BatchExtractionRequest,
    BatchResult,
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
    # Batch sends the same non-strict forced tool-use shape as the sync path (no
    # output_config), so neither path compiles the oversized strict grammar.
    assert "output_config" not in params
    tools = params["tools"]
    assert len(tools) == 1 and tools[0]["name"] == _OUTPUT_TOOL_NAME
    assert params["tool_choice"] == {"type": "tool", "name": _OUTPUT_TOOL_NAME}
    # _sanitize_schema is reused: unsupported keywords are stripped from input_schema.
    sent_props = tools[0]["input_schema"]["properties"]["n"]
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


# === retrieve / results / message mapping (19.3) ============================


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeUsage:
    input_tokens: int = 5
    output_tokens: int = 3


@dataclass
class _FakeStopDetails:
    explanation: str | None = None


@dataclass
class _FakeMessage:
    content: list[_FakeTextBlock]
    usage: _FakeUsage = field(default_factory=_FakeUsage)
    stop_reason: str = "end_turn"
    stop_details: _FakeStopDetails | None = None


def _message(
    text: str, *, stop_reason: str = "end_turn", explanation: str | None = None
) -> _FakeMessage:
    return _FakeMessage(
        content=[_FakeTextBlock(text=text)],
        stop_reason=stop_reason,
        stop_details=_FakeStopDetails(explanation=explanation) if explanation else None,
    )


@dataclass
class _FakeInnerError:
    type: str


@dataclass
class _FakeErrorResponse:
    error: _FakeInnerError


@dataclass
class _FakeResult:
    type: str
    message: Any = None
    error: Any = None


@dataclass
class _FakeEntry:
    custom_id: str
    result: _FakeResult


@dataclass
class _FakeRetrievedBatch:
    processing_status: str


class _FakeDecoder:
    def __init__(self, entries: list[_FakeEntry]) -> None:
        self._entries = entries

    async def __aiter__(self) -> AsyncIterator[_FakeEntry]:
        for entry in self._entries:
            yield entry


class _RetrieveResultsBatches:
    def __init__(
        self,
        *,
        status: str = "ended",
        entries: list[_FakeEntry] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._status = status
        self._entries = entries or []
        self._error = error

    async def retrieve(self, provider_batch_id: str, **kwargs: Any) -> _FakeRetrievedBatch:
        if self._error is not None:
            raise self._error
        return _FakeRetrievedBatch(processing_status=self._status)

    async def results(self, provider_batch_id: str, **kwargs: Any) -> _FakeDecoder:
        if self._error is not None:
            raise self._error
        return _FakeDecoder(self._entries)


class _RetrieveResultsClient:
    def __init__(self, batches: _RetrieveResultsBatches) -> None:
        self.batches = batches
        self.messages = _FakeMessagesResource(batches)  # type: ignore[arg-type]


def _rr_provider(batches: _RetrieveResultsBatches) -> AnthropicBatchProvider:
    return AnthropicBatchProvider(api_key="sk-ant-test", client=_RetrieveResultsClient(batches))  # type: ignore[arg-type]


async def test_retrieve_batch_returns_processing_status() -> None:
    provider = _rr_provider(_RetrieveResultsBatches(status="in_progress"))
    status = await provider.retrieve_batch("msgbatch_1")
    assert status.processing_status == "in_progress"


async def test_iter_results_normalizes_every_result_type() -> None:
    entries = [
        _FakeEntry("ebitem_ok", _FakeResult(type="succeeded", message=_message('{"items": []}'))),
        _FakeEntry(
            "ebitem_bad",
            _FakeResult(
                type="errored",
                error=_FakeErrorResponse(_FakeInnerError("invalid_request_error")),
            ),
        ),
        _FakeEntry(
            "ebitem_srv",
            _FakeResult(type="errored", error=_FakeErrorResponse(_FakeInnerError("api_error"))),
        ),
        _FakeEntry("ebitem_exp", _FakeResult(type="expired")),
        _FakeEntry("ebitem_can", _FakeResult(type="canceled")),
    ]
    provider = _rr_provider(_RetrieveResultsBatches(entries=entries))

    results = [r async for r in provider.iter_results("msgbatch_1")]
    by_id = {r.custom_id: r for r in results}
    assert by_id["ebitem_ok"].result_type == "succeeded"
    assert by_id["ebitem_bad"].result_type == "errored"
    assert by_id["ebitem_bad"].error_type == "invalid_request_error"
    assert by_id["ebitem_bad"].retryable is False
    assert by_id["ebitem_srv"].retryable is True  # server error is retryable
    assert by_id["ebitem_exp"].result_type == "expired"
    assert by_id["ebitem_can"].result_type == "canceled"


async def test_to_structured_output_uses_shared_mapping() -> None:
    provider = _rr_provider(_RetrieveResultsBatches())

    clean = BatchResult(custom_id="c", result_type="succeeded", message=_message('{"items": []}'))
    resp = provider.to_structured_output(clean, model="claude-sonnet-4-6")
    assert resp.output_json == {"items": []}
    assert resp.parse_error is None
    assert resp.provider == "anthropic"
    assert resp.model == "claude-sonnet-4-6"

    refusal = BatchResult(
        custom_id="c",
        result_type="succeeded",
        message=_message("", stop_reason="refusal", explanation="nope"),
    )
    rresp = provider.to_structured_output(refusal, model="claude-sonnet-4-6")
    assert rresp.output_json is None
    assert rresp.parse_error and "refused" in rresp.parse_error


async def test_retrieve_results_errors_wrapped() -> None:
    batches = _RetrieveResultsBatches(
        error=anthropic.APITimeoutError(request=_REQUEST_OBJ)
    )
    provider = _rr_provider(batches)
    with pytest.raises(LLMTechnicalError):
        await provider.retrieve_batch("msgbatch_1")


class _RaisingDecoder:
    """Decoder that yields one entry then raises mid-stream (connection drop)."""

    def __init__(self, first: _FakeEntry, exc: Exception) -> None:
        self._first = first
        self._exc = exc

    async def __aiter__(self) -> AsyncIterator[_FakeEntry]:
        yield self._first
        raise self._exc


class _RaisingResultsBatches:
    def __init__(self, decoder: _RaisingDecoder) -> None:
        self._decoder = decoder

    async def results(self, provider_batch_id: str, **kwargs: Any) -> _RaisingDecoder:
        return self._decoder


class _RaisingResultsClient:
    def __init__(self, decoder: _RaisingDecoder) -> None:
        self.batches = _RaisingResultsBatches(decoder)
        self.messages = _FakeMessagesResource(self.batches)  # type: ignore[arg-type]


class _InitialRaiseResultsBatches:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def results(self, provider_batch_id: str, **kwargs: Any) -> Any:
        raise self._exc


class _InitialRaiseClient:
    def __init__(self, exc: Exception) -> None:
        self.batches = _InitialRaiseResultsBatches(exc)
        self.messages = _FakeMessagesResource(self.batches)  # type: ignore[arg-type]


# The AsyncJSONLDecoder reads raw bytes, so mid-stream failures are httpx errors
# or json decode errors — NOT anthropic.APIError. All must normalize (review #2.1/#2.2).
@pytest.mark.parametrize(
    "exc",
    [
        anthropic.APIError("stream dropped", request=_REQUEST_OBJ, body=None),
        httpx.ReadError("connection reset"),
        httpx.RemoteProtocolError("peer closed"),
        json.JSONDecodeError("Expecting value", "", 0),
    ],
)
async def test_iter_results_normalizes_mid_stream_error(exc: Exception) -> None:
    # A failure mid-stream (during decoder iteration, not the initial call) must
    # surface as LLMTechnicalError so the poller's per-batch handler catches it
    # (review #1.1, #2.1, #2.2).
    first = _FakeEntry("ebitem_ok", _FakeResult(type="succeeded", message=_message("{}")))
    decoder = _RaisingDecoder(first, exc)
    provider = AnthropicBatchProvider(
        api_key="sk-ant-test", client=_RaisingResultsClient(decoder)  # type: ignore[arg-type]
    )
    seen = []
    with pytest.raises(LLMTechnicalError):
        async for result in provider.iter_results("msgbatch_1"):
            seen.append(result)
    assert len(seen) == 1  # the first result streamed before the failure


async def test_iter_results_initial_bare_anthropic_error_normalized() -> None:
    # batches.results() raises a bare anthropic.AnthropicError (not an APIError) on
    # an ended-but-not-ready batch race — _call_with_retry must normalize it (review #2.4).
    provider = AnthropicBatchProvider(
        api_key="sk-ant-test",
        client=_InitialRaiseClient(anthropic.AnthropicError("no results_url yet")),  # type: ignore[arg-type]
    )
    with pytest.raises(LLMTechnicalError):
        async for _ in provider.iter_results("msgbatch_1"):
            pass
