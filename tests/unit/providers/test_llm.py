"""Bind FakeLLMProvider and OpenAILLMProvider to the shared LLM contract suite.

The OpenAI tests inject a typed fake async client (a ``Protocol``-shaped stub
whose ``chat.completions.create`` is an async method), so they exercise the real
provider's branching with no network and no API key.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from rag_recipes.providers._observability import ProviderObservability, TraceContext
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.openai import OpenAILLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from tests.contracts.llm import LLMContract

_REQUEST = StructuredOutputRequest(
    provider="openai",
    model="gpt-4.1",
    prompt_version="recipe-v1",
    schema_version="recipe.v1",
    input="extract this",
    json_schema={"type": "object"},
)
_OUTPUT = {"title": "Soup", "ingredients": [{"name": "salt"}]}

_STRICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}

_OPENAI_REQUEST = StructuredOutputRequest(
    provider="openai",
    model="gpt-4.1",
    prompt_version="recipe-v1",
    schema_version="recipe.v1",
    input="Return ok=true",
    json_schema=_STRICT_SCHEMA,
)

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# --- typed fake async OpenAI client -----------------------------------------


@dataclass
class _FakeMessage:
    content: str | None = None
    refusal: str | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str = "stop"


@dataclass
class _FakeUsage:
    prompt_tokens: int = 11
    completion_tokens: int = 7


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]
    usage: _FakeUsage | None = field(default_factory=_FakeUsage)


_DEFAULT_USAGE = _FakeUsage()


def _completion(
    *,
    content: str | None = None,
    refusal: str | None = None,
    finish_reason: str = "stop",
    usage: _FakeUsage | None = _DEFAULT_USAGE,
) -> _FakeCompletion:
    return _FakeCompletion(
        choices=[
            _FakeChoice(
                message=_FakeMessage(content=content, refusal=refusal),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


class _FakeCompletions:
    def __init__(
        self, response: _FakeCompletion | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeAsyncClient:
    """Duck-typed stand-in for ``AsyncOpenAI`` exposing ``chat.completions.create``."""

    def __init__(
        self, response: _FakeCompletion | None = None, error: Exception | None = None
    ) -> None:
        self.completions = _FakeCompletions(response, error)
        self.chat = _FakeChat(self.completions)


def _client(*, response: _FakeCompletion | None = None, error: Exception | None = None) -> Any:
    return _FakeAsyncClient(response=response, error=error)


def _provider_with(response: _FakeCompletion) -> OpenAILLMProvider:
    return OpenAILLMProvider(
        api_key="sk-test", default_model="gpt-4.1", client=_client(response=response)
    )


_REQUEST_OBJ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
_RESPONSE_OBJ = httpx.Response(429, request=_REQUEST_OBJ)

_TECHNICAL_ERRORS = [
    APITimeoutError(request=_REQUEST_OBJ),
    APIConnectionError(message="boom", request=_REQUEST_OBJ),
    RateLimitError("rate limited", response=_RESPONSE_OBJ, body=None),
    APIStatusError("bad status", response=_RESPONSE_OBJ, body=None),
    APIError("base error", request=_REQUEST_OBJ, body=None),
]


# --- contract bindings -------------------------------------------------------


class TestFakeLLM(LLMContract):
    @pytest.fixture
    def provider(self) -> FakeLLMProvider:
        return FakeLLMProvider({FakeLLMProvider.request_hash(_REQUEST): _OUTPUT})

    @pytest.fixture
    def sample_request(self) -> StructuredOutputRequest:
        return _REQUEST

    @pytest.fixture
    def failure_provider(self) -> FakeLLMProvider:
        return FakeLLMProvider(fail_technically=True)


class TestOpenAILLM(LLMContract):
    @pytest.fixture
    def provider(self) -> OpenAILLMProvider:
        return _provider_with(_completion(content='{"ok": true}'))

    @pytest.fixture
    def sample_request(self) -> StructuredOutputRequest:
        return _OPENAI_REQUEST

    @pytest.fixture
    def failure_provider(self) -> OpenAILLMProvider:
        return OpenAILLMProvider(
            api_key="sk-test",
            default_model="gpt-4.1",
            client=_client(error=APITimeoutError(request=_REQUEST_OBJ)),
        )


# --- bespoke OpenAILLMProvider tests ----------------------------------------


def test_default_client_disables_sdk_retries() -> None:
    provider = OpenAILLMProvider(api_key="sk-test", default_model="gpt-4.1")
    assert provider._client.max_retries == 0


async def test_strict_json_schema_payload() -> None:
    client = _client(response=_completion(content='{"ok": true}'))
    provider = OpenAILLMProvider(api_key="sk-test", default_model="gpt-4.1", client=client)
    await provider.generate_structured_output(_OPENAI_REQUEST)

    kwargs = client.completions.calls[0]
    assert kwargs["model"] == _OPENAI_REQUEST.model
    response_format = kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == _OPENAI_REQUEST.json_schema


async def test_single_user_message_no_system() -> None:
    client = _client(response=_completion(content='{"ok": true}'))
    provider = OpenAILLMProvider(api_key="sk-test", default_model="gpt-4.1", client=client)
    await provider.generate_structured_output(_OPENAI_REQUEST)

    messages = client.completions.calls[0]["messages"]
    assert messages == [{"role": "user", "content": _OPENAI_REQUEST.input}]


@pytest.mark.parametrize(
    "schema_version",
    ["recipe.v1", "", "....", "x" * 100],
)
async def test_schema_name_is_always_valid(schema_version: str) -> None:
    client = _client(response=_completion(content='{"ok": true}'))
    provider = OpenAILLMProvider(api_key="sk-test", default_model="gpt-4.1", client=client)
    request = _OPENAI_REQUEST.model_copy(update={"schema_version": schema_version})
    await provider.generate_structured_output(request)

    name = client.completions.calls[0]["response_format"]["json_schema"]["name"]
    assert _NAME_RE.match(name), f"invalid schema name: {name!r}"


async def test_clean_parse_sets_output_json() -> None:
    provider = _provider_with(_completion(content='{"ok": true}', usage=_FakeUsage(13, 5)))
    response = await provider.generate_structured_output(_OPENAI_REQUEST)

    assert response.output_json == {"ok": True}
    assert response.parse_error is None
    assert response.raw_text == '{"ok": true}'
    assert response.usage.input_tokens == 13
    assert response.usage.output_tokens == 5
    assert response.provider == "openai"
    assert response.model == "gpt-4.1"


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [
        ("not json at all", "stop"),
        ("[1, 2, 3]", "stop"),
        ('{"ok": true}', "length"),
        ('{"ok": true}', "content_filter"),
    ],
)
async def test_parse_failures_do_not_raise(content: str, finish_reason: str) -> None:
    client = _client(response=_completion(content=content, finish_reason=finish_reason))
    provider = OpenAILLMProvider(api_key="sk-test", default_model="gpt-4.1", client=client)
    response = await provider.generate_structured_output(_OPENAI_REQUEST)

    assert response.output_json is None
    assert response.parse_error
    assert response.raw_text == content
    assert len(client.completions.calls) == 1


async def test_refusal_preserves_text() -> None:
    client = _client(response=_completion(content=None, refusal="I cannot help with that."))
    provider = OpenAILLMProvider(api_key="sk-test", default_model="gpt-4.1", client=client)
    response = await provider.generate_structured_output(_OPENAI_REQUEST)

    assert response.output_json is None
    assert response.parse_error
    assert response.raw_text == "I cannot help with that."
    assert len(client.completions.calls) == 1


async def test_missing_content_and_refusal_yields_empty_raw_text() -> None:
    provider = _provider_with(_completion(content=None, refusal=None))
    response = await provider.generate_structured_output(_OPENAI_REQUEST)

    assert response.output_json is None
    assert response.parse_error
    assert response.raw_text == ""


async def test_missing_usage_falls_back_to_zero() -> None:
    provider = _provider_with(_completion(content='{"ok": true}', usage=None))
    response = await provider.generate_structured_output(_OPENAI_REQUEST)

    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0


async def test_response_provider_is_openai_regardless_of_request() -> None:
    provider = _provider_with(_completion(content='{"ok": true}'))
    mislabelled = _OPENAI_REQUEST.model_copy(update={"provider": "anthropic"})
    response = await provider.generate_structured_output(mislabelled)

    assert response.provider == "openai"


@pytest.mark.parametrize("error", _TECHNICAL_ERRORS)
async def test_technical_errors_wrapped(error: Exception) -> None:
    provider = OpenAILLMProvider(
        api_key="sk-test", default_model="gpt-4.1", client=_client(error=error)
    )
    with pytest.raises(LLMTechnicalError) as exc_info:
        await provider.generate_structured_output(_OPENAI_REQUEST)
    assert exc_info.value.__cause__ is error


# --- Langfuse tracing --------------------------------------------------------


class _RecordingObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _FakeLangfuse:
    """``Protocol``-shaped stub satisfying ``LangfuseLike``."""

    def __init__(self) -> None:
        self.start_calls: list[dict[str, Any]] = []
        self.observations: list[_RecordingObservation] = []
        self.propagated_session_ids: list[str | None] = []

    @contextlib.contextmanager
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: str,
        input: Any = None,
        metadata: Any = None,
        model: str | None = None,
    ) -> Iterator[_RecordingObservation]:
        self.start_calls.append(
            {
                "name": name,
                "as_type": as_type,
                "input": input,
                "metadata": metadata,
                "model": model,
            }
        )
        observation = _RecordingObservation()
        self.observations.append(observation)
        yield observation

    @contextlib.contextmanager
    def propagate_attributes(self, *, session_id: str | None = None) -> Iterator[None]:
        self.propagated_session_ids.append(session_id)
        yield


def _traced_provider(
    fake: _FakeLangfuse, *, response: _FakeCompletion | None = None, error: Exception | None = None
) -> OpenAILLMProvider:
    return OpenAILLMProvider(
        api_key="sk-secret-key",
        default_model="gpt-4.1",
        client=_client(response=response, error=error),
        observability=ProviderObservability(fake, enabled=True),
    )


def _payloads(fake: _FakeLangfuse) -> str:
    parts = [json.dumps(call, default=str) for call in fake.start_calls]
    for obs in fake.observations:
        parts.extend(json.dumps(u, default=str) for u in obs.updates)
    return "\n".join(parts)


async def test_trace_records_clean_generation() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(
        fake, response=_completion(content='{"ok": true}', usage=_FakeUsage(13, 5))
    )
    await provider.generate_structured_output(_OPENAI_REQUEST)

    call = fake.start_calls[0]
    assert call["as_type"] == "generation"
    assert call["model"] == _OPENAI_REQUEST.model
    assert call["metadata"]["provider"] == "openai"
    assert call["metadata"]["prompt_version"] == _OPENAI_REQUEST.prompt_version
    assert call["metadata"]["schema_version"] == _OPENAI_REQUEST.schema_version

    update = fake.observations[0].updates[0]
    assert update["output"] == {"parsed": {"ok": True}, "raw": '{"ok": true}'}
    assert update["usage_details"] == {"input": 13, "output": 5}
    assert update["metadata"]["status"] == "success"
    assert update["level"] == "DEFAULT"
    assert update["status_message"] is None


async def test_trace_records_rejected_generation() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake, response=_completion(content="not json"))
    response = await provider.generate_structured_output(_OPENAI_REQUEST)

    # The returned response is unchanged by tracing.
    assert response.output_json is None
    assert response.parse_error

    update = fake.observations[0].updates[0]
    assert update["metadata"]["status"] == "rejected"
    assert update["level"] == "WARNING"
    assert update["status_message"] == response.parse_error


async def test_trace_records_technical_failure_and_reraises() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake, error=APITimeoutError(request=_REQUEST_OBJ))
    with pytest.raises(LLMTechnicalError):
        await provider.generate_structured_output(_OPENAI_REQUEST)

    update = fake.observations[0].updates[-1]
    assert update["level"] == "ERROR"
    assert update["status_message"]


async def test_trace_context_propagates_into_observation() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake, response=_completion(content='{"ok": true}'))
    await provider.generate_structured_output(
        _OPENAI_REQUEST,
        trace_context=TraceContext(
            session_id="sess-1",
            input_source_span_ids=["span-a"],
            input_hash="hash-1",
        ),
    )

    metadata = fake.start_calls[0]["metadata"]
    assert metadata["input_source_span_ids"] == ["span-a"]
    assert metadata["input_hash"] == "hash-1"
    # session_id is propagated as a trace attribute for Langfuse session grouping,
    # not folded into observation metadata.
    assert "session_id" not in metadata
    assert fake.propagated_session_ids == ["sess-1"]


async def test_trace_payload_carries_no_secret() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake, response=_completion(content='{"ok": true}'))
    await provider.generate_structured_output(_OPENAI_REQUEST)

    assert "sk-secret-key" not in _payloads(fake)


async def test_disabled_observability_never_touches_client() -> None:
    fake = _FakeLangfuse()
    provider = OpenAILLMProvider(
        api_key="sk-test",
        default_model="gpt-4.1",
        client=_client(response=_completion(content='{"ok": true}')),
        observability=ProviderObservability(fake, enabled=False),
    )
    await provider.generate_structured_output(_OPENAI_REQUEST)

    assert fake.start_calls == []
    assert fake.observations == []
