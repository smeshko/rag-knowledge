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

import anthropic
import httpx
import pytest
from anthropic._exceptions import OverloadedError
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from rag_recipes.answers.schema import build_answer_v1_json_schema
from rag_recipes.ingestion.pipeline.extraction import build_recipe_v1_json_schema
from rag_recipes.providers._observability import ProviderObservability, TraceContext
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.anthropic import (
    _OUTPUT_TOOL_NAME,
    _UNSUPPORTED_SCHEMA_KEYWORDS,
    AnthropicLLMProvider,
    _sanitize_schema,
)
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.openai import OpenAILLMProvider, _backoff_delay
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


# --- rate-limit retry / backoff / timeout (Phase 9.5) -----------------------


class _SequenceCompletions:
    """``chat.completions`` stub that replays a scripted list of outcomes.

    Each action is either an exception to raise or a ``_FakeCompletion`` to
    return, popped in order on successive ``create`` calls.
    """

    def __init__(self, actions: list[Any]) -> None:
        self._actions = list(actions)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class _SequenceClient:
    def __init__(self, actions: list[Any]) -> None:
        self.completions = _SequenceCompletions(actions)
        self.chat = _FakeChat(self.completions)


def _rate_limit_error(code: str | None, *, retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, headers=headers, request=_REQUEST_OBJ)
    # body must be a dict for the SDK to populate ``.code`` from it.
    return RateLimitError("rate limited", response=response, body={"code": code})


@pytest.fixture
def _recorded_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(
        "rag_recipes.providers.llm.openai.asyncio.sleep", _fake_sleep
    )
    return delays


async def test_rate_limit_retried_then_succeeds_honors_retry_after(
    _recorded_sleep: list[float],
) -> None:
    client = _SequenceClient(
        [
            _rate_limit_error("rate_limit_exceeded", retry_after="2"),
            _completion(content='{"ok": true}'),
        ]
    )
    provider = OpenAILLMProvider(
        api_key="sk-test",
        default_model="gpt-4.1",
        client=client,
        max_rate_limit_retries=5,
    )
    response = await provider.generate_structured_output(_OPENAI_REQUEST)

    assert response.output_json == {"ok": True}
    assert len(client.completions.calls) == 2
    # Retry-After header is preferred over computed backoff.
    assert _recorded_sleep == [2.0]


async def test_rate_limit_exhausts_retries_raises(_recorded_sleep: list[float]) -> None:
    err = _rate_limit_error("rate_limit_exceeded")
    client = _SequenceClient([err, err, err])
    provider = OpenAILLMProvider(
        api_key="sk-test",
        default_model="gpt-4.1",
        client=client,
        max_rate_limit_retries=2,
    )
    with pytest.raises(LLMTechnicalError):
        await provider.generate_structured_output(_OPENAI_REQUEST)

    # 1 initial attempt + 2 retries = 3 calls, 2 sleeps.
    assert len(client.completions.calls) == 3
    assert len(_recorded_sleep) == 2


async def test_insufficient_quota_fails_fast_no_retry(
    _recorded_sleep: list[float],
) -> None:
    client = _SequenceClient(
        [_rate_limit_error("insufficient_quota", retry_after="5")]
    )
    provider = OpenAILLMProvider(
        api_key="sk-test",
        default_model="gpt-4.1",
        client=client,
        max_rate_limit_retries=5,
    )
    with pytest.raises(LLMTechnicalError):
        await provider.generate_structured_output(_OPENAI_REQUEST)

    assert len(client.completions.calls) == 1
    assert _recorded_sleep == []


async def test_zero_retries_raises_on_first_rate_limit(
    _recorded_sleep: list[float],
) -> None:
    client = _SequenceClient([_rate_limit_error("rate_limit_exceeded")])
    provider = OpenAILLMProvider(
        api_key="sk-test",
        default_model="gpt-4.1",
        client=client,
        max_rate_limit_retries=0,
    )
    with pytest.raises(LLMTechnicalError):
        await provider.generate_structured_output(_OPENAI_REQUEST)

    assert len(client.completions.calls) == 1
    assert _recorded_sleep == []


async def test_create_receives_request_timeout() -> None:
    client = _client(response=_completion(content='{"ok": true}'))
    provider = OpenAILLMProvider(
        api_key="sk-test",
        default_model="gpt-4.1",
        client=client,
        request_timeout=42.0,
    )
    await provider.generate_structured_output(_OPENAI_REQUEST)

    assert client.completions.calls[0]["timeout"] == 42.0


async def test_retries_stay_in_one_observability_span(
    _recorded_sleep: list[float],
) -> None:
    fake = _FakeLangfuse()
    client = _SequenceClient(
        [
            _rate_limit_error("rate_limit_exceeded", retry_after="1"),
            _completion(content='{"ok": true}'),
        ]
    )
    provider = OpenAILLMProvider(
        api_key="sk-secret-key",
        default_model="gpt-4.1",
        client=client,
        observability=ProviderObservability(fake, enabled=True),
        max_rate_limit_retries=5,
    )
    await provider.generate_structured_output(_OPENAI_REQUEST)

    # Two billable attempts, but exactly one span for the whole operation.
    assert len(client.completions.calls) == 2
    assert len(fake.start_calls) == 1


def test_backoff_delay_is_bounded_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin jitter to 0 to test the deterministic core: base at attempt 0, capped
    # at the ceiling for large attempts.
    monkeypatch.setattr(
        "rag_recipes.providers.llm.openai.random.uniform", lambda _a, _b: 0.0
    )
    assert _backoff_delay(0, base=0.5, cap=30.0) == pytest.approx(0.5)
    assert _backoff_delay(2, base=0.5, cap=30.0) == pytest.approx(2.0)
    assert _backoff_delay(100, base=0.5, cap=30.0) == pytest.approx(30.0)


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


class _SessionScopeRecorder:
    """Records propagated session IDs (the module-level ``propagate_attributes``)."""

    def __init__(self) -> None:
        self.session_ids: list[str] = []

    @contextlib.contextmanager
    def __call__(self, *, session_id: str) -> Iterator[None]:
        self.session_ids.append(session_id)
        yield


def _traced_provider(
    fake: _FakeLangfuse,
    *,
    response: _FakeCompletion | None = None,
    error: Exception | None = None,
    session_scope: _SessionScopeRecorder | None = None,
) -> OpenAILLMProvider:
    return OpenAILLMProvider(
        api_key="sk-secret-key",
        default_model="gpt-4.1",
        client=_client(response=response, error=error),
        observability=ProviderObservability(fake, enabled=True, session_scope=session_scope),
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
    assert update["metadata"] == {"status": "failed"}


async def test_trace_context_propagates_into_observation() -> None:
    fake = _FakeLangfuse()
    session = _SessionScopeRecorder()
    provider = _traced_provider(
        fake, response=_completion(content='{"ok": true}'), session_scope=session
    )
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
    assert session.session_ids == ["sess-1"]


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


# === AnthropicLLMProvider ===================================================


_ANTHROPIC_REQUEST = StructuredOutputRequest(
    provider="anthropic",
    model="claude-sonnet-4-6",
    prompt_version="recipe-v1",
    schema_version="recipe.v1",
    input="Return ok=true",
    json_schema=_STRICT_SCHEMA,
)

_ANTHROPIC_REQUEST_OBJ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


# --- typed fake async Anthropic client --------------------------------------


@dataclass
class _FakeToolUseBlock:
    input: Any
    name: str = _OUTPUT_TOOL_NAME
    type: str = "tool_use"


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeAnthropicUsage:
    input_tokens: int = 11
    output_tokens: int = 7


@dataclass
class _FakeStopDetails:
    explanation: str | None = None
    category: str | None = None


@dataclass
class _FakeAnthropicMessage:
    content: list[Any]
    usage: _FakeAnthropicUsage = field(default_factory=_FakeAnthropicUsage)
    stop_reason: str = "tool_use"
    stop_details: _FakeStopDetails | None = None


def _anthropic_message(
    *,
    tool_input: Any = None,
    text: str | None = None,
    stop_reason: str = "tool_use",
    stop_details: _FakeStopDetails | None = None,
    usage: _FakeAnthropicUsage | None = None,
) -> _FakeAnthropicMessage:
    """Build a fake Anthropic message.

    ``tool_input`` (the success path) adds a ``tool_use`` block carrying it as
    ``.input``; ``text`` adds a ``text`` block (the refusal / no-tool_use fallback).
    """
    content: list[Any] = []
    if tool_input is not None:
        content.append(_FakeToolUseBlock(input=tool_input))
    if text is not None:
        content.append(_FakeTextBlock(text=text))
    return _FakeAnthropicMessage(
        content=content,
        usage=usage or _FakeAnthropicUsage(),
        stop_reason=stop_reason,
        stop_details=stop_details,
    )


class _FakeAnthropicMessages:
    def __init__(
        self,
        response: _FakeAnthropicMessage | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeAnthropicMessage:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _FakeAnthropicClient:
    """Duck-typed stand-in for ``AsyncAnthropic`` exposing ``messages.create``.

    ``with_options(...)`` returns ``self`` (the real client returns a configured
    copy), so the provider's ``with_options(timeout=…).messages.create(…)`` chain
    records both the option kwargs and the create kwargs on one object.
    """

    def __init__(
        self,
        response: _FakeAnthropicMessage | None = None,
        error: Exception | None = None,
    ) -> None:
        self.messages = _FakeAnthropicMessages(response, error)
        self.option_calls: list[dict[str, Any]] = []

    def with_options(self, **kwargs: Any) -> Any:
        self.option_calls.append(kwargs)
        return self


def _anthropic_provider_with(response: _FakeAnthropicMessage) -> AnthropicLLMProvider:
    return AnthropicLLMProvider(
        api_key="sk-ant-test",
        default_model="claude-sonnet-4-6",
        client=_FakeAnthropicClient(response=response),
    )


# --- contract binding -------------------------------------------------------


class TestAnthropicLLM(LLMContract):
    @pytest.fixture
    def provider(self) -> AnthropicLLMProvider:
        return _anthropic_provider_with(_anthropic_message(tool_input={"ok": True}))

    @pytest.fixture
    def sample_request(self) -> StructuredOutputRequest:
        return _ANTHROPIC_REQUEST

    @pytest.fixture
    def failure_provider(self) -> AnthropicLLMProvider:
        return AnthropicLLMProvider(
            api_key="sk-ant-test",
            default_model="claude-sonnet-4-6",
            client=_FakeAnthropicClient(
                error=anthropic.APITimeoutError(request=_ANTHROPIC_REQUEST_OBJ)
            ),
        )


# --- bespoke AnthropicLLMProvider tests -------------------------------------


def test_anthropic_default_client_disables_sdk_retries() -> None:
    provider = AnthropicLLMProvider(
        api_key="sk-ant-test", default_model="claude-sonnet-4-6"
    )
    assert provider._client.max_retries == 0


async def test_anthropic_clean_parse_sets_output_json() -> None:
    provider = _anthropic_provider_with(
        _anthropic_message(tool_input={"ok": True}, usage=_FakeAnthropicUsage(13, 5))
    )
    response = await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    assert response.output_json == {"ok": True}
    assert response.parse_error is None
    assert response.raw_text == '{"ok": true}'
    assert response.usage.input_tokens == 13
    assert response.usage.output_tokens == 5
    assert response.provider == "anthropic"
    assert response.model == "claude-sonnet-4-6"


async def test_anthropic_request_sends_non_strict_forced_tool_use() -> None:
    client = _FakeAnthropicClient(response=_anthropic_message(tool_input={"ok": True}))
    provider = AnthropicLLMProvider(
        api_key="sk-ant-test",
        default_model="claude-sonnet-4-6",
        client=client,
        max_tokens=4096,
        request_timeout=42.0,
    )
    await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    kwargs = client.messages.calls[0]
    assert kwargs["model"] == _ANTHROPIC_REQUEST.model
    assert kwargs["max_tokens"] == 4096
    assert kwargs["messages"] == [{"role": "user", "content": _ANTHROPIC_REQUEST.input}]
    # Non-strict forced tool-use replaces output_config — the strict json_schema path
    # compiles a constrained-decoding grammar that 400s on the real recipe.v1 schema.
    assert "output_config" not in kwargs
    tools = kwargs["tools"]
    assert len(tools) == 1
    tool = tools[0]
    assert tool["name"] == _OUTPUT_TOOL_NAME
    # Non-strict: a strict tool input_schema would compile the same oversized grammar.
    assert not tool.get("strict")
    # _STRICT_SCHEMA carries no unsupported keywords, so the sanitized schema equals it.
    assert tool["input_schema"] == _ANTHROPIC_REQUEST.json_schema
    assert kwargs["tool_choice"] == {"type": "tool", "name": _OUTPUT_TOOL_NAME}
    # Per-request timeout is applied via with_options, not the create() call.
    assert client.option_calls[0]["timeout"] == 42.0


async def test_anthropic_refusal_returns_parse_error() -> None:
    provider = _anthropic_provider_with(
        _anthropic_message(
            text="",
            stop_reason="refusal",
            stop_details=_FakeStopDetails(explanation="I cannot help with that."),
        )
    )
    response = await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    assert response.output_json is None
    assert response.parse_error
    assert "refused" in response.parse_error
    assert "I cannot help with that." in response.parse_error
    assert response.raw_text == ""


async def test_anthropic_truncation_returns_parse_error() -> None:
    provider = _anthropic_provider_with(
        _anthropic_message(text='{"ok": tr', stop_reason="max_tokens")
    )
    response = await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    assert response.output_json is None
    assert response.parse_error
    assert "truncated" in response.parse_error
    assert response.raw_text == '{"ok": tr'


async def test_anthropic_missing_tool_use_block_rejected() -> None:
    # The model answered in prose instead of calling the forced tool — a rejection,
    # not a crash. The text survives as raw_text (the model's explanation).
    client = _FakeAnthropicClient(
        response=_anthropic_message(text="I can't do that", stop_reason="end_turn")
    )
    provider = AnthropicLLMProvider(
        api_key="sk-ant-test", default_model="claude-sonnet-4-6", client=client
    )
    response = await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    assert response.output_json is None
    assert response.parse_error and "tool_use" in response.parse_error
    assert response.raw_text == "I can't do that"
    assert len(client.messages.calls) == 1


async def test_anthropic_non_object_tool_input_rejected() -> None:
    # A tool_use block whose .input is not a JSON object is rejected; raw_text is the
    # serialized input for debugging.
    client = _FakeAnthropicClient(response=_anthropic_message(tool_input=[1, 2, 3]))
    provider = AnthropicLLMProvider(
        api_key="sk-ant-test", default_model="claude-sonnet-4-6", client=client
    )
    response = await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    assert response.output_json is None
    assert response.parse_error and "JSON object" in response.parse_error
    assert response.raw_text == json.dumps([1, 2, 3])
    assert len(client.messages.calls) == 1


# --- rate-limit / overloaded retry parity -----------------------------------


class _SequenceAnthropicMessages:
    """``messages`` stub that replays a scripted list of outcomes in order."""

    def __init__(self, actions: list[Any]) -> None:
        self._actions = list(actions)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeAnthropicMessage:
        self.calls.append(kwargs)
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        assert isinstance(action, _FakeAnthropicMessage)
        return action


class _SequenceAnthropicClient:
    def __init__(self, actions: list[Any]) -> None:
        self.messages = _SequenceAnthropicMessages(actions)
        self.option_calls: list[dict[str, Any]] = []

    def with_options(self, **kwargs: Any) -> Any:
        self.option_calls.append(kwargs)
        return self


def _anthropic_rate_limit_error(*, retry_after: str | None = None) -> anthropic.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, headers=headers, request=_ANTHROPIC_REQUEST_OBJ)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def _anthropic_overloaded_error(*, retry_after: str | None = None) -> OverloadedError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(529, headers=headers, request=_ANTHROPIC_REQUEST_OBJ)
    return OverloadedError("overloaded", response=response, body=None)


@pytest.fixture
def _recorded_anthropic_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(
        "rag_recipes.providers.llm.anthropic.asyncio.sleep", _fake_sleep
    )
    return delays


def _sequence_provider(actions: list[Any], *, max_retries: int = 5) -> AnthropicLLMProvider:
    return AnthropicLLMProvider(
        api_key="sk-ant-test",
        default_model="claude-sonnet-4-6",
        client=_SequenceAnthropicClient(actions),
        max_rate_limit_retries=max_retries,
    )


async def test_anthropic_rate_limit_retried_then_succeeds_honors_retry_after(
    _recorded_anthropic_sleep: list[float],
) -> None:
    client = _SequenceAnthropicClient(
        [_anthropic_rate_limit_error(retry_after="2"), _anthropic_message(tool_input={"ok": True})]
    )
    provider = AnthropicLLMProvider(
        api_key="sk-ant-test",
        default_model="claude-sonnet-4-6",
        client=client,
        max_rate_limit_retries=5,
    )
    response = await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    assert response.output_json == {"ok": True}
    assert len(client.messages.calls) == 2
    # Retry-After header is preferred over computed backoff.
    assert _recorded_anthropic_sleep == [2.0]


async def test_anthropic_overloaded_retried_then_succeeds(
    _recorded_anthropic_sleep: list[float],
) -> None:
    client = _SequenceAnthropicClient(
        [_anthropic_overloaded_error(), _anthropic_message(tool_input={"ok": True})]
    )
    provider = AnthropicLLMProvider(
        api_key="sk-ant-test",
        default_model="claude-sonnet-4-6",
        client=client,
        max_rate_limit_retries=5,
    )
    response = await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    assert response.output_json == {"ok": True}
    assert len(client.messages.calls) == 2
    # No Retry-After header → one computed backoff sleep.
    assert len(_recorded_anthropic_sleep) == 1


async def test_anthropic_rate_limit_exhausts_retries_raises(
    _recorded_anthropic_sleep: list[float],
) -> None:
    err = _anthropic_rate_limit_error()
    provider = _sequence_provider([err, err, err], max_retries=2)
    with pytest.raises(LLMTechnicalError):
        await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    # 1 initial attempt + 2 retries = 3 calls, 2 sleeps.
    client = provider._client
    assert isinstance(client, _SequenceAnthropicClient)
    assert len(client.messages.calls) == 3
    assert len(_recorded_anthropic_sleep) == 2


async def test_anthropic_overloaded_exhausts_retries_raises(
    _recorded_anthropic_sleep: list[float],
) -> None:
    err = _anthropic_overloaded_error()
    provider = _sequence_provider([err, err], max_retries=1)
    with pytest.raises(LLMTechnicalError):
        await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    client = provider._client
    assert isinstance(client, _SequenceAnthropicClient)
    assert len(client.messages.calls) == 2
    assert len(_recorded_anthropic_sleep) == 1


async def test_anthropic_non_retryable_status_raises_immediately(
    _recorded_anthropic_sleep: list[float],
) -> None:
    status_error = anthropic.APIStatusError(
        "server error",
        response=httpx.Response(500, request=_ANTHROPIC_REQUEST_OBJ),
        body=None,
    )
    provider = _sequence_provider([status_error], max_retries=5)
    with pytest.raises(LLMTechnicalError):
        await provider.generate_structured_output(_ANTHROPIC_REQUEST)

    client = provider._client
    assert isinstance(client, _SequenceAnthropicClient)
    assert len(client.messages.calls) == 1
    assert _recorded_anthropic_sleep == []


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APITimeoutError(request=_ANTHROPIC_REQUEST_OBJ),
        anthropic.APIConnectionError(message="boom", request=_ANTHROPIC_REQUEST_OBJ),
        anthropic.APIError("base error", request=_ANTHROPIC_REQUEST_OBJ, body=None),
    ],
)
async def test_anthropic_technical_errors_wrapped(error: Exception) -> None:
    provider = AnthropicLLMProvider(
        api_key="sk-ant-test",
        default_model="claude-sonnet-4-6",
        client=_FakeAnthropicClient(error=error),
    )
    with pytest.raises(LLMTechnicalError) as exc_info:
        await provider.generate_structured_output(_ANTHROPIC_REQUEST)
    assert exc_info.value.__cause__ is error


# --- schema sanitizer + compatibility guard ---------------------------------


def test_sanitize_schema_strips_unsupported_keywords_recursively() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["n", "s", "arr"],
        "properties": {
            "n": {"type": "integer", "minimum": 0, "maximum": 10, "multipleOf": 2},
            "s": {"type": "string", "minLength": 1, "maxLength": 5, "pattern": "^a"},
            "arr": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "uniqueItems": True,
                "items": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
            },
        },
    }
    out = _sanitize_schema(schema)

    # Input is not mutated.
    assert "minimum" in schema["properties"]["n"]

    n = out["properties"]["n"]
    assert not ({"minimum", "maximum", "multipleOf"} & set(n))
    s = out["properties"]["s"]
    assert not ({"minLength", "maxLength", "pattern"} & set(s))
    arr = out["properties"]["arr"]
    assert not ({"minItems", "maxItems", "uniqueItems"} & set(arr))
    assert not ({"exclusiveMinimum", "exclusiveMaximum"} & set(arr["items"]))
    # Types and structure survive.
    assert n["type"] == "integer"
    assert arr["items"]["type"] == "number"


def test_sanitize_schema_preserves_supported_keywords() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "when": {"type": "string", "format": "date-time"},
            "kind": {"enum": ["a", "b"]},
            "ref": {"$ref": "#/$defs/Thing"},
        },
        "$defs": {
            "Thing": {"type": "object", "additionalProperties": False, "properties": {}}
        },
    }
    out = _sanitize_schema(schema)

    assert out["properties"]["when"]["format"] == "date-time"
    assert out["properties"]["kind"]["enum"] == ["a", "b"]
    assert out["properties"]["ref"]["$ref"] == "#/$defs/Thing"
    assert out["additionalProperties"] is False


def test_sanitize_schema_preserves_property_named_like_a_keyword() -> None:
    # A field literally named after a stripped keyword must survive — only its
    # subschema is sanitized, never the field itself (review #1.1). Otherwise a
    # future recipe/answer field named e.g. "pattern" would be silently deleted
    # and the strictified "required" list would reference a missing property.
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["pattern", "maximum"],
        "properties": {
            "pattern": {"type": "string", "maxLength": 5},
            "maximum": {"type": "integer", "minimum": 0},
        },
        "$defs": {
            "minItems": {"type": "object", "additionalProperties": False, "properties": {}}
        },
    }
    out = _sanitize_schema(schema)

    # Field names that collide with keywords are preserved.
    assert set(out["properties"]) == {"pattern", "maximum"}
    assert out["$defs"].keys() == {"minItems"}
    # ...but the unsupported keywords inside their subschemas are still stripped.
    assert "maxLength" not in out["properties"]["pattern"]
    assert "minimum" not in out["properties"]["maximum"]
    assert out["properties"]["pattern"]["type"] == "string"


def _iter_schema_nodes(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_schema_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_schema_nodes(item)


def _schema_has_recursive_ref(schema: dict[str, Any]) -> bool:
    """True if any ``$defs`` entry references itself directly or transitively.

    Claude structured outputs reject recursion; the sanitizer cannot rewrite a
    recursive ``$ref``, so the guard fails loudly instead of masking it.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}

    def refs_in(body: Any) -> set[str]:
        found: set[str] = set()
        for node in _iter_schema_nodes(body):
            ref = node.get("$ref")
            if isinstance(ref, str) and ("/$defs/" in ref or "/definitions/" in ref):
                found.add(ref.rsplit("/", 1)[-1])
        return found

    graph = {name: refs_in(body) for name, body in defs.items()}
    visiting: set[str] = set()
    done: set[str] = set()

    def has_cycle(name: str) -> bool:
        visiting.add(name)
        for target in graph.get(name, set()):
            if target not in graph:
                continue
            if target in visiting:
                return True
            if target not in done and has_cycle(target):
                return True
        visiting.discard(name)
        done.add(name)
        return False

    return any(name not in done and has_cycle(name) for name in graph)


@pytest.mark.parametrize(
    "build_schema",
    [build_recipe_v1_json_schema, build_answer_v1_json_schema],
)
def test_production_schemas_carry_no_unsupported_keywords_or_recursion(
    build_schema: Any,
) -> None:
    # A cheap drift-catcher (NOT a full compatibility proof — union/grammar
    # limits are verified by the opt-in live test). Fails loudly if the shared
    # schema later grows a constraint keyword or a recursive $ref.
    schema = build_schema()
    for node in _iter_schema_nodes(schema):
        present = set(node) & _UNSUPPORTED_SCHEMA_KEYWORDS
        assert not present, f"unsupported keyword(s) {present} in node {node}"
    assert not _schema_has_recursive_ref(schema)
