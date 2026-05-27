"""Bind FakeEmbeddingProvider and OpenAIEmbeddingProvider to the shared contract.

The OpenAI tests inject a typed fake async embeddings client (a ``Protocol``-shaped
stub whose ``embeddings.create`` is an async method returning deterministic-per-text
vectors), so they exercise the real provider's branching with no network and no key.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
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

from rag_recipes.providers._observability import (
    EMBEDDING_PREVIEW_CHARS,
    ProviderObservability,
    TraceContext,
)
from rag_recipes.providers.embeddings import openai as openai_provider
from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from rag_recipes.providers.embeddings.openai import OpenAIEmbeddingProvider
from rag_recipes.providers.errors import EmbeddingTechnicalError
from tests.contracts.embeddings import EmbeddingContract

_DIMENSIONS = 8


# --- typed fake async OpenAI embeddings client ------------------------------


def _deterministic_vector(text: str, dimensions: int) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]


@dataclass
class _FakeDatum:
    index: int
    embedding: list[float]


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeDatum]


class _FakeEmbeddings:
    def __init__(self, dimensions: int, error: Exception | None = None) -> None:
        self._dimensions = dimensions
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeEmbeddingResponse:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        inputs: list[str] = list(kwargs["input"])
        return _FakeEmbeddingResponse(
            data=[
                _FakeDatum(i, _deterministic_vector(text, self._dimensions))
                for i, text in enumerate(inputs)
            ]
        )


class _StubbedEmbeddings(_FakeEmbeddings):
    """Returns caller-supplied response data verbatim, ignoring input order.

    Lets tests force out-of-order, duplicate, or missing ``index`` values to
    exercise the provider's index-based re-mapping and validation.
    """

    def __init__(self, dimensions: int, data: list[_FakeDatum]) -> None:
        super().__init__(dimensions)
        self._data = data

    async def create(self, **kwargs: Any) -> _FakeEmbeddingResponse:
        self.calls.append(kwargs)
        return _FakeEmbeddingResponse(data=list(self._data))


@dataclass
class _FakeAsyncClient:
    """Duck-typed stand-in for ``AsyncOpenAI`` exposing ``embeddings.create``."""

    embeddings: _FakeEmbeddings


def _client(*, dimensions: int = _DIMENSIONS, error: Exception | None = None) -> Any:
    return _FakeAsyncClient(embeddings=_FakeEmbeddings(dimensions, error))


def _provider(
    *,
    dimensions: int = _DIMENSIONS,
    batch_size: int = 100,
    error: Exception | None = None,
    fake: Any = None,
) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=dimensions,
        batch_size=batch_size,
        client=fake if fake is not None else _client(dimensions=dimensions, error=error),
    )


_REQUEST_OBJ = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
_RESPONSE_OBJ = httpx.Response(429, request=_REQUEST_OBJ)

_TECHNICAL_ERRORS = [
    APITimeoutError(request=_REQUEST_OBJ),
    APIConnectionError(message="boom", request=_REQUEST_OBJ),
    RateLimitError("rate limited", response=_RESPONSE_OBJ, body=None),
    APIStatusError("bad status", response=_RESPONSE_OBJ, body=None),
    APIError("base error", request=_REQUEST_OBJ, body=None),
]


# --- contract bindings -------------------------------------------------------


class TestFakeEmbedding(EmbeddingContract):
    @pytest.fixture
    def provider(self) -> FakeEmbeddingProvider:
        return FakeEmbeddingProvider(dimensions=_DIMENSIONS)

    @pytest.fixture
    def expected_dimensions(self) -> int:
        return _DIMENSIONS


class TestOpenAIEmbedding(EmbeddingContract):
    @pytest.fixture
    def provider(self) -> OpenAIEmbeddingProvider:
        return _provider()

    @pytest.fixture
    def expected_dimensions(self) -> int:
        return _DIMENSIONS


# --- bespoke OpenAIEmbeddingProvider tests ----------------------------------


def test_default_client_disables_sdk_retries() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test", model="text-embedding-3-small", dimensions=_DIMENSIONS
    )
    assert provider._client.max_retries == 0


async def test_embed_text_passes_dimensions_to_api() -> None:
    fake = _client()
    provider = _provider(fake=fake)
    await provider.embed_text("hello")

    assert fake.embeddings.calls[0]["dimensions"] == _DIMENSIONS
    assert fake.embeddings.calls[0]["model"] == "text-embedding-3-small"


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
async def test_empty_text_short_circuits_to_zero_vector(text: str) -> None:
    fake = _client()
    provider = _provider(fake=fake)
    embedding = await provider.embed_text(text)

    assert embedding.vector == [0.0] * _DIMENSIONS
    assert embedding.provider == "openai"
    assert len(fake.embeddings.calls) == 0


async def test_embed_batch_preserves_order_and_chunks() -> None:
    fake = _client()
    provider = _provider(batch_size=2, fake=fake)
    texts = ["a", "", "b", "c", "d"]
    batch = await provider.embed_batch(texts)

    assert len(batch) == len(texts)
    # Empty slot gets a zero vector; non-empty slots match the deterministic
    # per-text vector (computed directly so embed_text calls don't pollute the
    # call-accounting asserted below).
    assert batch[1].vector == [0.0] * _DIMENSIONS
    for i in (0, 2, 3, 4):
        assert batch[i].vector == _deterministic_vector(texts[i], _DIMENSIONS)

    # 4 non-empty inputs at batch_size=2 → exactly 2 chunked create calls, each
    # with <= 2 inputs, and no empty string is ever sent to the API.
    assert len(fake.embeddings.calls) == 2
    for call in fake.embeddings.calls:
        assert len(call["input"]) <= 2
        assert "" not in call["input"]
        assert "   " not in call["input"]


async def test_embed_batch_maps_by_response_index_not_position() -> None:
    # Response data arrives in reverse order but carries correct indexes; the
    # provider must re-map by index, not trust list position.
    texts = ["a", "b", "c"]
    shuffled = [
        _FakeDatum(2, _deterministic_vector("c", _DIMENSIONS)),
        _FakeDatum(0, _deterministic_vector("a", _DIMENSIONS)),
        _FakeDatum(1, _deterministic_vector("b", _DIMENSIONS)),
    ]
    fake = _FakeAsyncClient(embeddings=_StubbedEmbeddings(_DIMENSIONS, shuffled))
    provider = _provider(fake=fake)
    batch = await provider.embed_batch(texts)

    for i, text in enumerate(texts):
        assert batch[i].vector == _deterministic_vector(text, _DIMENSIONS)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            [_FakeDatum(0, [0.0] * _DIMENSIONS), _FakeDatum(0, [0.0] * _DIMENSIONS)],
            id="duplicate-index",
        ),
        pytest.param([_FakeDatum(0, [0.0] * _DIMENSIONS)], id="missing-index"),
        pytest.param(
            [_FakeDatum(0, [0.0] * _DIMENSIONS), _FakeDatum(5, [0.0] * _DIMENSIONS)],
            id="out-of-range-index",
        ),
    ],
)
async def test_embed_chunk_rejects_malformed_indexes(data: list[_FakeDatum]) -> None:
    fake = _FakeAsyncClient(embeddings=_StubbedEmbeddings(_DIMENSIONS, data))
    provider = _provider(fake=fake)
    with pytest.raises(EmbeddingTechnicalError):
        await provider.embed_batch(["a", "b"])


async def test_embed_batch_rejects_oversized_input_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A single over-limit input is rejected in preflight, naming its slot, with
    # no API call spent (so earlier chunks in a real batch aren't paid-for then lost).
    monkeypatch.setattr(openai_provider, "_MAX_TOKENS_PER_INPUT", 4)
    fake = _client()
    provider = _provider(fake=fake)
    with pytest.raises(EmbeddingTechnicalError, match="input 1 is .* per-input limit"):
        await provider.embed_batch(["ok", "this input is far too long to embed"])
    assert len(fake.embeddings.calls) == 0


async def test_embed_batch_splits_on_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Count cap is high, but the aggregate token budget forces a second request.
    monkeypatch.setattr(openai_provider, "_MAX_TOKENS_PER_REQUEST", 2)
    fake = _client()
    provider = _provider(batch_size=100, fake=fake)
    texts = ["a", "b", "c"]  # 1 byte → 1 token each; budget 2 → 2 then 1
    batch = await provider.embed_batch(texts)

    assert len(fake.embeddings.calls) == 2
    for i, text in enumerate(texts):
        assert batch[i].vector == _deterministic_vector(text, _DIMENSIONS)


async def test_token_bound_counts_multibyte_text_by_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Token-dense input: 4 CJK chars = 12 UTF-8 bytes. A char/4 average would
    # estimate ~1 token and pass a 6-token limit; the byte upper bound rejects it,
    # so token-dense text can't slip an over-limit request past preflight.
    monkeypatch.setattr(openai_provider, "_MAX_TOKENS_PER_INPUT", 6)
    fake = _client()
    provider = _provider(fake=fake)
    with pytest.raises(EmbeddingTechnicalError, match="per-input limit"):
        await provider.embed_batch(["你好世界"])
    assert len(fake.embeddings.calls) == 0


@pytest.mark.parametrize("error", _TECHNICAL_ERRORS)
async def test_technical_errors_wrapped(error: Exception) -> None:
    provider = _provider(error=error)
    with pytest.raises(EmbeddingTechnicalError) as exc_info:
        await provider.embed_text("hello")
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("batch_size", [0, -1, 3000])
def test_constructor_rejects_out_of_range_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be in"):
        OpenAIEmbeddingProvider(
            api_key="sk-test",
            model="text-embedding-3-small",
            dimensions=_DIMENSIONS,
            batch_size=batch_size,
            client=_client(),
        )


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


def _traced_provider(
    fake: _FakeLangfuse,
    *,
    batch_size: int = 100,
    error: Exception | None = None,
) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        api_key="sk-secret-key",
        model="text-embedding-3-small",
        dimensions=_DIMENSIONS,
        batch_size=batch_size,
        client=_client(error=error),
        observability=ProviderObservability(fake, enabled=True),
    )


def _payloads(fake: _FakeLangfuse) -> str:
    return "\n".join(json.dumps(call, default=str) for call in fake.start_calls)


async def test_trace_records_embed_text_observation() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake)
    await provider.embed_text("a recipe to embed")

    call = fake.start_calls[0]
    assert call["as_type"] == "embedding"
    assert call["model"] == "text-embedding-3-small"
    assert call["input"] == "a recipe to embed"
    assert call["metadata"] == {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": _DIMENSIONS,
        "batch_size": 1,
        "text_preview": "a recipe to embed",
    }


async def test_trace_truncates_text_preview() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake)
    long_text = "x" * (EMBEDDING_PREVIEW_CHARS + 50)
    await provider.embed_text(long_text)

    call = fake.start_calls[0]
    assert call["input"] == long_text[:EMBEDDING_PREVIEW_CHARS]
    assert call["metadata"]["text_preview"] == long_text[:EMBEDDING_PREVIEW_CHARS]
    assert len(call["metadata"]["text_preview"]) == EMBEDDING_PREVIEW_CHARS


async def test_trace_records_batch_observation() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake, batch_size=2)
    await provider.embed_batch(["first", "second", "third"])

    call = fake.start_calls[0]
    assert call["as_type"] == "embedding"
    assert call["metadata"]["batch_size"] == 3
    assert call["metadata"]["text_preview"] == "first"
    # One observation for the whole batch, regardless of internal chunking.
    assert len(fake.start_calls) == 1


async def test_trace_records_technical_failure_and_reraises() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake, error=APITimeoutError(request=_REQUEST_OBJ))
    with pytest.raises(EmbeddingTechnicalError):
        await provider.embed_text("boom")

    update = fake.observations[0].updates[-1]
    assert update["level"] == "ERROR"
    assert update["status_message"]


async def test_trace_empty_text_short_circuit_does_not_crash() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake)
    embedding = await provider.embed_text("")

    assert embedding.vector == [0.0] * _DIMENSIONS
    # The observation is still opened (and closed cleanly) for the short-circuit.
    assert fake.start_calls[0]["as_type"] == "embedding"


async def test_trace_context_propagates_into_observation() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake)
    await provider.embed_text(
        "text",
        trace_context=TraceContext(session_id="sess-1", input_hash="hash-1"),
    )

    metadata = fake.start_calls[0]["metadata"]
    assert metadata["session_id"] == "sess-1"
    assert metadata["input_hash"] == "hash-1"


async def test_trace_payload_carries_no_secret_or_full_text() -> None:
    fake = _FakeLangfuse()
    provider = _traced_provider(fake)
    long_text = "secret-ingredient " * 50
    await provider.embed_text(long_text)

    payloads = _payloads(fake)
    assert "sk-secret-key" not in payloads
    # Only the bounded preview is recorded, never the full corpus text.
    assert long_text not in payloads


async def test_disabled_observability_never_touches_client() -> None:
    fake = _FakeLangfuse()
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=_DIMENSIONS,
        client=_client(),
        observability=ProviderObservability(fake, enabled=False),
    )
    await provider.embed_text("hello")
    await provider.embed_batch(["a", "b"])

    assert fake.start_calls == []
    assert fake.observations == []
