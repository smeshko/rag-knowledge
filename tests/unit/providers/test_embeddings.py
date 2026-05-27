"""Bind FakeEmbeddingProvider and OpenAIEmbeddingProvider to the shared contract.

The OpenAI tests inject a typed fake async embeddings client (a ``Protocol``-shaped
stub whose ``embeddings.create`` is an async method returning deterministic-per-text
vectors), so they exercise the real provider's branching with no network and no key.
"""

from __future__ import annotations

import hashlib
import random
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
            data=[_FakeDatum(_deterministic_vector(text, self._dimensions)) for text in inputs]
        )


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
