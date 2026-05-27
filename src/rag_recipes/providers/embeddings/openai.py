"""OpenAIEmbeddingProvider — embeddings via the OpenAI SDK (doc 11 § 4).

The OpenAI embeddings API sits behind our own ``EmbeddingProvider`` interface.
Every ``Embedding`` records ``provider="openai"`` plus the configured model and
dimensions (the provider/model/dimensions audit key, doc 11 § 4), and
``dimensions=self._dimensions`` is always passed to the API so the returned
length satisfies the ``Embedding`` validator by construction (DECISIONS § 2).
Empty / whitespace-only text short-circuits to a full-dimension zero vector with
no API call — OpenAI 400s on empty input, and a single empty string fails an
entire batch request, so empties never reach the wire (DECISIONS § 1).
Technical failures raise ``EmbeddingTechnicalError``.
"""

from __future__ import annotations

import openai
from openai import AsyncOpenAI

from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.embeddings.types import Embedding
from rag_recipes.providers.errors import EmbeddingTechnicalError

__all__ = ["OpenAIEmbeddingProvider"]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """``EmbeddingProvider`` backed by the OpenAI embeddings API.

    The constructor takes ``api_key`` / ``model`` / ``dimensions`` / ``batch_size``
    explicitly (the caller resolves them from ``Settings``); the provider never
    reads ``Settings``. ``client`` is injectable so unit tests supply a stub
    without monkeypatching. When ``client is None`` the SDK client is pinned to
    ``max_retries=0`` so the provider issues exactly one external call per request
    — retry/backoff stays an explicit Epic-10 concern (DECISIONS § 4).
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        dimensions: int,
        batch_size: int = 100,
        client: AsyncOpenAI | None = None,
    ) -> None:
        # The provider is constructible without Settings, so it enforces the full
        # valid range itself rather than relying on the Settings field's cap.
        if not 1 <= batch_size <= 2048:
            raise ValueError(f"batch_size must be in [1, 2048], got {batch_size}")
        self._client = client or AsyncOpenAI(api_key=api_key, max_retries=0)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    @staticmethod
    def _is_empty(text: str) -> bool:
        return text.strip() == ""

    def _embedding(self, vector: list[float]) -> Embedding:
        return Embedding(
            provider="openai",
            model=self._model,
            dimensions=self._dimensions,
            vector=vector,
        )

    async def _embed_chunk(self, chunk: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=chunk, dimensions=self._dimensions
            )
        except openai.APIError as exc:
            raise EmbeddingTechnicalError(str(exc)) from exc
        return [item.embedding for item in response.data]

    async def embed_text(self, text: str) -> Embedding:
        if self._is_empty(text):
            return self._embedding([0.0] * self._dimensions)
        vectors = await self._embed_chunk([text])
        return self._embedding(vectors[0])

    async def embed_batch(self, texts: list[str]) -> list[Embedding]:
        non_empty_indices = [i for i, text in enumerate(texts) if not self._is_empty(text)]
        vectors: dict[int, list[float]] = {}
        for start in range(0, len(non_empty_indices), self._batch_size):
            chunk_indices = non_empty_indices[start : start + self._batch_size]
            chunk_vectors = await self._embed_chunk([texts[i] for i in chunk_indices])
            for index, vector in zip(chunk_indices, chunk_vectors, strict=True):
                vectors[index] = vector
        return [
            self._embedding(vectors.get(i, [0.0] * self._dimensions))
            for i in range(len(texts))
        ]
