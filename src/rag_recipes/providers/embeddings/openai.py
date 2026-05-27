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

# OpenAI embeddings request limits for text-embedding-3-*: a single input may not
# exceed ~8192 tokens, and one request may not exceed 300k tokens in aggregate.
# Exceeding either 400s the *whole* request, so batching by input count alone can
# fail an otherwise-valid batch and waste any chunks already embedded in the same
# call. We carry no tokenizer dependency (tiktoken would add a runtime dep and
# fetch encodings over the network on first use, breaking offline unit tests), so
# token count is *upper-bounded* by UTF-8 byte length: text-embedding-3-* use the
# byte-level cl100k_base BPE, whose token count can never exceed the byte count.
# A character-based average (len/4) would undercount token-dense text (CJK, emoji,
# symbol-heavy) and could still pack an over-limit request; the byte bound is safe
# across all scripts at the cost of over-splitting plain ASCII. It only governs how
# inputs are packed into requests — it never touches returned vectors.
_MAX_TOKENS_PER_INPUT = 8192
_MAX_TOKENS_PER_REQUEST = 300_000


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

    @staticmethod
    def _token_upper_bound(text: str) -> int:
        # cl100k_base is byte-level BPE, so token count never exceeds UTF-8 bytes.
        return len(text.encode("utf-8"))

    def _build_request_chunks(
        self, texts: list[str], indices: list[int]
    ) -> list[list[int]]:
        # Pack non-empty input indices into requests bounded by BOTH the batch_size
        # count cap and the aggregate per-request token budget, preserving order.
        chunks: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for i in indices:
            est = self._token_upper_bound(texts[i])
            over_count = len(current) >= self._batch_size
            over_tokens = current_tokens + est > _MAX_TOKENS_PER_REQUEST
            if current and (over_count or over_tokens):
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(i)
            current_tokens += est
        if current:
            chunks.append(current)
        return chunks

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
        # Map by the response's own ``index`` rather than list position: the API
        # documents same-order results, but the ``index`` field is authoritative,
        # so honouring it turns any ordering drift into a loud error instead of a
        # silently mis-attached vector.
        by_index: dict[int, list[float]] = {}
        for item in response.data:
            if item.index in by_index:
                raise EmbeddingTechnicalError(
                    f"OpenAI returned duplicate embedding index {item.index}"
                )
            by_index[item.index] = item.embedding
        if by_index.keys() != set(range(len(chunk))):
            raise EmbeddingTechnicalError(
                f"OpenAI returned indexes {sorted(by_index)} for a chunk of {len(chunk)}"
            )
        return [by_index[i] for i in range(len(chunk))]

    async def embed_text(self, text: str) -> Embedding:
        if self._is_empty(text):
            return self._embedding([0.0] * self._dimensions)
        vectors = await self._embed_chunk([text])
        return self._embedding(vectors[0])

    async def embed_batch(self, texts: list[str]) -> list[Embedding]:
        non_empty_indices = [i for i, text in enumerate(texts) if not self._is_empty(text)]
        # Preflight: reject any single input over the per-input token limit before
        # spending on API calls, naming the offending slot — one oversized input
        # would otherwise 400 the whole request and discard earlier paid chunks.
        for i in non_empty_indices:
            est = self._token_upper_bound(texts[i])
            if est > _MAX_TOKENS_PER_INPUT:
                raise EmbeddingTechnicalError(
                    f"input {i} is ~{est} tokens, exceeding the "
                    f"{_MAX_TOKENS_PER_INPUT}-token per-input limit"
                )
        vectors: dict[int, list[float]] = {}
        for chunk_indices in self._build_request_chunks(texts, non_empty_indices):
            chunk_vectors = await self._embed_chunk([texts[i] for i in chunk_indices])
            for index, vector in zip(chunk_indices, chunk_vectors, strict=True):
                vectors[index] = vector
        return [
            self._embedding(vectors.get(i, [0.0] * self._dimensions))
            for i in range(len(texts))
        ]
