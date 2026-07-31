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

from typing import Any

import openai
from openai import AsyncOpenAI

from rag_recipes.providers._observability import (
    EMBEDDING_PREVIEW_CHARS,
    ProviderObservability,
    TraceContext,
)
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
        observability: ProviderObservability | None = None,
    ) -> None:
        # The provider is constructible without Settings, so it enforces the full
        # valid range itself rather than relying on the Settings field's cap.
        if not 1 <= batch_size <= 2048:
            raise ValueError(f"batch_size must be in [1, 2048], got {batch_size}")
        self._client = client or AsyncOpenAI(api_key=api_key, max_retries=0)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._obs = observability or ProviderObservability(None, enabled=False)

    @staticmethod
    def _is_empty(text: str) -> bool:
        return text.strip() == ""

    def _trace_metadata(self, *, batch_size: int, preview: str) -> dict[str, Any]:
        # The full corpus text is never recorded — only a bounded preview — so a
        # trace can't leak large or sensitive input (PLAN § secret/PII guard).
        return {
            "provider": "openai",
            "model": self._model,
            "dimensions": self._dimensions,
            "batch_size": batch_size,
            "text_preview": preview[:EMBEDDING_PREVIEW_CHARS],
        }

    @staticmethod
    def _token_upper_bound(text: str) -> int:
        # cl100k_base is byte-level BPE, so token count never exceeds UTF-8 bytes.
        return len(text.encode("utf-8"))

    def _build_request_chunks(self, texts: list[str], indices: list[int]) -> list[list[int]]:
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

    async def _embed_chunk(self, chunk: list[str]) -> tuple[list[list[float]], int, int]:
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
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage is not None else 0
        total_tokens = usage.total_tokens if usage is not None else 0
        return [by_index[i] for i in range(len(chunk))], prompt_tokens, total_tokens

    @staticmethod
    def _usage_details(prompt_tokens: int, total_tokens: int) -> dict[str, int]:
        # Langfuse derives token/cost surfacing from ``usage_details`` + model;
        # embeddings report only input tokens, so ``output`` is always zero.
        return {"input": prompt_tokens, "output": 0, "total": total_tokens}

    async def embed_text(
        self, text: str, *, trace_context: TraceContext | None = None
    ) -> Embedding:
        with self._obs.trace_embedding(
            name="openai.embed_text",
            model=self._model,
            input=text[:EMBEDDING_PREVIEW_CHARS],
            metadata=self._trace_metadata(batch_size=1, preview=text),
            trace_context=trace_context,
        ) as observation:
            if self._is_empty(text):
                observation.update(
                    usage_details=self._usage_details(0, 0), metadata={"status": "success"}
                )
                return self._embedding([0.0] * self._dimensions)
            vectors, prompt_tokens, total_tokens = await self._embed_chunk([text])
            observation.update(
                usage_details=self._usage_details(prompt_tokens, total_tokens),
                metadata={"status": "success"},
            )
            return self._embedding(vectors[0])

    async def embed_batch(
        self, texts: list[str], *, trace_context: TraceContext | None = None
    ) -> list[Embedding]:
        # One observation per batch (not per input) keeps the trace count
        # proportional to the operation rather than to the corpus size.
        preview = texts[0] if texts else ""
        with self._obs.trace_embedding(
            name="openai.embed_batch",
            model=self._model,
            input=preview[:EMBEDDING_PREVIEW_CHARS],
            metadata=self._trace_metadata(batch_size=len(texts), preview=preview),
            trace_context=trace_context,
        ) as observation:
            non_empty_indices = [i for i, text in enumerate(texts) if not self._is_empty(text)]
            # Preflight: reject any single input over the per-input token limit
            # before spending on API calls, naming the offending slot — one
            # oversized input would otherwise 400 the whole request and discard
            # earlier paid chunks.
            for i in non_empty_indices:
                est = self._token_upper_bound(texts[i])
                if est > _MAX_TOKENS_PER_INPUT:
                    raise EmbeddingTechnicalError(
                        f"input {i} is ~{est} tokens, exceeding the "
                        f"{_MAX_TOKENS_PER_INPUT}-token per-input limit"
                    )
            vectors: dict[int, list[float]] = {}
            # Aggregate token usage across internal chunks so the single batch
            # observation reports the true cost of the whole operation.
            prompt_tokens_total = 0
            total_tokens_total = 0
            for chunk_indices in self._build_request_chunks(texts, non_empty_indices):
                chunk_vectors, prompt_tokens, total_tokens = await self._embed_chunk(
                    [texts[i] for i in chunk_indices]
                )
                prompt_tokens_total += prompt_tokens
                total_tokens_total += total_tokens
                for index, vector in zip(chunk_indices, chunk_vectors, strict=True):
                    vectors[index] = vector
            observation.update(
                usage_details=self._usage_details(prompt_tokens_total, total_tokens_total),
                metadata={"status": "success"},
            )
            return [
                self._embedding(vectors.get(i, [0.0] * self._dimensions)) for i in range(len(texts))
            ]
