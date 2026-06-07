"""AnthropicBatchProvider — submit extraction windows to the Message Batches API.

A thin seam over ``AsyncAnthropic().messages.batches`` that keeps all Anthropic
batch request/response shapes out of the ingestion pipeline (DECISIONS #6). 19.2
implements ``submit_batch`` only; ``retrieve``/``results`` and the poller are 19.3.
Each request reuses the 19.1 schema sanitizer and the same ``output_config``
json_schema structured-output path as the synchronous provider — batch requests
support every Messages feature.

Idempotency caveat: the installed ``anthropic`` SDK leaves ``_idempotency_header``
unset (``None``), so it does **not** send an idempotency header for Anthropic, and
the Batches API's idempotency support is unverified. ``submit_batch`` still forwards
a caller-supplied key via ``extra_headers`` best-effort, but the submitter's
crash-recovery does **not** rely on it — TASK-005's reconcile-by-list is the
authoritative mechanism (see DECISIONS #7).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import JSONOutputFormatParam, MessageParam, OutputConfigParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.anthropic import (
    _OVERLOADED_STATUS_CODE,
    _backoff_delay,
    _retry_after_seconds,
    _sanitize_schema,
)

__all__ = [
    "AnthropicBatchProvider",
    "BatchExtractionRequest",
    "BatchInfo",
    "BatchSubmitResult",
]


@dataclass(frozen=True)
class BatchExtractionRequest:
    """One window's batch request — already-rendered input + per-request model/cap."""

    custom_id: str
    input: str
    model: str
    max_tokens: int
    json_schema: dict[str, Any]


@dataclass(frozen=True)
class BatchSubmitResult:
    provider_batch_id: str
    request_count: int


@dataclass(frozen=True)
class BatchInfo:
    """A recent provider batch, used by the submitter's reconciliation list-match."""

    provider_batch_id: str
    request_count: int


class AnthropicBatchProvider:
    """Submits extraction requests to the Anthropic Message Batches API.

    Constructor parity with ``AnthropicLLMProvider`` (19.1): ``client`` is
    injectable, and the default-built client is pinned to ``max_retries=0`` so the
    provider owns retry/backoff (DECISIONS #3). Config-agnostic — the caller passes
    per-request ``model``/``max_tokens`` and the sanitized schema.
    """

    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        client: AsyncAnthropic | None = None,
        max_rate_limit_retries: int = 5,
        request_timeout: float = 60.0,
    ) -> None:
        self._client = client or AsyncAnthropic(api_key=api_key, max_retries=0)
        self._max_rate_limit_retries = max_rate_limit_retries
        self._request_timeout = request_timeout

    async def submit_batch(
        self,
        requests: Sequence[BatchExtractionRequest],
        *,
        idempotency_key: str | None = None,
    ) -> BatchSubmitResult:
        batch_requests: list[Request] = [
            self._build_request(request) for request in requests
        ]
        # Best-effort idempotency header (see module docstring) — harmless if the
        # API ignores it; the submitter's reconciliation is the real safety net.
        extra_headers = (
            {"Idempotency-Key": idempotency_key} if idempotency_key is not None else None
        )

        attempt = 0
        while True:
            try:
                batch = await self._client.messages.batches.create(
                    requests=batch_requests,
                    extra_headers=extra_headers,
                    timeout=self._request_timeout,
                )
                break
            except anthropic.RateLimitError as exc:
                if attempt >= self._max_rate_limit_retries:
                    raise LLMTechnicalError(str(exc)) from exc
                await self._sleep_before_retry(exc, attempt)
                attempt += 1
            except anthropic.APIStatusError as exc:
                if (
                    exc.status_code != _OVERLOADED_STATUS_CODE
                    or attempt >= self._max_rate_limit_retries
                ):
                    raise LLMTechnicalError(str(exc)) from exc
                await self._sleep_before_retry(exc, attempt)
                attempt += 1
            except anthropic.APIError as exc:
                raise LLMTechnicalError(str(exc)) from exc

        return BatchSubmitResult(
            provider_batch_id=batch.id, request_count=len(requests)
        )

    async def list_recent_batches(self, *, limit: int = 100) -> list[BatchInfo]:
        """Return the most recent provider batches (first page only) for reconcile.

        Used by the submitter to confirm whether a stale ``SUBMITTING`` local batch
        was actually accepted after a crash, so it can resolve it without a
        re-submit (DECISIONS #7). Owns retries like ``submit_batch``; reads only the
        first page (recency-ordered) to stay bounded.
        """
        attempt = 0
        while True:
            try:
                page = await self._client.messages.batches.list(
                    limit=limit,
                    extra_headers=None,
                    timeout=self._request_timeout,
                )
                break
            except anthropic.RateLimitError as exc:
                if attempt >= self._max_rate_limit_retries:
                    raise LLMTechnicalError(str(exc)) from exc
                await self._sleep_before_retry(exc, attempt)
                attempt += 1
            except anthropic.APIStatusError as exc:
                if (
                    exc.status_code != _OVERLOADED_STATUS_CODE
                    or attempt >= self._max_rate_limit_retries
                ):
                    raise LLMTechnicalError(str(exc)) from exc
                await self._sleep_before_retry(exc, attempt)
                attempt += 1
            except anthropic.APIError as exc:
                raise LLMTechnicalError(str(exc)) from exc

        return [_batch_info(batch) for batch in page.data]

    @staticmethod
    def _build_request(request: BatchExtractionRequest) -> Request:
        output_config: OutputConfigParam = {
            "format": JSONOutputFormatParam(
                type="json_schema",
                schema=_sanitize_schema(request.json_schema),
            )
        }
        messages: list[MessageParam] = [{"role": "user", "content": request.input}]
        params = MessageCreateParamsNonStreaming(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=messages,
            output_config=output_config,
        )
        return Request(custom_id=request.custom_id, params=params)

    async def _sleep_before_retry(
        self, exc: anthropic.APIStatusError, attempt: int
    ) -> None:
        retry_after = _retry_after_seconds(exc)
        delay = retry_after if retry_after is not None else _backoff_delay(attempt)
        await asyncio.sleep(delay)


def _batch_info(batch: Any) -> BatchInfo:
    """Map a provider ``MessageBatch`` to ``BatchInfo`` (total request count)."""
    counts = batch.request_counts
    total = (
        counts.canceled
        + counts.errored
        + counts.expired
        + counts.processing
        + counts.succeeded
    )
    return BatchInfo(provider_batch_id=batch.id, request_count=total)
