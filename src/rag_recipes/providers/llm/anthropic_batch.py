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
crash-recovery does **not** rely on it — the submitter reconciles a stale
``SUBMITTING`` batch by unconditionally reverting it to ``PENDING`` for dedup-safe
re-submission (see DECISIONS #7 and review #2.1).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
import httpx
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
    map_message_to_structured_output,
)
from rag_recipes.providers.llm.types import StructuredOutputResponse

__all__ = [
    "AnthropicBatchProvider",
    "BatchExtractionRequest",
    "BatchResult",
    "BatchStatus",
    "BatchSubmitResult",
]

_T = TypeVar("_T")

# Provider error types that resubmitting cannot fix — mapped to a terminal
# REJECTED result. Everything else (server/overloaded/rate-limit/gateway-timeout)
# is treated as retryable (DECISIONS #2; 19.3 bounds the retry by submit_attempts).
_NON_RETRYABLE_ERROR_TYPES = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "billing_error",
        # An oversized window can't be fixed by re-submitting — don't burn the
        # retry budget on it (review #1.2; defensive — not in the 0.107 union yet).
        "request_too_large_error",
    }
)


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
class BatchStatus:
    """The provider's view of a batch (poller only needs ``processing_status``)."""

    processing_status: str


@dataclass(frozen=True)
class BatchResult:
    """One normalized per-window batch result (19.3)."""

    custom_id: str
    result_type: str  # "succeeded" | "errored" | "expired" | "canceled"
    message: Any | None = None  # the Anthropic Message, for succeeded
    error_type: str | None = None  # the inner provider error type, for errored
    retryable: bool = False  # errored: invalid_request etc. are non-retryable


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

    async def retrieve_batch(self, provider_batch_id: str) -> BatchStatus:
        """Return the provider's current status for a batch (over ``batches.retrieve``)."""
        batch = await self._call_with_retry(
            lambda: self._client.messages.batches.retrieve(
                provider_batch_id, timeout=self._request_timeout
            )
        )
        return BatchStatus(processing_status=batch.processing_status)

    async def iter_results(
        self, provider_batch_id: str
    ) -> AsyncIterator[BatchResult]:
        """Stream a completed batch's per-window results, normalized to ``BatchResult``.

        ``batches.results()`` returns a lazy ``AsyncJSONLDecoder`` that iterates the
        raw ``http_response.aiter_bytes()`` with **no** httpx→anthropic translation
        (unlike ``messages.create``). So a mid-stream connection drop raises a raw
        ``httpx`` error and a truncated/garbled line raises ``json.JSONDecodeError``
        — neither an ``anthropic.APIError``. The seam is the normalization boundary,
        so wrap the iteration and re-raise *all* of those as ``LLMTechnicalError`` —
        otherwise the raw error escapes the poller's per-batch handler and skips
        finalize for healthy batches already ingested this tick, leaving them to be
        reaped FAILED (review #1.1, #2.1, #2.2). ``AnthropicError`` is the SDK base
        (covers ``APIError`` and the bare ``AnthropicError``).
        """
        decoder = await self._call_with_retry(
            lambda: self._client.messages.batches.results(
                provider_batch_id, timeout=self._request_timeout
            )
        )
        try:
            async for entry in decoder:
                yield _normalize_result(entry)
        except (anthropic.AnthropicError, httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMTechnicalError(str(exc)) from exc

    def to_structured_output(
        self, result: BatchResult, *, model: str
    ) -> StructuredOutputResponse:
        """Map a succeeded result's message to a ``StructuredOutputResponse``.

        Uses the same 19.1 mapping the synchronous provider applies, so a batch
        refusal/truncation/unparseable becomes a ``parse_error`` identically.
        """
        return map_message_to_structured_output(
            result.message, provider=self.provider, model=model
        )

    async def _call_with_retry(self, op: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``op`` under the shared retry/error policy (429/529 retry, else raise)."""
        attempt = 0
        while True:
            try:
                return await op()
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
            except anthropic.AnthropicError as exc:
                # SDK base error (covers APIError subclasses *and* the bare
                # AnthropicError, e.g. results() raising "no results_url yet" on an
                # ended-but-not-ready batch race; review #2.4).
                raise LLMTechnicalError(str(exc)) from exc

    async def _sleep_before_retry(
        self, exc: anthropic.APIStatusError, attempt: int
    ) -> None:
        retry_after = _retry_after_seconds(exc)
        delay = retry_after if retry_after is not None else _backoff_delay(attempt)
        await asyncio.sleep(delay)


def _normalize_result(entry: Any) -> BatchResult:
    """Map a provider ``MessageBatchIndividualResponse`` to a ``BatchResult``."""
    result = entry.result
    result_type = result.type
    if result_type == "succeeded":
        return BatchResult(
            custom_id=entry.custom_id,
            result_type="succeeded",
            message=result.message,
        )
    if result_type == "errored":
        error_type = result.error.error.type
        return BatchResult(
            custom_id=entry.custom_id,
            result_type="errored",
            error_type=error_type,
            retryable=error_type not in _NON_RETRYABLE_ERROR_TYPES,
        )
    # expired / canceled carry no message or error.
    return BatchResult(custom_id=entry.custom_id, result_type=result_type)
