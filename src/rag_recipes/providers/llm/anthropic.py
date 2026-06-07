"""AnthropicLLMProvider — structured generation via the Anthropic SDK (Epic 19.1).

A synchronous Claude provider behind the same ``LLMProvider`` seam as
``OpenAILLMProvider`` (DECISIONS #1–#4). Structured output is requested with the
raw-schema ``output_config={"format": {"type": "json_schema", "schema": …}}``
path — the provider stays schema-agnostic (it receives a dict, not a Pydantic
model) and returns the identical ``output_json`` / ``parse_error`` / ``raw_text``
contract as the OpenAI path. Technical failures raise ``LLMTechnicalError``;
refused / truncated / unparseable output is returned with ``output_json=None``
and a ``parse_error`` (never raised), always preserving ``raw_text``.

The SDK client is pinned to ``max_retries=0`` so the provider owns retry/backoff:
transient ``RateLimitError`` (429) and overloaded (529) responses retry inside
the one observability span (Retry-After preferred, else capped exponential
backoff with jitter); every other ``anthropic.APIError`` raises
``LLMTechnicalError``. A defensive ``_sanitize_schema`` strips JSON-Schema
keywords Claude does not support (DECISIONS #2); a guard test asserts the shared
``recipe.v1`` / ``answer.v1`` schemas carry none today.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
from collections.abc import Iterable
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import (
    JSONOutputFormatParam,
    MessageParam,
    OutputConfigParam,
    RefusalStopDetails,
)

from rag_recipes.providers._observability import ProviderObservability, TraceContext
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    TokenUsage,
)

__all__ = ["AnthropicLLMProvider", "map_message_to_structured_output"]

# Capped exponential backoff bounds for retryable responses (mirrors the OpenAI
# provider's tuning; kept local so the two providers stay decoupled).
_RETRY_BASE_DELAY_SECONDS = 0.5
_RETRY_MAX_DELAY_SECONDS = 30.0
# 529 is the Anthropic "overloaded" status. The SDK maps it to ``OverloadedError``
# (an ``APIStatusError`` subclass not re-exported at the top level), so the
# provider branches on the status code rather than importing a private class.
_OVERLOADED_STATUS_CODE = 529

# JSON-Schema constraint keywords Claude structured outputs do not support
# (DECISIONS #2). Stripped defensively so a future constraint added to the shared
# recipe.v1 schema can't 400 the Anthropic path; value validation stays in the
# extraction layer. ``format`` / ``enum`` / ``$ref`` / ``additionalProperties``
# are supported and left intact.
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "pattern",
    }
)

# Keys whose value is a ``{name: subschema}`` map. The keys *inside* these maps are
# author-controlled field/definition names, NOT schema keywords — so a field
# literally named ``pattern`` / ``maximum`` / etc. must be recursed into, never
# stripped. Sanitization descends into the values of these maps only.
_SUBSCHEMA_MAP_KEYWORDS = frozenset({"properties", "$defs", "definitions", "patternProperties"})


def _backoff_delay(
    attempt: int,
    *,
    base: float = _RETRY_BASE_DELAY_SECONDS,
    cap: float = _RETRY_MAX_DELAY_SECONDS,
) -> float:
    """Capped exponential backoff with full jitter: ``min(base*2**attempt, cap) + U(0, base)``."""
    capped: float = min(base * (2**attempt), cap)
    jitter: float = random.uniform(0.0, base)
    return capped + jitter


def _retry_after_seconds(exc: anthropic.APIStatusError) -> float | None:
    """Parse a non-negative ``Retry-After`` (seconds) from the response, else None."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _extract_text(content: Iterable[Any]) -> str:
    """Return the text of the first ``text`` content block, or ``""`` if none.

    Read by attribute (``.type`` / ``.text``) rather than ``isinstance`` so the
    raw-schema ``output_config`` guarantee (first block is text holding valid
    JSON) is honoured for both the real ``TextBlock`` and the unit-test stub.
    """
    for block in content:
        if getattr(block, "type", None) == "text":
            text = block.text
            return text if isinstance(text, str) else ""
    return ""


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``schema`` with Claude-unsupported keywords stripped.

    Pure function — the caller's dict is never mutated. Strips keywords from each
    schema node and recurses every nested schema, so ``properties``, ``items``,
    ``$defs``/``definitions``, ``anyOf``/``allOf``/``oneOf`` are all covered. The
    *keys* of a ``properties``/``$defs``/``definitions``/``patternProperties`` map
    are field/definition names, not keywords, so they are never stripped — a field
    named ``pattern`` (or any other keyword) survives; only its subschema is
    sanitized (review #1.1).
    """
    sanitized = copy.deepcopy(schema)
    _strip_unsupported(sanitized)
    return sanitized


def _strip_unsupported(node: Any) -> None:
    if isinstance(node, dict):
        for keyword in _UNSUPPORTED_SCHEMA_KEYWORDS:
            node.pop(keyword, None)
        for key, value in node.items():
            if key in _SUBSCHEMA_MAP_KEYWORDS and isinstance(value, dict):
                # value is a {name: subschema} map — recurse into each subschema
                # but treat the map's own keys as opaque names, not keywords.
                for subschema in value.values():
                    _strip_unsupported(subschema)
            else:
                _strip_unsupported(value)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported(item)


class AnthropicLLMProvider(LLMProvider):
    """``LLMProvider`` backed by the Anthropic Messages API in JSON-schema mode.

    Constructor parity with ``OpenAILLMProvider``: ``api_key`` and
    ``default_model`` are resolved from ``Settings`` by the caller (the provider
    never reads ``Settings``); ``client`` is injectable so unit tests supply a
    stub without monkeypatching. When ``client is None`` the SDK client is pinned
    to ``max_retries=0`` so the provider issues exactly one external call per
    attempt and retry/backoff stays an explicit, in-span concern (DECISIONS #3).

    ``max_tokens`` is required by the Messages API; the default keeps requests
    under the SDK's ~16K non-streaming timeout guard. ``stop_reason ==
    "max_tokens"`` (truncation) is surfaced as a ``parse_error``, never a silent
    partial parse (DECISIONS #4). The response provenance ``provider`` is this
    class's own identity, not ``request.provider``.
    """

    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str,
        client: AsyncAnthropic | None = None,
        observability: ProviderObservability | None = None,
        max_rate_limit_retries: int = 5,
        request_timeout: float = 60.0,
        max_tokens: int = 8192,
    ) -> None:
        self._client = client or AsyncAnthropic(api_key=api_key, max_retries=0)
        self.default_model = default_model
        self._obs = observability or ProviderObservability(None, enabled=False)
        self._max_rate_limit_retries = max_rate_limit_retries
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
        *,
        trace_context: TraceContext | None = None,
    ) -> StructuredOutputResponse:
        metadata: dict[str, Any] = {
            "provider": self.provider,
            "prompt_version": request.prompt_version,
            "schema_version": request.schema_version,
        }
        with self._obs.trace_generation(
            name="anthropic.generate_structured_output",
            model=request.model,
            input=request.input,
            metadata=metadata,
            trace_context=trace_context,
        ) as observation:
            output_config: OutputConfigParam = {
                "format": JSONOutputFormatParam(
                    type="json_schema",
                    schema=_sanitize_schema(request.json_schema),
                )
            }
            messages: list[MessageParam] = [
                {"role": "user", "content": request.input}
            ]
            # The retry loop stays inside this one span so all billable attempts
            # are recorded under a single trace_generation and a technical failure
            # raises while the observability wrapper records an ERROR observation.
            # 429 (rate limit) and 529 (overloaded) retry up to
            # ``max_rate_limit_retries`` (Retry-After preferred, else capped
            # backoff with jitter); any other APIError fails fast.
            attempt = 0
            while True:
                try:
                    message = await self._client.with_options(
                        timeout=self._request_timeout
                    ).messages.create(
                        model=request.model,
                        max_tokens=self._max_tokens,
                        messages=messages,
                        output_config=output_config,
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

            response = map_message_to_structured_output(
                message, provider=self.provider, model=request.model
            )

            status = "success" if response.parse_error is None else "rejected"
            observation.update(
                output={"parsed": response.output_json, "raw": response.raw_text},
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
                metadata={"status": status},
                level="DEFAULT" if response.parse_error is None else "WARNING",
                status_message=response.parse_error,
            )
            return response

    async def _sleep_before_retry(
        self, exc: anthropic.APIStatusError, attempt: int
    ) -> None:
        retry_after = _retry_after_seconds(exc)
        delay = retry_after if retry_after is not None else _backoff_delay(attempt)
        await asyncio.sleep(delay)


def map_message_to_structured_output(
    message: Any, *, provider: str, model: str
) -> StructuredOutputResponse:
    """Map an Anthropic ``Message`` to a ``StructuredOutputResponse``.

    The single source of truth for Anthropic Messages-response parse semantics
    (refusal / truncation / unparseable / non-object → ``parse_error``; clean →
    ``output_json``), shared by the synchronous provider and 19.3's batch result
    ingestion so both paths produce byte-identical runs.
    """
    raw_text = _extract_text(message.content)
    usage = message.usage
    token_usage = TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
    parse_error = _parse_error(message.stop_reason, message.stop_details, raw_text)
    output_json = None if parse_error else json.loads(raw_text)
    return StructuredOutputResponse(
        output_json=output_json,
        parse_error=parse_error,
        raw_text=raw_text,
        usage=token_usage,
        provider=provider,
        model=model,
    )


def _parse_error(
    stop_reason: str | None,
    stop_details: RefusalStopDetails | None,
    raw_text: str,
) -> str | None:
    if stop_reason == "refusal":
        explanation = stop_details.explanation if stop_details is not None else None
        if explanation:
            return f"model refused to generate output: {explanation}"
        return "model refused to generate output"
    if stop_reason == "max_tokens":
        return "output truncated by provider (stop_reason=max_tokens)"
    try:
        parsed = json.loads(raw_text)
    except ValueError:
        return "model output is not valid JSON"
    if not isinstance(parsed, dict):
        return "model output is not a JSON object"
    return None
