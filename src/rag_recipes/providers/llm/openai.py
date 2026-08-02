"""OpenAILLMProvider — structured generation via the OpenAI SDK (doc 11 § 3, doc 13 § 4b).

The OpenAI native structured-output API sits behind our own ``LLMProvider``
interface — no ``instructor``, no ``PydanticAI`` (doc 13 § 4b). Validation
outcomes are first-class audit data, so the provider never retries: technical
failures raise ``LLMTechnicalError`` (→ ``ExtractionRun.status = failed``);
un-parseable / non-object / refused / truncated output is returned with
``output_json=None`` and a ``parse_error``, always preserving ``raw_text`` for
debugging (the extraction layer, Epic 9, decides rejection). No post-parse
JSON-Schema validation happens here (``recipe.v1`` validation is Epic 9).

Since Epic 23.4 the class serves any OpenAI-**compatible** endpoint, not just
OpenAI: ``base_url`` retargets the transport, ``provider`` carries the identity
that endpoint's runs are recorded under, and ``structured_output_mode`` selects
how schema-constrained output is requested. How strongly conformance is enforced
on the wire therefore depends on the mode — ``json_schema`` and ``strict_tool``
constrain decoding, plain ``tool`` is advisory. All three share one parse path,
so the *response* contract is identical whichever is in use.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from typing import Any, Literal

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)

from rag_recipes.providers._observability import ProviderObservability, TraceContext
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    TokenUsage,
)

__all__ = ["STRUCTURED_OUTPUT_MODES", "OpenAILLMProvider", "StructuredOutputMode"]

_DISALLOWED_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_TRUNCATING_FINISH_REASONS = frozenset({"length", "content_filter"})

#: How the provider asks for schema-constrained output (Epic 23.4).
#:
#: - ``json_schema`` — OpenAI's ``response_format={"type": "json_schema", strict: true}``.
#:   The default, and byte-identical to pre-23.4 behaviour.
#: - ``strict_tool`` — a forced function call whose schema carries ``strict: true``.
#:   The only schema-constrained mechanism some OpenAI-compatible vendors offer:
#:   DeepSeek's ``response_format.type`` accepts ``text`` and ``json_object`` only.
#: - ``tool`` — the same forced call without ``strict``. The retreat when a strict
#:   grammar is rejected for size: Epic 19 found Anthropic 400s with "compiled
#:   grammar is too large" on ``recipe.v1``, and the same ceiling may exist
#:   elsewhere. Downstream Pydantic + hard/soft validation backstops correctness.
StructuredOutputMode = Literal["json_schema", "strict_tool", "tool"]
STRUCTURED_OUTPUT_MODES: frozenset[str] = frozenset({"json_schema", "strict_tool", "tool"})

_TOOL_DESCRIPTION = (
    "Record the structured result. Call this tool exactly once, passing the result "
    "as its arguments conforming to the provided parameters schema."
)

# Phase 9.5 rate-limit retry tuning. base/cap bound the exponential backoff;
# only an explicit transient ``rate_limit_exceeded`` 429 is retried.
_RETRY_BASE_DELAY_SECONDS = 0.5
_RETRY_MAX_DELAY_SECONDS = 30.0
_RETRYABLE_RATE_LIMIT_CODE = "rate_limit_exceeded"


def _is_retryable_rate_limit(exc: openai.RateLimitError) -> bool:
    """True only for an explicit transient ``rate_limit_exceeded`` 429.

    ``insufficient_quota`` — and any absent or unknown error code — is treated as
    non-retryable: fail fast rather than loop against a quota wall or an
    unclassifiable error (RESEARCH Uncertainty: default to fail-fast).
    """
    return getattr(exc, "code", None) == _RETRYABLE_RATE_LIMIT_CODE


def _rate_limit_retry_after_seconds(exc: openai.RateLimitError) -> float | None:
    """Parse a non-negative ``Retry-After`` (seconds) from the 429 response, else None."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _request_fields(
    schema: dict[str, Any], name: str, mode: StructuredOutputMode
) -> dict[str, Any]:
    """Build the mode-specific structured-output request fields.

    One half of the strategy pair that keeps the three modes from growing three
    parse paths (the other half is ``_extract_payload``). The direct analogue of
    ``anthropic._tool_request_fields``, which exists for the same reason.

    ``json_schema`` emits exactly the fields the pre-23.4 provider emitted, so the
    default path is unchanged rather than merely equivalent. The tool modes emit a
    single forced function and **no** ``response_format`` — a vendor that supports
    one mechanism generally does not accept both in the same request.
    """
    if mode == "json_schema":
        return {
            "response_format": ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=JSONSchema(name=name, strict=True, schema=schema),
            )
        }
    function: dict[str, Any] = {
        "name": name,
        "description": _TOOL_DESCRIPTION,
        "parameters": schema,
    }
    if mode == "strict_tool":
        function["strict"] = True
    return {
        "tools": [{"type": "function", "function": function}],
        "tool_choice": {"type": "function", "function": {"name": name}},
    }


def _extract_payload(
    message: Any, mode: StructuredOutputMode
) -> tuple[str | None, str | None, str | None]:
    """Return ``(payload_text, refusal, extraction_error)`` for a completion message.

    The second half of the strategy pair. It normalises the two transports down to
    the same three values so exactly one ``_parse_error`` decides the outcome —
    which is what makes the modes' parse semantics identical *by construction*
    rather than by test. ``extraction_error`` is the only mode-specific verdict,
    and it is reachable only from the tool modes.

    Read by attribute so both real SDK objects and unit-test stubs work.
    """
    if mode == "json_schema":
        return message.content, message.refusal, None
    refusal = getattr(message, "refusal", None)
    for call in getattr(message, "tool_calls", None) or ():
        function = getattr(call, "function", None)
        if function is not None:
            arguments = getattr(function, "arguments", None)
            return arguments, refusal, None
    # A forced tool call that produced no tool_call block is a rejection, not a
    # technical failure — the same verdict anthropic._parse_error reaches for a
    # missing tool_use block. Ordered *after* the refusal and truncation checks in
    # _parse_error, so a refused or truncated response still reports that cause.
    return None, refusal, "model did not return structured output (no tool_call block)"


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


class OpenAILLMProvider(LLMProvider):
    """``LLMProvider`` backed by OpenAI chat completions in strict JSON-schema mode.

    The constructor takes ``api_key`` and ``default_model`` explicitly (the caller
    resolves them from ``Settings``); the provider never reads ``Settings`` itself.
    ``client`` is injectable so unit tests supply a stub without monkeypatching.
    When ``client is None`` the SDK client is pinned to ``max_retries=0`` so the
    provider issues exactly one external call per request — retry/backoff stays an
    explicit Epic-9 concern and a post-generation timeout can't trigger duplicate
    billable generations the audit record never sees (DECISIONS § 4).

    ``base_url`` retargets the SDK client at any OpenAI-**compatible** endpoint,
    and ``provider`` is the identity label that endpoint's generations are
    recorded under (Epic 23.4). The two travel together: the response provenance
    is always this instance's ``provider``, never the caller-supplied
    ``request.provider``, because the label is what the ``ExtractionRun`` audit
    row and the extraction cache key are keyed on — so it must describe the
    endpoint actually called, not a stale or mistaken caller label. Setting
    ``base_url`` without setting ``provider`` would file another vendor's runs
    under ``openai`` and let them satisfy each other's cache lookups.
    """

    provider = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str,
        provider: str = "openai",
        base_url: str | None = None,
        structured_output_mode: StructuredOutputMode = "json_schema",
        client: AsyncOpenAI | None = None,
        observability: ProviderObservability | None = None,
        max_rate_limit_retries: int = 5,
        request_timeout: float = 60.0,
    ) -> None:
        self._client = client or AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=0
        )
        self.provider = provider
        self.default_model = default_model
        self._mode: StructuredOutputMode = structured_output_mode
        self._obs = observability or ProviderObservability(None, enabled=False)
        # Phase 9.5: explicit retry/timeout, kept separate from the SDK (pinned at
        # max_retries=0) so retries stay inside the one observability span and a
        # post-generation timeout never triggers an unrecorded duplicate call.
        self._max_rate_limit_retries = max_rate_limit_retries
        self._request_timeout = request_timeout

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
            name="openai.generate_structured_output",
            model=request.model,
            input=request.input,
            metadata=metadata,
            trace_context=trace_context,
        ) as observation:
            request_fields = _request_fields(
                request.json_schema, _schema_name(request.schema_version), self._mode
            )
            messages: list[ChatCompletionUserMessageParam] = [
                {"role": "user", "content": request.input}
            ]
            # A technical failure raises inside the ``with`` so the observability
            # wrapper records an ERROR observation before the exception propagates.
            # The retry loop stays inside this one span: a transient
            # rate_limit_exceeded 429 is retried (Retry-After preferred, else
            # capped exponential backoff with jitter) up to
            # ``self._max_rate_limit_retries``; insufficient_quota / unknown codes
            # fail fast. The SDK at max_retries=0 means one billable call per
            # attempt, all recorded under this single trace_generation.
            attempt = 0
            while True:
                try:
                    completion = await self._client.chat.completions.create(
                        model=request.model,
                        messages=messages,
                        timeout=self._request_timeout,
                        **request_fields,
                    )
                    break
                except openai.RateLimitError as exc:
                    if (
                        not _is_retryable_rate_limit(exc)
                        or attempt >= self._max_rate_limit_retries
                    ):
                        raise LLMTechnicalError(str(exc)) from exc
                    retry_after = _rate_limit_retry_after_seconds(exc)
                    delay = (
                        retry_after
                        if retry_after is not None
                        else _backoff_delay(attempt)
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                except openai.APIError as exc:
                    raise LLMTechnicalError(str(exc)) from exc

            choice = completion.choices[0]
            message = choice.message
            content, refusal, extraction_error = _extract_payload(message, self._mode)
            raw_text = content if content is not None else (refusal or "")

            usage = completion.usage
            token_usage = TokenUsage(
                input_tokens=usage.prompt_tokens if usage is not None else 0,
                output_tokens=usage.completion_tokens if usage is not None else 0,
            )

            parse_error = self._parse_error(
                content, refusal, choice.finish_reason, raw_text, extraction_error
            )
            output_json = None if parse_error else json.loads(raw_text)
            response = StructuredOutputResponse(
                output_json=output_json,
                parse_error=parse_error,
                raw_text=raw_text,
                usage=token_usage,
                provider=self.provider,
                model=request.model,
            )

            # Un-parseable / refused / truncated output is a *rejection* (not a
            # technical failure): record it WARNING with the parse_error message,
            # mirroring the success/rejected distinction the audit record keeps.
            status = "success" if parse_error is None else "rejected"
            observation.update(
                output={"parsed": output_json, "raw": raw_text},
                usage_details={
                    "input": token_usage.input_tokens,
                    "output": token_usage.output_tokens,
                },
                metadata={"status": status},
                level="DEFAULT" if parse_error is None else "WARNING",
                status_message=parse_error,
            )
            return response

    @staticmethod
    def _parse_error(
        content: str | None,
        refusal: str | None,
        finish_reason: str,
        raw_text: str,
        extraction_error: str | None = None,
    ) -> str | None:
        """The single parse verdict, shared by every structured-output mode.

        Check order is load-bearing for cross-mode identity: refusal and truncation
        are decided from fields both transports carry, *before* the tool-only
        ``extraction_error``. A refused or truncated tool response therefore reports
        that cause rather than the (also true, but less useful) absence of a
        tool_call block — which is what makes a refusal in ``strict_tool`` mode
        produce the same ``parse_error`` string as a refusal in ``json_schema`` mode.
        """
        if content is None and refusal is not None:
            return f"model refused to generate output: {refusal}"
        if finish_reason in _TRUNCATING_FINISH_REASONS:
            return f"output truncated by provider (finish_reason={finish_reason})"
        if extraction_error is not None:
            return extraction_error
        try:
            parsed = json.loads(raw_text)
        except ValueError:
            return "model output is not valid JSON"
        if not isinstance(parsed, dict):
            return "model output is not a JSON object"
        return None


def _schema_name(schema_version: str) -> str:
    """Map any ``schema_version`` to a valid OpenAI schema name (``^[A-Za-z0-9_-]{1,64}$``)."""
    sanitised = _DISALLOWED_NAME_CHARS.sub("_", schema_version)[:64]
    if sanitised.strip("_"):
        return sanitised
    return f"schema_{hashlib.sha256(schema_version.encode()).hexdigest()[:16]}"
