"""OpenAILLMProvider — structured generation via the OpenAI SDK (doc 11 § 3, doc 13 § 4b).

The OpenAI native structured-output API sits behind our own ``LLMProvider``
interface — no ``instructor``, no ``PydanticAI`` (doc 13 § 4b). Validation
outcomes are first-class audit data, so the provider never retries: technical
failures raise ``LLMTechnicalError`` (→ ``ExtractionRun.status = failed``);
un-parseable / non-object / refused / truncated output is returned with
``output_json=None`` and a ``parse_error``, always preserving ``raw_text`` for
debugging (the extraction layer, Epic 9, decides rejection). Schema conformance
of a clean-parsed object is enforced on the wire by OpenAI strict mode; 5.1 adds
no post-parse JSON-Schema validation (``recipe.v1`` validation is Epic 9).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

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

__all__ = ["OpenAILLMProvider"]

_DISALLOWED_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_TRUNCATING_FINISH_REASONS = frozenset({"length", "content_filter"})


class OpenAILLMProvider(LLMProvider):
    """``LLMProvider`` backed by OpenAI chat completions in strict JSON-schema mode.

    The constructor takes ``api_key`` and ``default_model`` explicitly (the caller
    resolves them from ``Settings``); the provider never reads ``Settings`` itself.
    ``client`` is injectable so unit tests supply a stub without monkeypatching.
    When ``client is None`` the SDK client is pinned to ``max_retries=0`` so the
    provider issues exactly one external call per request — retry/backoff stays an
    explicit Epic-9 concern and a post-generation timeout can't trigger duplicate
    billable generations the audit record never sees (DECISIONS § 4).

    The response provenance ``provider`` is the provider's own identity
    (``provider``), not the caller-supplied ``request.provider``: this class can
    only ever produce OpenAI generations, so the audit/cache label must reflect
    that regardless of a stale or mistaken caller label.
    """

    provider = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str,
        client: AsyncOpenAI | None = None,
        observability: ProviderObservability | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(api_key=api_key, max_retries=0)
        self.default_model = default_model
        self._obs = observability or ProviderObservability(None, enabled=False)

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
            response_format = ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=JSONSchema(
                    name=_schema_name(request.schema_version),
                    strict=True,
                    schema=request.json_schema,
                ),
            )
            messages: list[ChatCompletionUserMessageParam] = [
                {"role": "user", "content": request.input}
            ]
            # A technical failure raises inside the ``with`` so the observability
            # wrapper records an ERROR observation before the exception propagates.
            try:
                completion = await self._client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    response_format=response_format,
                )
            except openai.APIError as exc:
                raise LLMTechnicalError(str(exc)) from exc

            choice = completion.choices[0]
            message = choice.message
            content = message.content
            refusal = message.refusal
            raw_text = content if content is not None else (refusal or "")

            usage = completion.usage
            token_usage = TokenUsage(
                input_tokens=usage.prompt_tokens if usage is not None else 0,
                output_tokens=usage.completion_tokens if usage is not None else 0,
            )

            parse_error = self._parse_error(content, refusal, choice.finish_reason, raw_text)
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
        content: str | None, refusal: str | None, finish_reason: str, raw_text: str
    ) -> str | None:
        if content is None and refusal is not None:
            return f"model refused to generate output: {refusal}"
        if finish_reason in _TRUNCATING_FINISH_REASONS:
            return f"output truncated by provider (finish_reason={finish_reason})"
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
