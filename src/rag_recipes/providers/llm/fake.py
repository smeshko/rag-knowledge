"""In-memory FakeLLMProvider returning canned structured output (doc 13 § 9).

Production code: canned responses keyed on the doc 11 § 3 cache-key hash, a
configurable technical-failure mode, and a copy-on-read call log. A canned
response is either a bare ``output_json`` dict (clean parse — the Fake wraps it,
echoing ``provider``/``model`` and deriving usage) or a full
``StructuredOutputResponse``, which lets a test model the parse-failure contract
state (``output_json=None`` + ``parse_error``, with ``raw_text`` preserved) that
the extraction layer must distinguish from a technical failure. Honours the
state-isolation invariant — caller-owned mutable state is deep-copied in, and
retained mutable state is deep-copied out — so fake-backed tests stay
order-independent and prompt/schema drift can never silently return stale output.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    TokenUsage,
)

__all__ = ["FakeLLMProvider"]


class FakeLLMProvider(LLMProvider):
    """LLM provider returning canned structured output keyed on a request hash."""

    def __init__(
        self,
        responses_by_hash: dict[str, dict[str, Any] | StructuredOutputResponse]
        | None = None,
        *,
        fail_technically: bool = False,
        default_usage: TokenUsage | None = None,
        default_output: dict[str, Any] | StructuredOutputResponse | None = None,
    ) -> None:
        self._responses_by_hash: dict[str, dict[str, Any] | StructuredOutputResponse] = (
            copy.deepcopy(responses_by_hash or {})
        )
        self._fail_technically = fail_technically
        self._default_usage = (
            default_usage.model_copy(deep=True) if default_usage is not None else None
        )
        self._default_output: dict[str, Any] | StructuredOutputResponse | None = (
            copy.deepcopy(default_output)
        )
        self._calls: list[StructuredOutputRequest] = []

    @staticmethod
    def request_hash(request: StructuredOutputRequest) -> str:
        payload = json.dumps(
            [
                request.input,
                request.provider,
                request.model,
                request.prompt_version,
                request.schema_version,
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def calls(self) -> tuple[StructuredOutputRequest, ...]:
        return tuple(call.model_copy(deep=True) for call in self._calls)

    async def generate_structured_output(
        self, request: StructuredOutputRequest
    ) -> StructuredOutputResponse:
        self._calls.append(request.model_copy(deep=True))

        if self._fail_technically:
            raise LLMTechnicalError("fake technical failure")

        canned = self._responses_by_hash.get(self.request_hash(request))
        if canned is None:
            if self._default_output is None:
                raise LookupError(
                    f"no canned response for request_hash {self.request_hash(request)}"
                )
            canned = self._default_output

        if isinstance(canned, StructuredOutputResponse):
            # Echo provider/model from the request, exactly as the dict path does,
            # so a fixture reused across requests can't return mismatched metadata.
            return canned.model_copy(
                update={"provider": request.provider, "model": request.model},
                deep=True,
            )

        chosen = copy.deepcopy(canned)
        usage = (
            self._default_usage.model_copy(deep=True)
            if self._default_usage is not None
            else self._derive_usage(request)
        )
        return StructuredOutputResponse(
            output_json=chosen,
            raw_text=json.dumps(chosen),
            usage=usage,
            provider=request.provider,
            model=request.model,
        )

    @staticmethod
    def _derive_usage(request: StructuredOutputRequest) -> TokenUsage:
        return TokenUsage(input_tokens=len(request.input) // 4, output_tokens=0)
