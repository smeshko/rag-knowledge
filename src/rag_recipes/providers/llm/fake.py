"""In-memory FakeLLMProvider returning canned structured output (doc 13 § 9).

Production code: canned ``output_json`` keyed on the doc 11 § 3 cache-key hash,
a configurable technical-failure mode, and a copy-on-read call log. Honours the
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
        responses_by_hash: dict[str, dict[str, Any]] | None = None,
        *,
        fail_technically: bool = False,
        default_usage: TokenUsage | None = None,
        default_output: dict[str, Any] | None = None,
    ) -> None:
        self._responses_by_hash: dict[str, dict[str, Any]] = copy.deepcopy(
            responses_by_hash or {}
        )
        self._fail_technically = fail_technically
        self._default_usage = default_usage
        self._default_output = copy.deepcopy(default_output)
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

        output_json = self._responses_by_hash.get(self.request_hash(request))
        if output_json is None:
            if self._default_output is None:
                raise LookupError(
                    f"no canned response for request_hash {self.request_hash(request)}"
                )
            output_json = self._default_output

        chosen = copy.deepcopy(output_json)
        return StructuredOutputResponse(
            output_json=chosen,
            raw_text=json.dumps(chosen),
            usage=self._default_usage or self._derive_usage(request),
            provider=request.provider,
            model=request.model,
        )

    @staticmethod
    def _derive_usage(request: StructuredOutputRequest) -> TokenUsage:
        return TokenUsage(input_tokens=len(request.input) // 4, output_tokens=0)
