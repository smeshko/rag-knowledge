"""LLMProvider interface (doc 11 § 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_recipes.providers._observability import TraceContext
from rag_recipes.providers.llm.types import StructuredOutputRequest, StructuredOutputResponse

__all__ = ["LLMProvider"]


class LLMProvider(ABC):
    """Abstract provider for structured (JSON-Schema-constrained) generation.

    Technical failures (transport, timeout, provider error) raise
    ``LLMTechnicalError`` (``providers.errors``). Schema-rejected or otherwise
    non-conforming output is the caller's decision, not a technical failure: the
    interface returns the response with ``output_json=None`` and ``parse_error``
    set when the output could not be parsed, always preserving ``raw_text`` for
    debugging.
    """

    @abstractmethod
    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
        *,
        trace_context: TraceContext | None = None,
    ) -> StructuredOutputResponse:
        """Generate structured output for ``request`` and return the response.

        ``trace_context`` carries optional observability fields (session/span
        ids, input hash) for tracing; whether and how a trace is emitted is the
        implementation's concern (the Fake ignores it).
        """
