"""LLMProvider interface (doc 11 § 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

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

    Implementations expose ``provider`` (a stable identity label) and
    ``default_model`` (the model used by default). Callers read these to label
    the ``ExtractionRun`` audit row and the cache key without passing a separate
    ``model`` argument.
    """

    #: Stable provider identity label, e.g. ``"openai"`` / ``"fake"``.
    provider: ClassVar[str]
    #: The model the provider uses by default.
    default_model: str

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
