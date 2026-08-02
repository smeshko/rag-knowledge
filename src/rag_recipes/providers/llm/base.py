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

    Implementations expose ``provider`` (a stable identity label) and
    ``default_model`` (the model used by default). Callers read these to label
    the ``ExtractionRun`` audit row and the cache key without passing a separate
    ``model`` argument.

    ``provider`` is a **per-instance** attribute, not a class constant (Epic
    23.4): one OpenAI-compatible transport can be pointed at several vendors, so
    the identity belongs to the configured instance rather than to the class.
    Implementations that only ever talk to one vendor may still set it as a
    class-level default. Getting it wrong is not cosmetic — the label is written
    to every ``ExtractionRun`` audit row and is a component of the extraction
    cache key ``(input_hash, provider, model, prompt_version, schema_version)``,
    so a mislabelled provider both corrupts provenance and lets one vendor's
    cached run satisfy another vendor's lookup.
    """

    #: Stable provider identity label, e.g. ``"openai"`` / ``"deepseek"`` / ``"fake"``.
    provider: str
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
