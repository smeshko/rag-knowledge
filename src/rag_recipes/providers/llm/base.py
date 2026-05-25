"""LLMProvider interface (doc 11 § 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_recipes.providers.llm.types import StructuredOutputRequest, StructuredOutputResponse

__all__ = ["LLMProvider"]


class LLMProvider(ABC):
    """Abstract provider for structured (JSON-Schema-constrained) generation.

    Technical failures (transport, timeout, provider error) raise
    ``LLMTechnicalError`` (``providers.errors``). Schema-rejected or otherwise
    non-conforming output is the caller's decision — the interface just returns
    the response, including ``raw_text`` for debugging.
    """

    @abstractmethod
    async def generate_structured_output(
        self, request: StructuredOutputRequest
    ) -> StructuredOutputResponse:
        """Generate structured output for ``request`` and return the response."""
