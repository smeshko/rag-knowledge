"""EmbeddingProvider interface (doc 11 § 4)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_recipes.providers._observability import TraceContext
from rag_recipes.providers.embeddings.types import Embedding

__all__ = ["EmbeddingProvider"]


class EmbeddingProvider(ABC):
    """Abstract provider turning text into embedding vectors.

    Technical failures raise ``EmbeddingTechnicalError`` (``providers.errors``).
    ``trace_context`` carries optional observability fields; whether a trace is
    emitted is the implementation's concern (the Fake ignores it).
    """

    @abstractmethod
    async def embed_text(
        self, text: str, *, trace_context: TraceContext | None = None
    ) -> Embedding:
        """Embed a single ``text`` and return its vector."""

    @abstractmethod
    async def embed_batch(
        self, texts: list[str], *, trace_context: TraceContext | None = None
    ) -> list[Embedding]:
        """Embed each of ``texts`` and return one vector per input."""
