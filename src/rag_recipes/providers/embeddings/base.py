"""EmbeddingProvider interface (doc 11 § 4)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_recipes.providers.embeddings.types import Embedding

__all__ = ["EmbeddingProvider"]


class EmbeddingProvider(ABC):
    """Abstract provider turning text into embedding vectors.

    Technical failures raise ``EmbeddingTechnicalError`` (``providers.errors``).
    """

    @abstractmethod
    async def embed_text(self, text: str) -> Embedding:
        """Embed a single ``text`` and return its vector."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[Embedding]:
        """Embed each of ``texts`` and return one vector per input."""
