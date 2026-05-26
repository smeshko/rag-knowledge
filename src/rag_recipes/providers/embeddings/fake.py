"""Deterministic in-memory FakeEmbeddingProvider (doc 13 § 9).

Production code: seeds ``random.Random`` from ``sha256`` of the
provider/model/dimensions/text to emit reproducible, correctly-dimensioned
vectors with no paid API call and no numpy dependency (see DECISIONS.md § 2).
Folding provider/model/dimensions into the seed keeps distinct embedding spaces
distinct, so fake-backed retrieval tests can't pass while missing a
provider/model filter.
"""

from __future__ import annotations

import hashlib
import random

from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.embeddings.types import Embedding

__all__ = ["FakeEmbeddingProvider"]


class FakeEmbeddingProvider(EmbeddingProvider):
    """Embedding provider emitting deterministic stdlib-seeded vectors."""

    def __init__(
        self,
        *,
        provider: str = "fake",
        model: str = "fake-embedding",
        dimensions: int = 1536,
    ) -> None:
        self._provider = provider
        self._model = model
        self._dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        # Seed from provider/model/dimensions as well as text so distinct
        # embedding spaces yield distinct vectors — mirroring the architecture
        # invariant that embeddings from different provider/model spaces must
        # never be compared, so fake-backed retrieval tests can't silently pass
        # while missing an embedding_provider/embedding_model filter.
        payload = "\x00".join(
            [self._provider, self._model, str(self._dimensions), text]
        )
        seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dimensions)]

    async def embed_text(self, text: str) -> Embedding:
        return Embedding(
            provider=self._provider,
            model=self._model,
            dimensions=self._dimensions,
            vector=self._vector(text),
        )

    async def embed_batch(self, texts: list[str]) -> list[Embedding]:
        return [await self.embed_text(text) for text in texts]
