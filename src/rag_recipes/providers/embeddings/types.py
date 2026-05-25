"""Pydantic payload types for the embedding provider (doc 11 § 4)."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["Embedding"]


class Embedding(BaseModel):
    """A single embedding vector with the provider/model that produced it.

    ``dimensions`` is stored explicitly so vector-space comparisons can filter by
    provider + model + dimensions (doc 11 § 4) — matching the future
    ``ChunkEmbedding`` row shape.
    """

    provider: str
    model: str
    dimensions: int
    vector: list[float]
