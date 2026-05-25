"""Pydantic payload types for the embedding provider (doc 11 § 4)."""

from __future__ import annotations

from pydantic import BaseModel, FiniteFloat, model_validator

__all__ = ["Embedding"]


class Embedding(BaseModel):
    """A single embedding vector with the provider/model that produced it.

    ``dimensions`` is stored explicitly so vector-space comparisons can filter by
    provider + model + dimensions (doc 11 § 4) — matching the future
    ``ChunkEmbedding`` row shape. The invariant ``dimensions == len(vector)`` is
    enforced so a provider or fake cannot record metadata that contradicts the
    vector it returns (which would otherwise surface only at DB insert time).
    """

    provider: str
    model: str
    dimensions: int
    vector: list[FiniteFloat]

    @model_validator(mode="after")
    def _dimensions_match_vector(self) -> Embedding:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if self.dimensions != len(self.vector):
            raise ValueError(
                f"dimensions ({self.dimensions}) must equal len(vector) ({len(self.vector)})"
            )
        return self
