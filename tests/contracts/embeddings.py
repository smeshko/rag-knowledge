"""Reusable contract suite for EmbeddingProvider implementations (doc 12 § 2).

Subclasses override ``provider`` and ``expected_dimensions``. Named ``…Contract``
so pytest does not collect the abstract base directly.
"""

from __future__ import annotations

import pytest

from rag_recipes.providers.embeddings.base import EmbeddingProvider

__all__ = ["EmbeddingContract"]


class EmbeddingContract:
    """Interface guarantees every EmbeddingProvider must satisfy."""

    @pytest.fixture
    def provider(self) -> EmbeddingProvider:
        raise NotImplementedError("subclasses must override the `provider` fixture")

    @pytest.fixture
    def expected_dimensions(self) -> int:
        raise NotImplementedError("subclasses must override the `expected_dimensions` fixture")

    async def test_embed_text_returns_correct_dimensions(
        self, provider: EmbeddingProvider, expected_dimensions: int
    ) -> None:
        embedding = await provider.embed_text("some text")
        assert embedding.dimensions == expected_dimensions
        assert len(embedding.vector) == expected_dimensions

    async def test_records_provider_and_model(self, provider: EmbeddingProvider) -> None:
        embedding = await provider.embed_text("some text")
        assert isinstance(embedding.provider, str) and embedding.provider
        assert isinstance(embedding.model, str) and embedding.model

    async def test_embed_batch_length_and_order(self, provider: EmbeddingProvider) -> None:
        texts = ["first", "second", "third"]
        batch = await provider.embed_batch(texts)
        assert len(batch) == len(texts)
        single = await provider.embed_text(texts[0])
        assert batch[0].vector == single.vector

    async def test_empty_text_handled_consistently(
        self, provider: EmbeddingProvider, expected_dimensions: int
    ) -> None:
        embedding = await provider.embed_text("")
        assert len(embedding.vector) == expected_dimensions

    async def test_determinism(self, provider: EmbeddingProvider) -> None:
        first = await provider.embed_text("repeatable")
        second = await provider.embed_text("repeatable")
        assert first.vector == second.vector
