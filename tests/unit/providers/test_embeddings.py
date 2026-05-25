"""Bind FakeEmbeddingProvider to the shared Embedding contract suite."""

from __future__ import annotations

import pytest

from rag_recipes.providers.embeddings.fake import FakeEmbeddingProvider
from tests.contracts.embeddings import EmbeddingContract


class TestFakeEmbedding(EmbeddingContract):
    @pytest.fixture
    def provider(self) -> FakeEmbeddingProvider:
        return FakeEmbeddingProvider(dimensions=8)

    @pytest.fixture
    def expected_dimensions(self) -> int:
        return 8

    async def test_distinct_provider_model_yield_distinct_vectors(self) -> None:
        # Same text is deterministic within one provider/model, but different
        # provider/model configurations occupy distinct vector spaces — so a
        # fake-backed retrieval test can't pass while missing a provider/model
        # filter (the cross-space-comparison invariant).
        text = "tomato soup"
        old = FakeEmbeddingProvider(model="old", dimensions=8)
        new = FakeEmbeddingProvider(model="new", dimensions=8)
        other_provider = FakeEmbeddingProvider(
            provider="other", model="old", dimensions=8
        )

        assert (await old.embed_text(text)).vector == (await old.embed_text(text)).vector
        assert (await old.embed_text(text)).vector != (await new.embed_text(text)).vector
        assert (await old.embed_text(text)).vector != (
            await other_provider.embed_text(text)
        ).vector
