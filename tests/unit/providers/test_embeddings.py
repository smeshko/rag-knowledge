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
