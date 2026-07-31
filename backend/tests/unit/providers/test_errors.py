"""Unit tests for the shared provider exception hierarchy."""

from __future__ import annotations

import pytest

from rag_recipes.providers.errors import (
    EmbeddingTechnicalError,
    FileStorageError,
    LLMTechnicalError,
    PdfExtractionError,
    ProviderError,
)

_SUBCLASSES = [
    FileStorageError,
    PdfExtractionError,
    LLMTechnicalError,
    EmbeddingTechnicalError,
]


@pytest.mark.parametrize("subclass", _SUBCLASSES)
def test_subclass_of_provider_error(subclass: type[ProviderError]) -> None:
    assert issubclass(subclass, ProviderError)


@pytest.mark.parametrize("subclass", _SUBCLASSES)
def test_caught_by_provider_error(subclass: type[ProviderError]) -> None:
    with pytest.raises(ProviderError):
        raise subclass("boom")
