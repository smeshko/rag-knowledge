"""Reusable contract suite for LLMProvider implementations (doc 12 § 2).

Subclasses override ``provider`` and ``request``. ``failure_provider`` skips by
default; an implementation that can deterministically fail (the Fake now, a
real provider in Epic 5) overrides it so the technical-failure contract item is
exercised. Named ``…Contract`` so pytest does not collect the abstract base.
"""

from __future__ import annotations

import pytest

from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest

__all__ = ["LLMContract"]


class LLMContract:
    """Interface guarantees every LLMProvider must satisfy."""

    @pytest.fixture
    def provider(self) -> LLMProvider:
        raise NotImplementedError("subclasses must override the `provider` fixture")

    @pytest.fixture
    def sample_request(self) -> StructuredOutputRequest:
        raise NotImplementedError("subclasses must override the `sample_request` fixture")

    @pytest.fixture
    def failure_provider(self) -> LLMProvider:
        pytest.skip("no deterministic failure mode for this provider")

    async def test_returns_structured_output(
        self, provider: LLMProvider, sample_request: StructuredOutputRequest
    ) -> None:
        response = await provider.generate_structured_output(sample_request)
        assert isinstance(response.output_json, dict)
        assert isinstance(response.raw_text, str)

    async def test_reports_provider_and_model(
        self, provider: LLMProvider, sample_request: StructuredOutputRequest
    ) -> None:
        response = await provider.generate_structured_output(sample_request)
        assert response.provider == sample_request.provider
        assert response.model == sample_request.model

    async def test_reports_usage(
        self, provider: LLMProvider, sample_request: StructuredOutputRequest
    ) -> None:
        response = await provider.generate_structured_output(sample_request)
        assert isinstance(response.usage.input_tokens, int)
        assert isinstance(response.usage.output_tokens, int)

    async def test_technical_failure_surfaces_llm_technical_error(
        self, failure_provider: LLMProvider, sample_request: StructuredOutputRequest
    ) -> None:
        with pytest.raises(LLMTechnicalError):
            await failure_provider.generate_structured_output(sample_request)
