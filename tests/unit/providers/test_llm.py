"""Bind FakeLLMProvider to the shared LLM contract suite."""

from __future__ import annotations

import pytest

from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from tests.contracts.llm import LLMContract

_REQUEST = StructuredOutputRequest(
    provider="openai",
    model="gpt-4.1",
    prompt_version="recipe-v1",
    schema_version="recipe.v1",
    input="extract this",
    json_schema={"type": "object"},
)
_OUTPUT = {"title": "Soup", "ingredients": [{"name": "salt"}]}


class TestFakeLLM(LLMContract):
    @pytest.fixture
    def provider(self) -> FakeLLMProvider:
        return FakeLLMProvider({FakeLLMProvider.request_hash(_REQUEST): _OUTPUT})

    @pytest.fixture
    def sample_request(self) -> StructuredOutputRequest:
        return _REQUEST

    @pytest.fixture
    def failure_provider(self) -> FakeLLMProvider:
        return FakeLLMProvider(fail_technically=True)
