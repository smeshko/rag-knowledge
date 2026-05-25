"""Bind FakeLLMProvider to the shared LLM contract suite, plus Fake-only tests."""

from __future__ import annotations

import pytest

from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    TokenUsage,
)
from tests.contracts.llm import LLMContract


def _make_request(*, prompt_version: str = "recipe-v1") -> StructuredOutputRequest:
    return StructuredOutputRequest(
        provider="openai",
        model="gpt-4.1",
        prompt_version=prompt_version,
        schema_version="recipe.v1",
        input="extract this",
        json_schema={"type": "object"},
    )


_REQUEST = _make_request()
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

    async def test_records_calls(self) -> None:
        provider = FakeLLMProvider({FakeLLMProvider.request_hash(_REQUEST): _OUTPUT})
        await provider.generate_structured_output(_REQUEST)
        await provider.generate_structured_output(_REQUEST)
        assert len(provider.calls) == 2
        assert provider.calls[0].prompt_version == "recipe-v1"
        assert provider.calls[0].schema_version == "recipe.v1"

    async def test_records_calls_are_immutable_snapshots(self) -> None:
        request = _make_request()
        provider = FakeLLMProvider({FakeLLMProvider.request_hash(request): _OUTPUT})
        await provider.generate_structured_output(request)

        # (a) mutating the original request after the call must not leak in.
        request.prompt_version = "mutated-original"
        assert provider.calls[0].prompt_version == "recipe-v1"

        # (b) mutating a retrieved snapshot must not corrupt internal state.
        snapshot = provider.calls[0]
        snapshot.prompt_version = "mutated-snapshot"
        assert provider.calls[0].prompt_version == "recipe-v1"

    async def test_unregistered_input_raises(self) -> None:
        provider = FakeLLMProvider()
        with pytest.raises(LookupError):
            await provider.generate_structured_output(_REQUEST)

    async def test_default_output_opt_in(self) -> None:
        default = {"fallback": True}
        provider = FakeLLMProvider(default_output=default)
        response = await provider.generate_structured_output(_REQUEST)
        assert response.output_json == default

    async def test_parse_failure_response_round_trips(self) -> None:
        # A registered full StructuredOutputResponse lets the Fake emit the
        # parse-failure contract state: output_json=None + parse_error, raw_text kept.
        canned = StructuredOutputResponse(
            output_json=None,
            parse_error="model returned non-JSON text",
            raw_text="here is your recipe!{ not json",
            usage=TokenUsage(input_tokens=3, output_tokens=7),
            provider="openai",
            model="gpt-4.1",
        )
        provider = FakeLLMProvider({FakeLLMProvider.request_hash(_REQUEST): canned})
        response = await provider.generate_structured_output(_REQUEST)
        assert response.output_json is None
        assert response.parse_error == "model returned non-JSON text"
        assert response.raw_text == "here is your recipe!{ not json"
        assert response.provider == "openai"
        assert response.model == "gpt-4.1"

    async def test_full_response_echoes_request_provider_and_model(self) -> None:
        # A canned full response reused across requests must still report the
        # calling request's provider/model, not whatever the fixture carried.
        canned = StructuredOutputResponse(
            output_json=None,
            parse_error="bad json",
            raw_text="{nope",
            usage=TokenUsage(input_tokens=1, output_tokens=0),
            provider="anthropic",
            model="claude-3",
        )
        provider = FakeLLMProvider({FakeLLMProvider.request_hash(_REQUEST): canned})
        response = await provider.generate_structured_output(_REQUEST)
        assert response.provider == _REQUEST.provider == "openai"
        assert response.model == _REQUEST.model == "gpt-4.1"

    async def test_schema_nonconforming_output_returned_verbatim(self) -> None:
        # The interface does not validate output_json against json_schema; the
        # Fake returns whatever was registered so the caller decides conformance.
        nonconforming = {"unexpected_field": 123}
        provider = FakeLLMProvider(
            {FakeLLMProvider.request_hash(_REQUEST): nonconforming}
        )
        response = await provider.generate_structured_output(_REQUEST)
        assert response.output_json == nonconforming
        assert response.parse_error is None

    async def test_default_usage_returns_are_isolated(self) -> None:
        # Mutating a returned usage must not poison the retained default_usage
        # nor leak into later calls.
        default_usage = TokenUsage(input_tokens=10, output_tokens=20)
        provider = FakeLLMProvider(
            {FakeLLMProvider.request_hash(_REQUEST): _OUTPUT},
            default_usage=default_usage,
        )
        first = await provider.generate_structured_output(_REQUEST)
        first.usage.input_tokens = 999

        # Mutating the original constructor argument must not leak in either.
        default_usage.output_tokens = 888

        second = await provider.generate_structured_output(_REQUEST)
        assert second.usage.input_tokens == 10
        assert second.usage.output_tokens == 20

    async def test_output_json_returns_are_isolated(self) -> None:
        registered = FakeLLMProvider({FakeLLMProvider.request_hash(_REQUEST): _OUTPUT})
        response = await registered.generate_structured_output(_REQUEST)
        assert response.output_json is not None
        response.output_json["title"] = "Mutated"
        response.output_json["ingredients"][0]["name"] = "pepper"
        again = await registered.generate_structured_output(_REQUEST)
        assert again.output_json == _OUTPUT

        default = {"fallback": True, "nested": {"k": "v"}}
        defaulted = FakeLLMProvider(default_output=default)
        first = await defaulted.generate_structured_output(_REQUEST)
        assert first.output_json is not None
        first.output_json["nested"]["k"] = "mutated"
        second = await defaulted.generate_structured_output(_REQUEST)
        assert second.output_json == default
