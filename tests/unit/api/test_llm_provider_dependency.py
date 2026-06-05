"""Unit tests for get_llm_provider (model-resolution fallback at the boundary)."""

from __future__ import annotations

from typing import Any

from rag_recipes.api.dependencies import get_llm_provider
from rag_recipes.providers.llm.openai import OpenAILLMProvider


def _settings(answer_llm_model: str | None) -> Any:
    class _S:
        openai_api_key = "sk-test"
        llm_model = "gpt-4.1"
        llm_max_rate_limit_retries = 5
        llm_request_timeout_seconds = 60.0

        def __init__(self, model: str | None) -> None:
            self.answer_llm_model = model

    return _S(answer_llm_model)


def test_get_llm_provider_builds_openai_provider() -> None:
    provider = get_llm_provider(settings=_settings(None))
    assert isinstance(provider, OpenAILLMProvider)
    assert provider.provider == "openai"


def test_default_model_falls_back_to_llm_model_when_unset() -> None:
    provider = get_llm_provider(settings=_settings(None))
    assert provider.default_model == "gpt-4.1"


def test_default_model_uses_answer_llm_model_when_set() -> None:
    provider = get_llm_provider(settings=_settings("gpt-4.1-mini"))
    assert provider.default_model == "gpt-4.1-mini"
