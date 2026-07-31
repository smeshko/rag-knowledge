"""Unit tests for get_llm_provider (provider dispatch + model resolution)."""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.api.dependencies import get_llm_provider
from rag_recipes.providers.llm.anthropic import AnthropicLLMProvider
from rag_recipes.providers.llm.openai import OpenAILLMProvider


def _settings(
    answer_llm_model: str | None = None,
    *,
    llm_provider: str = "openai",
    anthropic_api_key: str | None = "sk-ant-test",
    anthropic_llm_model: str = "claude-sonnet-4-6",
    anthropic_max_tokens: int = 8192,
) -> Any:
    class _S:
        openai_api_key = "sk-test"
        llm_model = "gpt-4.1"
        llm_max_rate_limit_retries = 5
        llm_request_timeout_seconds = 60.0

    s = _S()
    s.llm_provider = llm_provider
    s.answer_llm_model = answer_llm_model
    s.anthropic_api_key = anthropic_api_key
    s.anthropic_llm_model = anthropic_llm_model
    s.anthropic_max_tokens = anthropic_max_tokens
    return s


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


def test_get_llm_provider_builds_anthropic_provider() -> None:
    provider = get_llm_provider(settings=_settings(llm_provider="anthropic"))
    assert isinstance(provider, AnthropicLLMProvider)
    assert provider.provider == "anthropic"
    assert provider.default_model == "claude-sonnet-4-6"


def test_anthropic_ignores_stale_openai_answer_model() -> None:
    # A deployment that set ANSWER_LLM_MODEL for OpenAI then flipped to Anthropic
    # must NOT send the GPT id to Claude — the answer model is anthropic_llm_model.
    provider = get_llm_provider(
        settings=_settings("gpt-4.1-mini", llm_provider="anthropic")
    )
    assert isinstance(provider, AnthropicLLMProvider)
    assert provider.default_model == "claude-sonnet-4-6"


def test_get_llm_provider_anthropic_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="anthropic_api_key"):
        get_llm_provider(
            settings=_settings(llm_provider="anthropic", anthropic_api_key=None)
        )
