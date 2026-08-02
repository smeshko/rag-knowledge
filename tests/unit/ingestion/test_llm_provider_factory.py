"""Unit tests for ingestion ``_build_llm_provider`` dispatch (Epic 19.1).

Mirrors the answer-path dispatch tests, but the ingestion factory also threads a
``ProviderObservability`` hook (extraction is a traced job) where the answer
factory deliberately passes none.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.ingestion.jobs import _build_llm_provider
from rag_recipes.providers._observability import ProviderObservability
from rag_recipes.providers.llm.anthropic import AnthropicLLMProvider
from rag_recipes.providers.llm.openai import OpenAILLMProvider


def _settings(
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
        # Epic 23.4: the registry's openai factory reads these. Extending a
        # duck-typed stub with fields production now reads is a fixture change,
        # not a behaviour change — every assertion below is unchanged.
        llm_base_url = None
        llm_provider_label = None
        llm_structured_output_mode = None

    s = _S()
    s.llm_provider = llm_provider
    s.anthropic_api_key = anthropic_api_key
    s.anthropic_llm_model = anthropic_llm_model
    s.anthropic_max_tokens = anthropic_max_tokens
    return s


def test_build_openai_provider_passes_observability() -> None:
    obs = ProviderObservability(None, enabled=False)
    provider = _build_llm_provider(_settings(), obs)
    assert isinstance(provider, OpenAILLMProvider)
    assert provider.default_model == "gpt-4.1"
    assert provider._obs is obs


def test_build_anthropic_provider_uses_anthropic_settings() -> None:
    obs = ProviderObservability(None, enabled=False)
    provider = _build_llm_provider(_settings(llm_provider="anthropic"), obs)
    assert isinstance(provider, AnthropicLLMProvider)
    assert provider.default_model == "claude-sonnet-4-6"
    assert provider._max_tokens == 8192
    # The observability hook is threaded through (extraction is a traced job).
    assert provider._obs is obs


def test_build_anthropic_provider_reads_anthropic_max_tokens() -> None:
    provider = _build_llm_provider(
        _settings(llm_provider="anthropic", anthropic_max_tokens=4096), None
    )
    assert isinstance(provider, AnthropicLLMProvider)
    assert provider._max_tokens == 4096


def test_build_anthropic_provider_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="anthropic_api_key"):
        _build_llm_provider(
            _settings(llm_provider="anthropic", anthropic_api_key=None), None
        )
