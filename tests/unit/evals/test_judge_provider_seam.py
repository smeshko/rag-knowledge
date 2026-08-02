"""The judge construction seam and its settings (Epic 23.3 TASK-001).

With one provider serving both extraction and judging, a cross-provider
comparison grades every candidate with itself. These tests pin the two halves of
the fix: the settings that express "judge on something else", and the seam that
builds it — including the two ways it can quietly go wrong (a judge provider
without its API key, and a judge provider handed the *extraction* provider's
model id).
"""

from __future__ import annotations

import pytest
from evals.cli import _build_judge_provider, _build_llm_provider
from pydantic import ValidationError

from rag_recipes.config import Settings
from rag_recipes.providers.llm.anthropic import AnthropicLLMProvider
from rag_recipes.providers.llm.openai import OpenAILLMProvider


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


# --- settings ---------------------------------------------------------------


def test_judge_settings_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert settings.judge_llm_provider is None
    assert settings.judge_llm_model is None


def test_judge_provider_must_be_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="judge_llm_provider"):
        _settings(monkeypatch, JUDGE_LLM_PROVIDER="gemini")


@pytest.mark.parametrize("name", ["openai", "anthropic", "deepseek"])
def test_judge_provider_accepts_every_registered_provider(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    settings = _settings(
        monkeypatch,
        JUDGE_LLM_PROVIDER=name,
        ANTHROPIC_API_KEY="sk-ant-test",
        DEEPSEEK_API_KEY="sk-ds-test",
    )
    assert settings.judge_llm_provider == name


def test_judge_provider_key_is_required_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure must land at load, not at the first judge call.

    By the time a judge call happens, extraction has already run — and in a live
    eval that means money already spent on a run that is about to abort.
    """
    with pytest.raises(ValidationError) as excinfo:
        _settings(monkeypatch, JUDGE_LLM_PROVIDER="anthropic")
    assert "anthropic_api_key is required when judge_llm_provider == 'anthropic'" in str(
        excinfo.value
    )


def test_extraction_provider_key_rule_is_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extending the validator must not change the message it already emitted."""
    with pytest.raises(ValidationError) as excinfo:
        _settings(monkeypatch, LLM_PROVIDER="anthropic")
    assert "anthropic_api_key is required when llm_provider == 'anthropic'" in str(
        excinfo.value
    )


# --- the seam ---------------------------------------------------------------


def test_unset_judge_settings_build_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """None, not "a second identically-configured provider".

    Returning an equivalent-but-distinct provider would double construction and
    break the identity guarantee the driver relies on to keep the unset path
    byte-identical to today's behaviour.
    """
    assert _build_judge_provider(_settings(monkeypatch)) is None


def test_judge_provider_can_differ_from_the_extraction_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch, JUDGE_LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant-test"
    )
    extraction = _build_llm_provider(settings)
    judge = _build_judge_provider(settings)

    # One Settings, two vendors — this is the whole point of the phase.
    assert isinstance(extraction, OpenAILLMProvider)
    assert isinstance(judge, AnthropicLLMProvider)
    assert judge.provider == "anthropic"


def test_judge_model_falls_back_to_the_judge_provider_s_own_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never to llm_model.

    With JUDGE_LLM_PROVIDER set and JUDGE_LLM_MODEL unset, falling back to
    `llm_model` would send `gpt-4.1` to Claude — the identical bug Epic 19.1
    DECISIONS #5 exists to prevent, reintroduced through the judge path.
    """
    settings = _settings(
        monkeypatch,
        LLM_MODEL="gpt-4.1",
        JUDGE_LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="sk-ant-test",
        ANTHROPIC_LLM_MODEL="claude-sonnet-4-6",
    )
    judge = _build_judge_provider(settings)

    assert judge is not None
    assert judge.default_model == "claude-sonnet-4-6"


def test_judge_model_alone_keeps_the_extraction_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different model on the same vendor is a legitimate configuration."""
    settings = _settings(monkeypatch, LLM_MODEL="gpt-4.1", JUDGE_LLM_MODEL="gpt-4.1-mini")
    judge = _build_judge_provider(settings)

    assert judge is not None
    assert judge.provider == "openai"
    assert judge.default_model == "gpt-4.1-mini"


def test_explicit_judge_model_wins_over_the_provider_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch,
        JUDGE_LLM_PROVIDER="anthropic",
        JUDGE_LLM_MODEL="claude-opus-4-6",
        ANTHROPIC_API_KEY="sk-ant-test",
    )
    judge = _build_judge_provider(settings)

    assert judge is not None
    assert judge.default_model == "claude-opus-4-6"
