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


# --- holes found by adversarial review of the plan ---------------------------


def test_structured_output_mode_is_validated_against_the_judge_provider_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact twin of the API-key hole, on the mode setting.

    `llm_structured_output_mode` is global but the two providers need not be.
    openai + json_schema + a DeepSeek judge would otherwise load clean and die
    at the first *judge* call — after extraction has already spent the money.
    """
    with pytest.raises(ValidationError, match="judge_llm_provider"):
        _settings(
            monkeypatch,
            LLM_PROVIDER="openai",
            LLM_STRUCTURED_OUTPUT_MODE="json_schema",
            JUDGE_LLM_PROVIDER="deepseek",
            DEEPSEEK_API_KEY="sk-ds-test",
        )


def test_the_extraction_side_of_the_mode_rule_still_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError, match="llm_provider"):
        _settings(
            monkeypatch,
            LLM_PROVIDER="deepseek",
            DEEPSEEK_API_KEY="sk-ds-test",
            LLM_STRUCTURED_OUTPUT_MODE="json_schema",
        )


def test_an_openai_judge_is_refused_when_the_endpoint_is_retargeted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subtlest way this phase could have failed silently.

    `llm_base_url` / `llm_provider_label` are read by the `openai` registry entry
    whoever asks for it. So with the endpoint retargeted, an "openai" judge is
    built against the *same* third-party endpoint under the *same* identity
    label — a self-judged run that looks correctly configured, and which the
    judge cache key cannot distinguish because both sides carry that one label.
    """
    with pytest.raises(ValidationError, match="ambiguous while llm_base_url is set"):
        _settings(
            monkeypatch,
            LLM_BASE_URL="https://api.together.xyz/v1",
            LLM_PROVIDER_LABEL="together",
            JUDGE_LLM_PROVIDER="openai",
        )


def test_a_genuinely_different_judge_is_fine_with_a_retargeted_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative control: only the ambiguous pairing is refused."""
    settings = _settings(
        monkeypatch,
        LLM_BASE_URL="https://api.together.xyz/v1",
        LLM_PROVIDER_LABEL="together",
        JUDGE_LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="sk-ant-test",
    )
    judge = _build_judge_provider(settings)
    assert judge is not None
    assert judge.provider == "anthropic"


def test_a_judge_configured_identically_to_the_extractor_collapses_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing the judge out explicitly is a natural runbook habit.

    `LLM_PROVIDER=anthropic JUDGE_LLM_PROVIDER=anthropic` is how someone spells
    "judge pinned to Claude" when Claude is also under test. Building a second,
    equivalent client would split the rate-limit retry budget for no benefit.
    """
    settings = _settings(
        monkeypatch,
        LLM_PROVIDER="anthropic",
        JUDGE_LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="sk-ant-test",
    )
    assert _build_judge_provider(settings) is None


def test_importing_the_cli_does_not_import_a_vendor_sdk() -> None:
    """A real assertion, not "the help text rendered".

    `evals/cli.py`'s docstring promises that importing the module — and so
    rendering `--help` — can never reach a provider. Now that there are *two*
    construction seams, that promise has twice the surface. Checked in a clean
    subprocess: an in-process check would pass trivially, since this test module
    imports both providers at the top.
    """
    import subprocess
    import sys

    code = (
        "import sys; import evals.cli; "
        "print(int('openai' in sys.modules), int('anthropic' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "0 0", result.stdout
