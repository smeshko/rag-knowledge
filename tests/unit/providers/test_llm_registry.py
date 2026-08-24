"""Provider registry: construction, model resolution, and the invariants it protects.

The registry replaced four copies of the same ``if settings.llm_provider ==``
block. The tests here are less about the happy path — the pre-existing suites in
``tests/unit/api/test_llm_provider_dependency.py`` and
``tests/unit/ingestion/test_llm_provider_factory.py`` already pin that — and more
about the two invariants that are easy to break silently: importing the registry
must not pull in a vendor SDK (or ``rag-evals --help`` starts loading providers),
and a retargeted ``base_url`` must carry its own identity (or two vendors share
extraction cache entries).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from rag_recipes.config import Settings
from rag_recipes.providers._observability import ProviderObservability
from rag_recipes.providers.llm.anthropic import AnthropicLLMProvider
from rag_recipes.providers.llm.claude_cli import ClaudeCLILLMProvider
from rag_recipes.providers.llm.openai import OpenAILLMProvider
from rag_recipes.providers.llm.registry import (
    build_llm_provider,
    get_spec,
    resolve_extraction_model,
    supported_providers,
)


def _required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    _required_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


# --- the lazy-import invariant ----------------------------------------------


def test_importing_the_registry_does_not_import_a_vendor_sdk() -> None:
    """``evals/cli.py`` promises ``--help`` can never reach a provider.

    That promise now runs through this module, so the factories must import their
    provider classes in-body. A module-level import here would drag the ``openai``
    and ``anthropic`` SDKs into every ``rag-evals`` invocation — and into
    ``rag_recipes.config``, which imports the registry for its allow-list.

    Run in a clean subprocess: an in-process check would pass trivially because
    this test module itself imports both providers at the top.
    """
    code = (
        "import sys; "
        "import rag_recipes.providers.llm.registry; "
        "print(int('openai' in sys.modules), int('anthropic' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "0 0", result.stdout


def test_importing_config_does_not_cycle() -> None:
    """``config`` imports the registry; ``providers._observability`` imports ``config``.

    That is only non-circular because the registry keeps ``Settings`` under
    ``TYPE_CHECKING`` and never imports ``_observability`` at runtime. A future
    edit adding either as a real import breaks ``import rag_recipes.config``
    outright, so pin it.
    """
    for module in ("rag_recipes.config", "rag_recipes.api.dependencies"):
        subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, check=True)


# --- identity and base_url --------------------------------------------------


def test_base_url_without_a_label_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one-env-var path to a poisoned cache key must not exist.

    ``provider`` is part of the extraction cache key, so retargeting the endpoint
    while leaving the label at "openai" would let DeepSeek's runs satisfy
    OpenAI's cache lookups and vice versa.
    """
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    with pytest.raises(ValidationError, match="llm_provider_label"):
        Settings(_env_file=None)


def test_base_url_with_a_label_builds_a_relabelled_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("rag_recipes.providers.llm.openai.AsyncOpenAI", _capture)
    settings = _settings(
        monkeypatch,
        LLM_BASE_URL="https://api.deepseek.com/v1",
        LLM_PROVIDER_LABEL="acme",
    )
    provider = build_llm_provider(settings)

    # Asserted through build_llm_provider, not through the constructor: a test at
    # the class level would pass even if the registry never read the settings.
    assert captured["base_url"] == "https://api.deepseek.com/v1"
    assert captured["max_retries"] == 0
    assert provider.provider == "acme"


def test_structured_output_mode_override_reaches_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LLM_STRUCTURED_OUTPUT_MODE`` is 23.5's documented emergency fallback.

    If the registry does not read it, an operator flips it mid-paid-run and
    nothing changes — the worst kind of no-op.
    """
    settings = _settings(monkeypatch, LLM_STRUCTURED_OUTPUT_MODE="tool")
    provider = build_llm_provider(settings)
    assert isinstance(provider, OpenAILLMProvider)
    assert provider._mode == "tool"


def test_unset_structured_output_mode_uses_the_entry_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_llm_provider(_settings(monkeypatch))
    assert isinstance(provider, OpenAILLMProvider)
    assert provider._mode == "json_schema"


# --- model resolution -------------------------------------------------------


def test_resolve_extraction_model_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, LLM_MODEL="gpt-4.1")
    assert resolve_extraction_model(settings) == "gpt-4.1"


def test_resolve_extraction_model_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="sk-ant-test",
        ANTHROPIC_LLM_MODEL="claude-sonnet-4-6",
    )
    assert resolve_extraction_model(settings) == "claude-sonnet-4-6"
    assert isinstance(build_llm_provider(settings), AnthropicLLMProvider)


def test_explicit_model_overrides_the_spec_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam Phase 23.3's judge provider uses — no ``role`` needed."""
    settings = _settings(monkeypatch, LLM_MODEL="gpt-4.1")
    provider = build_llm_provider(settings, model="gpt-4.1-mini")
    assert provider.default_model == "gpt-4.1-mini"


def test_explicit_provider_name_overrides_the_configured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Also for 23.3: a judge on a different vendor than the extractor."""
    settings = _settings(monkeypatch, ANTHROPIC_API_KEY="sk-ant-test")
    assert settings.llm_provider == "openai"
    provider = build_llm_provider(settings, provider_name="anthropic")
    assert isinstance(provider, AnthropicLLMProvider)


def test_build_llm_provider_forwards_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    obs = ProviderObservability(None, enabled=False)
    provider = build_llm_provider(_settings(monkeypatch), observability=obs)
    assert isinstance(provider, OpenAILLMProvider)
    assert provider._obs is obs


# --- the allow-list and key validators --------------------------------------


def test_allow_list_is_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    supported = supported_providers()
    assert {"openai", "anthropic"} <= supported
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValidationError, match="llm_provider"):
        Settings(_env_file=None)


def test_anthropic_key_message_is_byte_identical_to_the_pre_registry_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validator became generic; its user-facing message must not have.

    This refactor is supposed to change no behaviour, and an operator reading a
    config error is part of the behaviour.
    """
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "anthropic_api_key is required when llm_provider == 'anthropic'" in str(excinfo.value)


def test_openai_spec_has_no_conditional_key_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """``openai_api_key`` is already unconditionally required.

    Attaching a conditional validator to it would add new behaviour — firing on
    an empty string — under a refactor that promises none. So the spec's
    ``api_key_field`` is ``None``, and this pins that branch.
    """
    assert get_spec("openai").api_key_field is None
    settings = _settings(monkeypatch, OPENAI_API_KEY="")
    assert settings.openai_api_key == ""


def test_get_spec_names_the_supported_set() -> None:
    with pytest.raises(ValueError, match="llm_provider must be one of"):
        get_spec("gemini")


# --- deepseek: the third provider that proves the registry generalises -------


def _deepseek_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    return _settings(monkeypatch, LLM_PROVIDER="deepseek", DEEPSEEK_API_KEY="sk-ds-test", **env)


def test_deepseek_builds_an_openai_transport_with_its_own_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("rag_recipes.providers.llm.openai.AsyncOpenAI", _capture)
    provider = build_llm_provider(_deepseek_settings(monkeypatch))

    assert isinstance(provider, OpenAILLMProvider)
    # Shared transport, separate identity. If this said "openai", DeepSeek's runs
    # would satisfy OpenAI's extraction-cache lookups and vice versa.
    assert provider.provider == "deepseek"
    assert provider.default_model == "deepseek-v4-pro"
    assert captured["base_url"] == "https://api.deepseek.com/v1"
    assert captured["max_retries"] == 0


def test_deepseek_defaults_to_strict_tool_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its response_format offers text/json_object only — no json_schema variant."""
    provider = build_llm_provider(_deepseek_settings(monkeypatch))
    assert isinstance(provider, OpenAILLMProvider)
    assert provider._mode == "strict_tool"


def test_deepseek_rejects_json_schema_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A guaranteed 400 must fail at load, not mid-run during a paid eval."""
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_MODE", "json_schema")
    with pytest.raises(ValidationError, match="not supported by"):
        Settings(_env_file=None)


def test_deepseek_accepts_the_non_strict_tool_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 23.5's retreat if strict grammar compilation rejects recipe.v1.

    Epic 19 hit exactly that on Anthropic ("compiled grammar is too large"), so
    the fallback has to be reachable by config change, not by code change.
    """
    settings = _deepseek_settings(monkeypatch, LLM_STRUCTURED_OUTPUT_MODE="tool")
    provider = build_llm_provider(settings)
    assert isinstance(provider, OpenAILLMProvider)
    assert provider._mode == "tool"


def test_deepseek_key_is_required_when_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "deepseek_api_key is required when llm_provider == 'deepseek'" in str(excinfo.value)


def test_deepseek_answer_model_ignores_the_openai_shaped_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ANSWER_LLM_MODEL is a GPT id by convention; it must never reach DeepSeek.

    Epic 19.1 DECISIONS #5 established this for Claude. Registering a third
    provider without extending the rule would reintroduce the identical bug —
    /answers under LLM_PROVIDER=deepseek routing `gpt-4.1-mini` to DeepSeek.
    """
    from rag_recipes.api.dependencies import resolve_answer_model

    settings = _deepseek_settings(monkeypatch, ANSWER_LLM_MODEL="gpt-4.1-mini")
    assert resolve_answer_model(settings) == "deepseek-v4-pro"


def test_deepseek_is_in_the_allow_list() -> None:
    assert "deepseek" in supported_providers()
    assert get_spec("deepseek").api_key_field == "deepseek_api_key"


# --- claude_cli: subscription-quota provider, no API key ---------------------


def _claude_cli_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "claude_cli")
    # No vendor key beyond the unconditionally-required openai_api_key: the CLI
    # authenticates via the local `claude` login, not an env credential.
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_claude_cli_builds_with_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _claude_cli_settings(monkeypatch)
    provider = build_llm_provider(settings)

    assert isinstance(provider, ClaudeCLILLMProvider)
    assert provider.provider == "claude_cli"
    assert provider.default_model == settings.claude_cli_model


def test_claude_cli_threads_binary_and_timeout_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _claude_cli_settings(
        monkeypatch,
        CLAUDE_CLI_BINARY="/opt/homebrew/bin/claude",
        CLAUDE_CLI_TIMEOUT_SECONDS="120",
    )
    provider = build_llm_provider(settings)

    assert isinstance(provider, ClaudeCLILLMProvider)
    assert provider._binary == "/opt/homebrew/bin/claude"
    assert provider._timeout_seconds == 120.0


def test_resolve_extraction_model_claude_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    # This resolution is what eval RunMetadata.llm_model stamps; the pinned full
    # model name (never the `opus` alias) keeps the extraction cache key honest.
    settings = _claude_cli_settings(monkeypatch)
    assert resolve_extraction_model(settings, "claude_cli") == "claude-opus-5"

    overridden = _claude_cli_settings(monkeypatch, CLAUDE_CLI_MODEL="claude-sonnet-5")
    assert resolve_extraction_model(overridden) == "claude-sonnet-5"


def test_claude_cli_is_in_the_allow_list() -> None:
    assert "claude_cli" in supported_providers()
    assert get_spec("claude_cli").api_key_field is None


def test_importing_the_registry_does_not_import_the_claude_cli_module() -> None:
    """The lazy-import invariant extends to in-repo provider modules too."""
    code = (
        "import sys; "
        "import rag_recipes.providers.llm.registry; "
        "print(int('rag_recipes.providers.llm.claude_cli' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "0", result.stdout
