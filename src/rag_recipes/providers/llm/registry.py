"""Provider registry — the single ``Settings``→``LLMProvider`` construction seam (Epic 23.4).

Before this module the same ``if settings.llm_provider == "anthropic"`` block was
copied into four places (``api/dependencies.py``, ``ingestion/jobs.py``,
``evals/cli.py``, ``evals/reports.py``), and ``config.py`` carried a literal
``{"openai", "anthropic"}`` allow-list plus a hand-written Anthropic key
validator. Adding a provider meant five coordinated edits, and forgetting one was
silent: ``evals/reports.py``'s copy resolves the model stamped into every eval
``RunMetadata``, so a missed edit there mislabels a committed baseline rather than
raising.

Two invariants this module exists to hold:

**Lazy factories.** Every factory imports its provider class *inside its own
body*. Importing this module therefore never pulls in the ``openai`` or
``anthropic`` SDKs, which is what lets ``config.py`` import ``supported_providers``
cheaply and preserves the promise in ``evals/cli.py``'s docstring that rendering
``--help`` can never issue a live call.

**Typing-only ``Settings``.** ``providers/_observability.py`` already imports
``rag_recipes.config`` at module level, so ``providers`` → ``config`` is an
existing runtime edge. The reverse edge (``config`` importing this module) is only
safe because ``Settings`` and ``ProviderObservability`` are referenced under
``TYPE_CHECKING`` alone. Adding a runtime import of either here would make
``import rag_recipes.config`` fail with a circular import.

Note there is deliberately **no** ``role`` parameter. An earlier design selected
the model field via ``role: Literal["extraction", "answer"]``; that does not
extend to Phase 23.3's judge provider, whose *name* comes from a different
settings field (``judge_llm_provider``) — a ``role="judge"`` would have forced
this module to know about judge settings and re-created the dispatch it deletes.
Callers pass ``provider_name`` and/or ``model`` explicitly instead, so a new role
is a call-site change with no registry edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rag_recipes.config import Settings
    from rag_recipes.providers._observability import ProviderObservability
    from rag_recipes.providers.llm.base import LLMProvider

__all__ = [
    "ProviderSpec",
    "SupportsProviderSelection",
    "build_llm_provider",
    "get_spec",
    "resolve_extraction_model",
    "supported_providers",
]


class SupportsProviderSelection(Protocol):
    """The minimum surface ``resolve_extraction_model`` needs.

    Only ``llm_provider`` is read statically; the model attribute itself is named
    by the selected spec and read dynamically. Deliberately narrower than
    ``Settings`` so ``evals.reports.SettingsLike`` — a five-field protocol whose
    whole point is that report metadata copies named fields rather than dumping
    settings wholesale — can call this without growing a field per provider.
    """

    @property
    def llm_provider(self) -> str: ...


class ProviderFactory(Protocol):
    """Builds a configured provider. Imports its provider class lazily, in-body."""

    def __call__(
        self,
        settings: Settings,
        *,
        model: str,
        observability: ProviderObservability | None,
    ) -> LLMProvider: ...


@dataclass(frozen=True)
class ProviderSpec:
    """Everything the system needs to know about one LLM provider.

    ``model_field`` is the ``Settings`` attribute holding this provider's
    *extraction* model. It is read by ``resolve_extraction_model`` for both
    provider construction and eval run metadata, so the two can never disagree.

    ``api_key_field`` is ``None`` when the key is already unconditionally required
    on ``Settings`` — true of ``openai_api_key``. Attaching a conditional
    validator to an already-required field would add new behaviour (firing on an
    empty string) rather than preserving today's.
    """

    name: str
    factory: ProviderFactory
    model_field: str
    api_key_field: str | None
    default_structured_output_mode: str


def _require_key(settings: Settings, field: str) -> str:
    """Narrow ``str | None`` → ``str`` for mypy, with a defensive runtime check.

    ``Settings``' cross-field validator already guarantees the key is present when
    the provider is selected; this is defence in depth for a provider built
    directly rather than through a validated ``Settings``.
    """
    key = getattr(settings, field)
    if not key:
        raise ValueError(f"{field} is required when llm_provider == '{settings.llm_provider}'")
    return str(key)


def _resolve_mode(settings: Settings, spec: ProviderSpec) -> str:
    """Operator override wins; otherwise the provider's own default.

    ``llm_structured_output_mode`` is ``str | None`` precisely so this is
    expressible — a non-``None`` default could not distinguish "left alone" from
    "explicitly chose json_schema".
    """
    return settings.llm_structured_output_mode or spec.default_structured_output_mode


def _build_openai(
    settings: Settings,
    *,
    model: str,
    observability: ProviderObservability | None,
) -> LLMProvider:
    from rag_recipes.providers.llm.openai import OpenAILLMProvider, StructuredOutputMode

    mode: StructuredOutputMode = _resolve_mode(settings, _OPENAI)  # type: ignore[assignment]
    return OpenAILLMProvider(
        settings.openai_api_key,
        default_model=model,
        # An operator pointing llm_base_url at a third-party endpoint MUST also set
        # llm_provider_label, or that vendor's runs are filed under "openai" and the
        # two share cache entries. A Settings validator enforces the pairing.
        provider=settings.llm_provider_label or "openai",
        base_url=settings.llm_base_url,
        structured_output_mode=mode,
        observability=observability,
        max_rate_limit_retries=settings.llm_max_rate_limit_retries,
        request_timeout=settings.llm_request_timeout_seconds,
    )


def _build_anthropic(
    settings: Settings,
    *,
    model: str,
    observability: ProviderObservability | None,
) -> LLMProvider:
    from rag_recipes.providers.llm.anthropic import AnthropicLLMProvider

    return AnthropicLLMProvider(
        _require_key(settings, "anthropic_api_key"),
        default_model=model,
        observability=observability,
        max_rate_limit_retries=settings.llm_max_rate_limit_retries,
        request_timeout=settings.llm_request_timeout_seconds,
        max_tokens=settings.anthropic_max_tokens,
    )


_OPENAI = ProviderSpec(
    name="openai",
    factory=_build_openai,
    model_field="llm_model",
    api_key_field=None,
    default_structured_output_mode="json_schema",
)

_ANTHROPIC = ProviderSpec(
    name="anthropic",
    factory=_build_anthropic,
    model_field="anthropic_llm_model",
    api_key_field="anthropic_api_key",
    # Non-strict by construction: a strict input_schema compiles a
    # constrained-decoding grammar whose ceiling recipe.v1 exceeds (Epic 19).
    # Recorded for completeness — the Anthropic provider does not take a mode.
    default_structured_output_mode="tool",
)


def _build_deepseek(
    settings: Settings,
    *,
    model: str,
    observability: ProviderObservability | None,
) -> LLMProvider:
    from rag_recipes.providers.llm.openai import OpenAILLMProvider, StructuredOutputMode

    mode: StructuredOutputMode = _resolve_mode(settings, _DEEPSEEK)  # type: ignore[assignment]
    return OpenAILLMProvider(
        _require_key(settings, "deepseek_api_key"),
        default_model=model,
        # Its own identity, NOT "openai" — the transport is shared but the cache
        # key and every ExtractionRun audit row must distinguish the two vendors.
        provider="deepseek",
        base_url=settings.deepseek_base_url,
        structured_output_mode=mode,
        observability=observability,
        max_rate_limit_retries=settings.llm_max_rate_limit_retries,
        request_timeout=settings.llm_request_timeout_seconds,
    )


_DEEPSEEK = ProviderSpec(
    name="deepseek",
    factory=_build_deepseek,
    model_field="deepseek_llm_model",
    api_key_field="deepseek_api_key",
    # DeepSeek's response_format accepts "text" and "json_object" only — there is
    # no json_schema variant — so schema-constrained output has to go through a
    # forced tool call. Pairing this entry with json_schema is rejected at load.
    default_structured_output_mode="strict_tool",
)


def _build_claude_cli(
    settings: Settings,
    *,
    model: str,
    observability: ProviderObservability | None,
) -> LLMProvider:
    from rag_recipes.providers.llm.claude_cli import ClaudeCLILLMProvider

    return ClaudeCLILLMProvider(
        model,
        binary=settings.claude_cli_binary,
        timeout_seconds=settings.claude_cli_timeout_seconds,
        observability=observability,
    )


_CLAUDE_CLI = ProviderSpec(
    name="claude_cli",
    factory=_build_claude_cli,
    model_field="claude_cli_model",
    # No key: auth is the local `claude` binary's own claude.ai login. A missing
    # login surfaces as a clear first-call LLMTechnicalError, not a config error.
    api_key_field=None,
    # Recorded for completeness like the Anthropic entry — the CLI enforces the
    # schema itself via --json-schema; the provider does not consume a mode.
    default_structured_output_mode="json_schema",
)


#: Modes a provider's transport cannot serve, rejected at Settings load rather
#: than at the first (possibly paid) request.
_UNSUPPORTED_MODES: dict[str, frozenset[str]] = {
    "deepseek": frozenset({"json_schema"}),
}


def unsupported_structured_output_modes(provider_name: str) -> frozenset[str]:
    """Modes ``provider_name`` cannot serve. Read by ``Settings``' validator."""
    return _UNSUPPORTED_MODES.get(provider_name, frozenset())


_REGISTRY: dict[str, ProviderSpec] = {
    _OPENAI.name: _OPENAI,
    _ANTHROPIC.name: _ANTHROPIC,
    _DEEPSEEK.name: _DEEPSEEK,
    _CLAUDE_CLI.name: _CLAUDE_CLI,
}


def supported_providers() -> frozenset[str]:
    """Every registered provider name. The source of ``config.py``'s allow-list."""
    return frozenset(_REGISTRY)


def get_spec(name: str) -> ProviderSpec:
    """The spec for ``name``, or ``ValueError`` naming the supported set."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"llm_provider must be one of {sorted(_REGISTRY)}; got {name!r}") from None


def resolve_extraction_model(
    settings: SupportsProviderSelection, provider_name: str | None = None
) -> str:
    """The extraction model for ``provider_name`` (default: the configured provider).

    Public and separate from ``build_llm_provider`` because ``evals/reports.py``
    needs the model *name* without constructing a provider, to stamp
    ``RunMetadata.llm_model``. ``build_llm_provider`` calls this internally so a
    committed baseline can never name a different model than the run used.
    """
    spec = get_spec(provider_name or settings.llm_provider)
    return str(getattr(settings, spec.model_field))


def build_llm_provider(
    settings: Settings,
    *,
    provider_name: str | None = None,
    model: str | None = None,
    observability: ProviderObservability | None = None,
) -> LLMProvider:
    """Build the configured ``LLMProvider``.

    ``provider_name`` defaults to ``settings.llm_provider``; ``model`` defaults to
    that provider's extraction model. Callers needing a different model — the
    answer route, and Phase 23.3's judge — pass it explicitly rather than asking
    this module to know about their settings fields.
    """
    name = provider_name or settings.llm_provider
    spec = get_spec(name)
    resolved_model = model if model is not None else resolve_extraction_model(settings, name)
    return spec.factory(settings, model=resolved_model, observability=observability)
