"""ClaudeCLILLMProvider — structured extraction via the local ``claude`` CLI.

An ``LLMProvider`` that spawns ``claude -p --output-format json --json-schema``
per request, drawing the user's claude.ai subscription quota instead of API
billing. The CLI's JSON envelope (``structured_output`` / ``result`` / ``usage``
/ ``is_error``, pinned to the observed 2.1.241 shape) is mapped to the standard
``StructuredOutputResponse`` contract by ``map_envelope_to_structured_output`` —
a pure, tested function mirroring ``anthropic.py``'s
``map_message_to_structured_output``. An unrecognized envelope raises
``LLMTechnicalError``, never a silent partial parse.

Invocation hygiene (plan DECISIONS #1/#2/#4):

- prompt passed as argv with ``stdin=DEVNULL`` — the CLI waits 3s for stdin when
  the stream is open but silent;
- the subprocess env is the parent env **minus** ``ANTHROPIC_API_KEY`` /
  ``ANTHROPIC_AUTH_TOKEN``, so a key exported for the API provider can never
  silently switch these "free" calls to API billing;
- cwd is one empty temp dir per provider instance and all tools / MCP servers /
  session persistence are disabled, keeping the CLI context minimal and
  deterministic. Per-instance (not per-call) because the CLI embeds cwd in its
  system prompt, making cwd part of the server-side prompt-cache key: a fresh
  dir per call re-created ~3.7K tokens every window (TASK-005).

There is deliberately no retry loop (DECISIONS #5): Max-quota exhaustion is an
hours-scale wall, not a transient — it raises ``LLMTechnicalError`` and the
pipeline's resume semantics plus the extraction cache make re-runs cheap. The
``runner`` callable is injectable so unit tests never execute a real subprocess.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag_recipes.providers._observability import ProviderObservability, TraceContext
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    TokenUsage,
)

__all__ = [
    "CLIResult",
    "ClaudeCLILLMProvider",
    "extract_billing_details",
    "map_envelope_to_structured_output",
]

# Credential vars stripped from the subprocess env (DECISIONS #2). The CLI
# prefers ``ANTHROPIC_API_KEY`` over the claude.ai login, so leaking either var
# through would silently bill the API key instead of drawing subscription quota.
_STRIPPED_ENV_VARS = frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"})

# Error messages carry at most this much of stderr / stdout / result text.
_EXCERPT_CHARS = 500


@dataclass(frozen=True)
class CLIResult:
    """A completed CLI invocation: exit code plus decoded stdout/stderr."""

    returncode: int
    stdout: str
    stderr: str


class CLIRunner(Protocol):
    """Executes one CLI invocation. Raises ``TimeoutError`` on timeout."""

    async def __call__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: str,
        timeout: float,
    ) -> CLIResult: ...


async def _run_claude_cli(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: str,
    timeout: float,
) -> CLIResult:
    """Default runner: ``create_subprocess_exec`` bounded by ``wait_for``.

    ``stdin=DEVNULL`` skips the CLI's 3s wait-for-stdin when no data is piped
    (DECISIONS #1). On timeout the process is killed and reaped before the
    ``TimeoutError`` propagates, so no orphan keeps burning quota.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env),
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    return CLIResult(
        returncode=process.returncode if process.returncode is not None else 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


def _excerpt(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _EXCERPT_CHARS else text[:_EXCERPT_CHARS] + "…"


def _failure_summary(stdout: str) -> str:
    """The CLI's own explanation of a failure, pulled out of its stdout envelope.

    A head-truncated excerpt is nearly useless here: the envelope leads with
    ``usage``/``session_id``/timing and puts the human-readable ``result`` near
    the END, so the first 500 characters are exactly the part that says nothing.
    A real failure surfaced as 500 characters of zero token counts with the
    reason cut off.

    Pulls ``result`` (and the diagnostic ``subtype``/``stop_reason``) to the
    front, falling back to the head excerpt when stdout is not a JSON object —
    which is itself the interesting case for a crash or a usage message.
    """
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return _excerpt(stdout)
    if not isinstance(envelope, dict):
        return _excerpt(stdout)
    parts: list[str] = []
    for key in ("subtype", "stop_reason", "api_error_status", "terminal_reason"):
        value = envelope.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    result = envelope.get("result")
    if isinstance(result, str) and result.strip():
        parts.append(f"result={_excerpt(result)}")
    return " ".join(parts) if parts else _excerpt(stdout)


class ClaudeCLILLMProvider(LLMProvider):
    """``LLMProvider`` backed by the local ``claude`` binary in ``-p`` mode.

    Constructor parity with the API providers: configuration is resolved from
    ``Settings`` by the caller — the provider never reads ``Settings``. There is
    no API key; auth is the ``claude`` binary's own claude.ai login, and a
    missing login surfaces as a clear first-call ``LLMTechnicalError`` (the CLI
    exits non-zero). ``runner`` is injectable so tests supply a fake instead of
    spawning processes.
    """

    provider = "claude_cli"

    def __init__(
        self,
        default_model: str,
        *,
        binary: str = "claude",
        timeout_seconds: float = 300.0,
        runner: CLIRunner | None = None,
        observability: ProviderObservability | None = None,
    ) -> None:
        self.default_model = default_model
        self._binary = binary
        self._timeout_seconds = timeout_seconds
        self._runner: CLIRunner = runner if runner is not None else _run_claude_cli
        self._obs = observability or ProviderObservability(None, enabled=False)
        # One hermetic cwd for the instance's lifetime, NOT per call: the CLI
        # embeds cwd in its system prompt, so cwd is part of the server-side
        # prompt-cache key — a fresh dir per call re-created ~3.7K tokens on
        # every window (TASK-005 probe: cache_creation 3704 → 0 with a shared
        # dir). Cleanup runs via TemporaryDirectory's own finalizer.
        self._cwd = tempfile.TemporaryDirectory(prefix="claude-cli-")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
        *,
        trace_context: TraceContext | None = None,
    ) -> StructuredOutputResponse:
        metadata: dict[str, Any] = {
            "provider": self.provider,
            "prompt_version": request.prompt_version,
            "schema_version": request.schema_version,
        }
        with self._obs.trace_generation(
            name="claude_cli.generate_structured_output",
            model=request.model,
            input=request.input,
            metadata=metadata,
            trace_context=trace_context,
        ) as observation:
            result = await self._invoke(request)
            if result.returncode != 0:
                # Both streams, because the CLI routinely fails with an EMPTY
                # stderr and its actual diagnosis — an error envelope, an auth or
                # quota message — on stdout. Reporting stderr alone produced
                # "exited with code 1: " and threw the only evidence away.
                raise LLMTechnicalError(
                    f"claude CLI exited with code {result.returncode}: "
                    f"stderr={_excerpt(result.stderr) or '(empty)'} "
                    f"stdout={_failure_summary(result.stdout) or '(empty)'}"
                )
            try:
                envelope = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise LLMTechnicalError(
                    f"claude CLI stdout is not valid JSON: {_excerpt(result.stdout)}"
                ) from exc
            if not isinstance(envelope, dict):
                raise LLMTechnicalError(
                    f"claude CLI envelope is not a JSON object: {_excerpt(result.stdout)}"
                )

            response = map_envelope_to_structured_output(
                envelope, provider=self.provider, model=request.model
            )

            status = "success" if response.parse_error is None else "rejected"
            observation.update(
                output={"parsed": response.output_json, "raw": response.raw_text},
                # The CLI's own accounting, forwarded verbatim (see
                # ``extract_billing_details``). ``StructuredOutputResponse`` is a
                # cross-provider contract and must not grow CLI-specific fields,
                # so this rides on the observation instead — which is also where
                # it is queryable for quota tracking.
                metadata={"status": status, **extract_billing_details(envelope)},
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
                level="DEFAULT" if response.parse_error is None else "WARNING",
                status_message=response.parse_error,
            )
            return response

    async def _invoke(self, request: StructuredOutputRequest) -> CLIResult:
        """Run the CLI once in the instance's hermetic cwd with a scrubbed env."""
        argv = [
            self._binary,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(request.json_schema),
            # Hermetic invocation (DECISIONS #4): no tools, no MCP servers
            # (--strict-mcp-config with no --mcp-config), no session persistence.
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--model",
            request.model,
            request.input,
        ]
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV_VARS}
        try:
            return await self._runner(
                argv, env=env, cwd=self._cwd.name, timeout=self._timeout_seconds
            )
        except TimeoutError as exc:
            raise LLMTechnicalError(f"claude CLI timed out after {self._timeout_seconds}s") from exc
        except OSError as exc:
            raise LLMTechnicalError(
                f"failed to spawn claude CLI binary {self._binary!r}: {exc}"
            ) from exc


def extract_billing_details(envelope: dict[str, Any]) -> dict[str, Any]:
    """The CLI's own cost/cache accounting, for quota tracking.

    ``TokenUsage`` collapses the CLI's three-way input split into one honest
    context size, which is right for the cross-provider contract but destroys
    the only thing that makes a *cost* estimate accurate: fresh, cache-creation
    and cache-read tokens bill at roughly 1x, 1.25x and 0.1x. Re-deriving spend
    from the collapsed figure over-charges every cached window several-fold.

    ``total_cost_usd`` is better still — Anthropic's own number for the call,
    covering the sub-model turns (the CLI dispatches some work to Haiku) that no
    client-side estimate from the Opus rate card can see. Missing keys are simply
    absent from the result: this is telemetry, and it must never fail a call.
    """
    usage = envelope.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details: dict[str, Any] = {}
    cost = envelope.get("total_cost_usd")
    if isinstance(cost, int | float):
        details["cli_total_cost_usd"] = float(cost)
    for source, key in (
        ("input_tokens", "cli_input_tokens_fresh"),
        ("cache_creation_input_tokens", "cli_cache_creation_tokens"),
        ("cache_read_input_tokens", "cli_cache_read_tokens"),
        ("output_tokens", "cli_output_tokens"),
    ):
        value = usage.get(source)
        if isinstance(value, int):
            details[key] = value
    return details


def map_envelope_to_structured_output(
    envelope: dict[str, Any], *, provider: str, model: str
) -> StructuredOutputResponse:
    """Map a CLI JSON envelope to a ``StructuredOutputResponse``.

    The single source of truth for CLI envelope semantics, pinned to the
    observed 2.1.241 shape. ``is_error: true`` is a technical failure (the CLI
    itself failed — auth, quota, crash) and raises. A well-formed envelope
    without a ``structured_output`` object is a *rejection*: ``output_json=None``
    plus ``parse_error``, with the CLI's final ``result`` text preserved as
    ``raw_text`` for debugging. The CLI splits input tokens three ways (fresh /
    cache-creation / cache-read); their sum is the honest context size.
    """
    if envelope.get("is_error"):
        result_text = envelope.get("result")
        detail = _excerpt(result_text) if isinstance(result_text, str) else str(result_text)
        raise LLMTechnicalError(f"claude CLI reported an error result: {detail}")

    usage = envelope.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    token_usage = TokenUsage(
        input_tokens=(
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
        ),
        output_tokens=int(usage.get("output_tokens") or 0),
    )

    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        return StructuredOutputResponse(
            output_json=structured,
            parse_error=None,
            raw_text=json.dumps(structured),
            usage=token_usage,
            provider=provider,
            model=model,
        )

    result_text = envelope.get("result")
    raw_text = result_text if isinstance(result_text, str) else ""
    parse_error = (
        "model output is not a JSON object"
        if structured is not None
        else "model did not return structured output (no structured_output in envelope)"
    )
    return StructuredOutputResponse(
        output_json=None,
        parse_error=parse_error,
        raw_text=raw_text,
        usage=token_usage,
        provider=provider,
        model=model,
    )
