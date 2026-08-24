"""ClaudeCLILLMProvider tests — fake runner only, no real ``claude`` subprocess.

The envelope fixtures pin the observed CLI 2.1.241 JSON shape (plan RESEARCH.md);
if a CLI upgrade changes the envelope these tests are the tripwire.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rag_recipes.providers._observability import ProviderObservability
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.claude_cli import (
    ClaudeCLILLMProvider,
    CLIResult,
    _run_claude_cli,
    map_envelope_to_structured_output,
)
from rag_recipes.providers.llm.types import StructuredOutputRequest
from tests.contracts.llm import LLMContract

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}

_REQUEST = StructuredOutputRequest(
    provider="claude_cli",
    model="claude-opus-5",
    prompt_version="recipe-v1",
    schema_version="recipe.v1",
    input="Return ok=true",
    json_schema=_SCHEMA,
)


def _envelope(
    *,
    structured_output: Any = None,
    omit_structured_output: bool = False,
    result: str = "done",
    is_error: bool = False,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A CLI JSON envelope pinned to the observed 2.1.241 shape."""
    envelope: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "duration_ms": 25706,
        "duration_api_ms": 25314,
        "num_turns": 2,
        "result": result,
        "session_id": "8e4f7c1a-0000-0000-0000-000000000000",
        "total_cost_usd": 0.0,
        "usage": usage
        if usage is not None
        else {
            "input_tokens": 10,
            "cache_creation_input_tokens": 11376,
            "cache_read_input_tokens": 0,
            "output_tokens": 3230,
        },
        "modelUsage": {"claude-opus-5": {"inputTokens": 10, "outputTokens": 3230}},
    }
    if not omit_structured_output:
        envelope["structured_output"] = structured_output
    return envelope


_SUCCESS_ENVELOPE = _envelope(structured_output={"ok": True})


# === map_envelope_to_structured_output ======================================


def test_map_success_envelope() -> None:
    response = map_envelope_to_structured_output(
        _SUCCESS_ENVELOPE, provider="claude_cli", model="claude-opus-5"
    )
    assert response.output_json == {"ok": True}
    assert response.parse_error is None
    assert response.raw_text == json.dumps({"ok": True})
    assert response.usage.input_tokens == 10 + 11376 + 0
    assert response.usage.output_tokens == 3230
    assert response.provider == "claude_cli"
    assert response.model == "claude-opus-5"


def test_map_missing_structured_output_is_a_rejection() -> None:
    envelope = _envelope(omit_structured_output=True, result="sorry, cannot")
    response = map_envelope_to_structured_output(
        envelope, provider="claude_cli", model="claude-opus-5"
    )
    assert response.output_json is None
    assert response.parse_error is not None
    assert "structured output" in response.parse_error
    # The model's own final text survives for debugging.
    assert response.raw_text == "sorry, cannot"


def test_map_non_dict_structured_output_is_a_rejection() -> None:
    envelope = _envelope(structured_output=[1, 2], result="a list, oddly")
    response = map_envelope_to_structured_output(
        envelope, provider="claude_cli", model="claude-opus-5"
    )
    assert response.output_json is None
    assert response.parse_error == "model output is not a JSON object"
    assert response.raw_text == "a list, oddly"


def test_map_usage_sums_input_with_absent_cache_fields() -> None:
    envelope = _envelope(
        structured_output={"ok": True},
        usage={"input_tokens": 7, "output_tokens": 3},
    )
    response = map_envelope_to_structured_output(
        envelope, provider="claude_cli", model="claude-opus-5"
    )
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3


def test_map_missing_usage_falls_back_to_zero() -> None:
    envelope = _envelope(structured_output={"ok": True})
    del envelope["usage"]
    response = map_envelope_to_structured_output(
        envelope, provider="claude_cli", model="claude-opus-5"
    )
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0


def test_map_is_error_envelope_raises() -> None:
    envelope = _envelope(is_error=True, result="Credit balance too low")
    with pytest.raises(LLMTechnicalError) as excinfo:
        map_envelope_to_structured_output(envelope, provider="claude_cli", model="claude-opus-5")
    assert "Credit balance too low" in str(excinfo.value)


# === provider — fake runner =================================================


@dataclass
class _RunnerCall:
    argv: list[str]
    env: dict[str, str]
    cwd: str
    timeout: float
    cwd_was_empty_dir: bool


@dataclass
class _FakeRunner:
    """Records every invocation; returns a canned result or raises."""

    result: CLIResult | None = None
    error: Exception | None = None
    calls: list[_RunnerCall] = field(default_factory=list)

    async def __call__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: str,
        timeout: float,
    ) -> CLIResult:
        cwd_path = Path(cwd)
        self.calls.append(
            _RunnerCall(
                argv=list(argv),
                env=dict(env),
                cwd=cwd,
                timeout=timeout,
                cwd_was_empty_dir=cwd_path.is_dir() and not any(cwd_path.iterdir()),
            )
        )
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _cli_result(envelope: dict[str, Any]) -> CLIResult:
    return CLIResult(returncode=0, stdout=json.dumps(envelope), stderr="")


def _provider_with(runner: _FakeRunner, **kwargs: Any) -> ClaudeCLILLMProvider:
    return ClaudeCLILLMProvider("claude-opus-5", runner=runner, **kwargs)


async def test_command_construction() -> None:
    runner = _FakeRunner(result=_cli_result(_SUCCESS_ENVELOPE))
    provider = _provider_with(runner, binary="/opt/bin/claude", timeout_seconds=123.0)
    await provider.generate_structured_output(_REQUEST)

    call = runner.calls[0]
    assert call.argv == [
        "/opt/bin/claude",
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(_REQUEST.json_schema),
        "--tools",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--model",
        _REQUEST.model,
        _REQUEST.input,
    ]
    assert call.timeout == 123.0


async def test_subprocess_env_strips_anthropic_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A key in the worker env must never reach the CLI — it would silently switch
    # subscription-quota calls to API billing (DECISIONS #2).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-live")
    monkeypatch.setenv("SOME_OTHER_VAR", "kept")
    runner = _FakeRunner(result=_cli_result(_SUCCESS_ENVELOPE))
    await _provider_with(runner).generate_structured_output(_REQUEST)

    env = runner.calls[0].env
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["SOME_OTHER_VAR"] == "kept"
    assert env.get("PATH") == os.environ.get("PATH")


async def test_cwd_is_fresh_empty_temp_dir_and_cleaned_up() -> None:
    runner = _FakeRunner(result=_cli_result(_SUCCESS_ENVELOPE))
    provider = _provider_with(runner)
    await provider.generate_structured_output(_REQUEST)
    await provider.generate_structured_output(_REQUEST)

    first, second = runner.calls
    assert first.cwd_was_empty_dir
    assert second.cwd_was_empty_dir
    assert first.cwd != second.cwd
    assert not Path(first.cwd).exists()
    assert not Path(second.cwd).exists()


async def test_temp_dir_cleaned_up_on_failure() -> None:
    runner = _FakeRunner(result=CLIResult(returncode=1, stdout="", stderr="boom"))
    with pytest.raises(LLMTechnicalError):
        await _provider_with(runner).generate_structured_output(_REQUEST)
    assert not Path(runner.calls[0].cwd).exists()


async def test_nonzero_exit_raises_with_stderr_excerpt() -> None:
    runner = _FakeRunner(
        result=CLIResult(returncode=1, stdout="", stderr="Invalid API key · please run /login")
    )
    with pytest.raises(LLMTechnicalError) as excinfo:
        await _provider_with(runner).generate_structured_output(_REQUEST)
    message = str(excinfo.value)
    assert "exit" in message
    assert "Invalid API key" in message


async def test_non_json_stdout_raises() -> None:
    runner = _FakeRunner(result=CLIResult(returncode=0, stdout="not json at all", stderr=""))
    with pytest.raises(LLMTechnicalError) as excinfo:
        await _provider_with(runner).generate_structured_output(_REQUEST)
    assert "not valid JSON" in str(excinfo.value)


async def test_non_object_envelope_raises() -> None:
    runner = _FakeRunner(result=CLIResult(returncode=0, stdout="[1, 2]", stderr=""))
    with pytest.raises(LLMTechnicalError):
        await _provider_with(runner).generate_structured_output(_REQUEST)


async def test_is_error_envelope_raises_through_provider() -> None:
    runner = _FakeRunner(result=_cli_result(_envelope(is_error=True, result="quota exhausted")))
    with pytest.raises(LLMTechnicalError) as excinfo:
        await _provider_with(runner).generate_structured_output(_REQUEST)
    assert "quota exhausted" in str(excinfo.value)


async def test_timeout_raises_llm_technical_error() -> None:
    runner = _FakeRunner(error=TimeoutError())
    provider = _provider_with(runner, timeout_seconds=45.0)
    with pytest.raises(LLMTechnicalError) as excinfo:
        await provider.generate_structured_output(_REQUEST)
    assert "45" in str(excinfo.value)


async def test_missing_binary_raises_llm_technical_error() -> None:
    runner = _FakeRunner(error=FileNotFoundError("No such file or directory: 'claude'"))
    with pytest.raises(LLMTechnicalError) as excinfo:
        await _provider_with(runner).generate_structured_output(_REQUEST)
    assert "claude" in str(excinfo.value)


async def test_response_provider_is_claude_cli_regardless_of_request() -> None:
    request = _REQUEST.model_copy(update={"provider": "anthropic", "model": "claude-opus-5"})
    runner = _FakeRunner(result=_cli_result(_SUCCESS_ENVELOPE))
    response = await _provider_with(runner).generate_structured_output(request)
    assert response.provider == "claude_cli"
    assert response.model == request.model


# === observability ==========================================================


class _RecordingObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _FakeLangfuse:
    """``Protocol``-shaped stub satisfying ``LangfuseLike``."""

    def __init__(self) -> None:
        self.start_calls: list[dict[str, Any]] = []
        self.observations: list[_RecordingObservation] = []

    @contextlib.contextmanager
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: str,
        input: Any = None,
        metadata: Any = None,
        model: str | None = None,
    ) -> Iterator[_RecordingObservation]:
        self.start_calls.append(
            {
                "name": name,
                "as_type": as_type,
                "input": input,
                "metadata": metadata,
                "model": model,
            }
        )
        observation = _RecordingObservation()
        self.observations.append(observation)
        yield observation


def _traced_provider(fake: _FakeLangfuse, runner: _FakeRunner) -> ClaudeCLILLMProvider:
    return ClaudeCLILLMProvider(
        "claude-opus-5",
        runner=runner,
        observability=ProviderObservability(fake, enabled=True),
    )


async def test_trace_records_clean_generation() -> None:
    fake = _FakeLangfuse()
    runner = _FakeRunner(result=_cli_result(_SUCCESS_ENVELOPE))
    await _traced_provider(fake, runner).generate_structured_output(_REQUEST)

    call = fake.start_calls[0]
    assert call["as_type"] == "generation"
    assert call["name"] == "claude_cli.generate_structured_output"
    assert call["model"] == _REQUEST.model
    assert call["metadata"]["provider"] == "claude_cli"

    update = fake.observations[0].updates[0]
    assert update["output"] == {"parsed": {"ok": True}, "raw": json.dumps({"ok": True})}
    assert update["usage_details"] == {"input": 11386, "output": 3230}
    assert update["metadata"]["status"] == "success"
    assert update["level"] == "DEFAULT"
    assert update["status_message"] is None


async def test_trace_records_rejected_generation() -> None:
    fake = _FakeLangfuse()
    runner = _FakeRunner(result=_cli_result(_envelope(omit_structured_output=True)))
    response = await _traced_provider(fake, runner).generate_structured_output(_REQUEST)

    assert response.output_json is None
    update = fake.observations[0].updates[0]
    assert update["metadata"]["status"] == "rejected"
    assert update["level"] == "WARNING"
    assert update["status_message"] == response.parse_error


# === default runner (no real subprocess — create_subprocess_exec is faked) ==


@dataclass
class _FakeProcess:
    stdout_bytes: bytes = b""
    stderr_bytes: bytes = b""
    returncode: int = 0
    hang: bool = False
    killed: bool = False
    waited: bool = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.hang:
            await asyncio.sleep(30)
        return self.stdout_bytes, self.stderr_bytes

    def kill(self) -> None:
        self.killed = True
        self.hang = False

    async def wait(self) -> int:
        self.waited = True
        return self.returncode


async def test_default_runner_spawns_with_devnull_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = _FakeProcess(stdout_bytes=b"{}", stderr_bytes=b"warn", returncode=0)
    spawn_calls: list[dict[str, Any]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        spawn_calls.append({"argv": list(argv), **kwargs})
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    result = await _run_claude_cli(
        ["claude", "-p", "hi"], env={"PATH": "/usr/bin"}, cwd=str(tmp_path), timeout=5.0
    )

    call = spawn_calls[0]
    assert call["argv"] == ["claude", "-p", "hi"]
    assert call["stdin"] == asyncio.subprocess.DEVNULL
    assert call["stdout"] == asyncio.subprocess.PIPE
    assert call["stderr"] == asyncio.subprocess.PIPE
    assert call["env"] == {"PATH": "/usr/bin"}
    assert call["cwd"] == str(tmp_path)
    assert result == CLIResult(returncode=0, stdout="{}", stderr="warn")


async def test_default_runner_kills_process_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = _FakeProcess(hang=True)

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(TimeoutError):
        await _run_claude_cli(["claude"], env={}, cwd=str(tmp_path), timeout=0.01)
    assert process.killed
    assert process.waited


# === shared LLM contract ====================================================


class TestClaudeCLILLM(LLMContract):
    @pytest.fixture
    def provider(self) -> ClaudeCLILLMProvider:
        return _provider_with(_FakeRunner(result=_cli_result(_SUCCESS_ENVELOPE)))

    @pytest.fixture
    def sample_request(self) -> StructuredOutputRequest:
        return _REQUEST

    @pytest.fixture
    def failure_provider(self) -> ClaudeCLILLMProvider:
        return _provider_with(
            _FakeRunner(result=CLIResult(returncode=1, stdout="", stderr="login required"))
        )
