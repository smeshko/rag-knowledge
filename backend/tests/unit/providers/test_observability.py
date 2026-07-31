"""Unit tests for the provider observability wrapper (``_observability.py``).

The subject under test is *our* wrapper and factory — the injected Langfuse
client is a typed ``Protocol``-shaped stub (scaffolding used *by* these tests,
not a subject), recording ``start_as_current_observation`` kwargs and the
observation's ``.update(...)`` payloads so we can assert the emitted contract.
The real ``Langfuse`` is never constructed here (it registers a global OTEL
tracer provider).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from typing import Any

import langfuse
import pytest

from rag_recipes.config import Settings
from rag_recipes.providers._observability import (
    ProviderObservability,
    TraceContext,
    build_provider_observability,
)

_SECRET = "sk-super-secret-value"


# --- typed fake Langfuse client ---------------------------------------------


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


class _SessionScopeRecorder:
    """Records the session IDs propagated as trace attributes.

    Mirrors the *module-level* ``langfuse.propagate_attributes`` callable — the
    real SDK has no such client method, so it is injected separately from the
    client (see review #2.1).
    """

    def __init__(self) -> None:
        self.session_ids: list[str] = []

    @contextlib.contextmanager
    def __call__(self, *, session_id: str) -> Iterator[None]:
        self.session_ids.append(session_id)
        yield


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "database_url": "postgresql://u:p@localhost:5432/db",
        "redis_url": "redis://:pwd@localhost:6379/0",
        "redis_password": "",
        "openai_api_key": "sk-test",
    }
    base.update(overrides)
    # Isolate from the developer's real .env so its REDIS_PASSWORD etc. can't
    # collide with the fixture DSNs.
    return Settings(_env_file=None, **base)


def _all_recorded_payloads(fake: _FakeLangfuse) -> str:
    parts: list[str] = [json.dumps(call, default=str) for call in fake.start_calls]
    for obs in fake.observations:
        parts.extend(json.dumps(u, default=str) for u in obs.updates)
    return "\n".join(parts)


# --- disabled path -----------------------------------------------------------


@pytest.mark.parametrize(
    "obs",
    [
        ProviderObservability(_FakeLangfuse(), enabled=False),
        ProviderObservability(None, enabled=True),
    ],
)
def test_disabled_path_yields_noop_and_never_touches_client(
    obs: ProviderObservability,
) -> None:
    fake = obs._client
    with obs.trace_generation(name="x", model="m", input="in", metadata={"k": "v"}) as handle:
        handle.update(output={"parsed": 1}, level="DEFAULT")
    with obs.trace_embedding(name="y", model="m", input="in", metadata={"k": "v"}) as handle:
        handle.update(output={"parsed": 1}, level="DEFAULT")
    # The disabled wrapper builds no payload and never calls the client.
    if isinstance(fake, _FakeLangfuse):
        assert fake.start_calls == []
        assert fake.observations == []


# --- enabled path: generation + embedding -----------------------------------


def test_trace_generation_opens_generation_observation() -> None:
    fake = _FakeLangfuse()
    session = _SessionScopeRecorder()
    obs = ProviderObservability(fake, enabled=True, session_scope=session)
    ctx = TraceContext(
        session_id="sess-1",
        input_source_span_ids=["span-a", "span-b"],
        input_hash="hash-1",
    )
    with obs.trace_generation(
        name="openai.generate_structured_output",
        model="gpt-4.1",
        input="extract this",
        metadata={"provider": "openai", "status": "success"},
        trace_context=ctx,
    ) as handle:
        handle.update(output={"parsed": {"ok": True}})

    call = fake.start_calls[0]
    assert call["as_type"] == "generation"
    assert call["name"] == "openai.generate_structured_output"
    assert call["model"] == "gpt-4.1"
    assert call["input"] == "extract this"
    assert call["metadata"]["provider"] == "openai"
    assert call["metadata"]["status"] == "success"
    # input_source_span_ids / input_hash ride in observation metadata...
    assert call["metadata"]["input_source_span_ids"] == ["span-a", "span-b"]
    assert call["metadata"]["input_hash"] == "hash-1"
    # ...but session_id drives Langfuse session grouping, so it is propagated as a
    # trace attribute via the session scope, never recorded in observation metadata.
    assert "session_id" not in call["metadata"]
    assert session.session_ids == ["sess-1"]
    assert fake.observations[0].updates == [{"output": {"parsed": {"ok": True}}}]


def test_trace_embedding_opens_embedding_observation() -> None:
    fake = _FakeLangfuse()
    obs = ProviderObservability(fake, enabled=True)
    with obs.trace_embedding(
        name="openai.embed_text",
        model="text-embedding-3-small",
        input="preview",
        metadata={"provider": "openai", "dimensions": 1536},
    ) as handle:
        handle.update(usage_details={"input": 3})

    call = fake.start_calls[0]
    assert call["as_type"] == "embedding"
    assert call["model"] == "text-embedding-3-small"
    assert call["metadata"] == {"provider": "openai", "dimensions": 1536}


def test_no_trace_context_leaves_metadata_untouched() -> None:
    fake = _FakeLangfuse()
    obs = ProviderObservability(fake, enabled=True)
    with obs.trace_generation(name="n", model="m", input="i", metadata={"provider": "openai"}):
        pass
    assert fake.start_calls[0]["metadata"] == {"provider": "openai"}


# --- exception path ----------------------------------------------------------


def test_exception_records_error_and_reraises() -> None:
    fake = _FakeLangfuse()
    obs = ProviderObservability(fake, enabled=True)

    class _Boom(Exception):
        pass

    with (
        pytest.raises(_Boom),
        obs.trace_generation(name="n", model="m", input="i", metadata={}) as handle,
    ):
        handle.update(output={"parsed": None})
        raise _Boom("kaboom")

    updates = fake.observations[0].updates
    assert updates[-1] == {
        "level": "ERROR",
        "status_message": "kaboom",
        "metadata": {"status": "failed"},
    }


# --- tracing-failure isolation (Langfuse is auxiliary, never in-band) --------


class _TraceBackendError(Exception):
    pass


class _RaisingObservation:
    def update(self, **kwargs: Any) -> None:
        raise _TraceBackendError("update exploded")


class _RaisingLangfuse:
    """Stub whose backend fails — on observation start and/or ``update``."""

    def __init__(self, *, fail_on_start: bool = False) -> None:
        self._fail_on_start = fail_on_start

    @contextlib.contextmanager
    def start_as_current_observation(self, **kwargs: Any) -> Iterator[_RaisingObservation]:
        if self._fail_on_start:
            raise _TraceBackendError("start exploded")
        yield _RaisingObservation()


def test_trace_start_failure_runs_the_block_untraced() -> None:
    obs = ProviderObservability(_RaisingLangfuse(fail_on_start=True), enabled=True)
    ran = False
    with obs.trace_generation(name="n", model="m", input="i", metadata={}) as handle:
        ran = True
        handle.update(output={"parsed": 1})  # the no-op handle absorbs this
    assert ran  # the provider body still executed despite the trace backend failing


def test_observation_update_failure_is_swallowed() -> None:
    obs = ProviderObservability(_RaisingLangfuse(), enabled=True)
    with obs.trace_generation(name="n", model="m", input="i", metadata={}) as handle:
        handle.update(output={"parsed": 1})  # must not raise _TraceBackendError


def test_provider_exception_survives_a_failing_trace_update() -> None:
    obs = ProviderObservability(_RaisingLangfuse(), enabled=True)

    class _ProviderError(Exception):
        pass

    # The error-path observation.update raises inside the wrapper, but the
    # provider's own exception is what must propagate — not the tracing error.
    with (
        pytest.raises(_ProviderError),
        obs.trace_generation(name="n", model="m", input="i", metadata={}),
    ):
        raise _ProviderError("real failure")


# --- factory -----------------------------------------------------------------


def test_factory_disabled_does_not_construct_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, Any]] = []
    monkeypatch.setattr(langfuse, "Langfuse", lambda **kw: constructed.append(kw), raising=True)
    obs = build_provider_observability(_settings(langfuse_enabled=False))

    assert constructed == []
    # The returned wrapper is a no-op: a trace call touches nothing.
    with obs.trace_generation(name="n", model="m", input="i", metadata={}) as handle:
        handle.update(output={})


def test_factory_enabled_constructs_client_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeClient:
        pass

    def _fake_ctor(**kw: Any) -> _FakeClient:
        captured.update(kw)
        return _FakeClient()

    monkeypatch.setattr(langfuse, "Langfuse", _fake_ctor, raising=True)
    obs = build_provider_observability(
        _settings(
            langfuse_enabled=True,
            langfuse_public_key="pk-1",
            langfuse_secret_key=_SECRET,
            langfuse_host="http://localhost:3001",
        )
    )

    assert captured == {
        "public_key": "pk-1",
        "secret_key": _SECRET,
        "host": "http://localhost:3001",
        "tracing_enabled": True,
    }
    # The session scope must be wired to the real *module-level* function — not a
    # client method, which the SDK does not expose (review #2.1).
    assert obs._session_scope is langfuse.propagate_attributes


def test_session_scope_is_a_module_function_not_a_client_method() -> None:
    # Pins the round-2 #1 SDK-shape contract: the v4 ``Langfuse`` client has no
    # ``propagate_attributes`` method, so wiring it as a client call would silently
    # fall back to the no-op path on the real SDK.
    assert callable(langfuse.propagate_attributes)
    assert not hasattr(langfuse.Langfuse, "propagate_attributes")


# --- secret guard ------------------------------------------------------------


def test_no_secret_leaks_into_recorded_payloads() -> None:
    fake = _FakeLangfuse()
    obs = ProviderObservability(fake, enabled=True)
    with obs.trace_generation(
        name="n",
        model="gpt-4.1",
        input="recipe text",
        metadata={"provider": "openai", "status": "success"},
        trace_context=TraceContext(session_id="sess"),
    ) as handle:
        handle.update(output={"parsed": {"ok": True}}, usage_details={"input": 1})

    assert _SECRET not in _all_recorded_payloads(fake)
