"""Provider observability — a thin Langfuse tracing wrapper (doc 13 § 13).

Langfuse is an added dev-loop *lens*, never the canonical record: ``ExtractionRun``
(doc 2 § 7) stays authoritative. This module wraps the installed Langfuse **v4**
OTEL SDK — ``Langfuse.start_as_current_observation(...)`` used as a context
manager — *not* the v2 ``@observe`` decorator API that doc 13 § 13's prose
describes.

The wrapper no-ops with zero overhead when disabled: ``trace_generation`` /
``trace_embedding`` yield a sentinel handle and touch no client when
``enabled is False`` or ``client is None`` — so an un-wired provider (every
current call site) behaves exactly as it did before tracing existed. The real
``Langfuse`` client is constructed only by ``build_provider_observability`` and
only when ``langfuse_enabled`` is set (it registers a global OTEL tracer
provider); unit tests inject a typed fake satisfying ``LangfuseLike`` instead.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from rag_recipes.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "EMBEDDING_PREVIEW_CHARS",
    "LangfuseLike",
    "LangfuseObservation",
    "ProviderObservability",
    "SessionScope",
    "TraceContext",
    "build_provider_observability",
]

# Truncation length for the embedding ``text_preview`` recorded in traces. A
# module constant (not a Settings field) keeps config churn out of 5.3; promote
# to a setting only if a consumer needs to tune it (DECISIONS).
EMBEDDING_PREVIEW_CHARS = 200

_ObservationType = Literal["generation", "embedding"]
Level = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]


class TraceContext(BaseModel):
    """Caller-supplied trace fields that don't live on the request/text.

    All optional; 5.3 records whatever is present. ``input_source_span_ids`` is
    Epic 8 (PDF ingestion) and ``input_hash`` is Epic 9 (caching) production —
    5.3 only plumbs the fields through.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str | None = None
    input_source_span_ids: list[str] | None = None
    input_hash: str | None = None


class LangfuseObservation(Protocol):
    """The observation handle yielded by ``start_as_current_observation``."""

    def update(self, **kwargs: Any) -> Any: ...


class LangfuseLike(Protocol):
    """Narrow structural view of the v4 ``Langfuse`` surface we depend on.

    Both the real SDK client and the test stub satisfy this without importing
    concrete SDK types into our signatures.
    """

    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: _ObservationType,
        input: Any = None,
        metadata: Any = None,
        model: str | None = None,
    ) -> AbstractContextManager[LangfuseObservation]: ...


class SessionScope(Protocol):
    """The v4 ``langfuse.propagate_attributes`` *module-level* function.

    Session grouping is **not** an instance method on the ``Langfuse`` client — it
    is a free function that sets trace-level attributes on the active OTEL context.
    We hold it as an injectable callable so the real one is wired by the factory
    and a recorder can be injected in tests (the client and the session scope are
    distinct collaborators).
    """

    def __call__(self, *, session_id: str) -> AbstractContextManager[Any]: ...


class _NoopObservation:
    """Sentinel yielded on the disabled path — ``update`` does nothing."""

    def update(self, **kwargs: Any) -> None:
        return None


_NOOP = _NoopObservation()


class _SafeObservation:
    """Wraps a real observation so a failing ``update`` can never reach the caller.

    Langfuse is an auxiliary lens (doc 13 § 13): a degraded trace backend must
    never turn a successful provider call into a failure, nor mask the provider's
    own exception on the error path. Swallow-and-log instead.
    """

    def __init__(self, observation: LangfuseObservation) -> None:
        self._observation = observation

    def update(self, **kwargs: Any) -> None:
        try:
            self._observation.update(**kwargs)
        except Exception:
            logger.warning("Langfuse observation.update failed; trace dropped", exc_info=True)


def _safe_close(stack: contextlib.ExitStack) -> None:
    """Close the trace context stack without letting its teardown reach the caller."""
    try:
        stack.close()
    except Exception:
        logger.warning("Langfuse trace finalisation failed", exc_info=True)


class ProviderObservability:
    """Wraps an injectable Langfuse-like client behind an ``enabled`` flag.

    When disabled (or no client), the trace context managers return immediately
    yielding ``_NOOP`` — building no payload and touching no client — so the
    traced and un-traced code paths are behaviourally identical.
    """

    def __init__(
        self,
        client: LangfuseLike | None,
        *,
        enabled: bool,
        session_scope: SessionScope | None = None,
    ) -> None:
        self._client = client
        self._enabled = enabled and client is not None
        self._session_scope = session_scope

    def trace_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        metadata: dict[str, Any],
        trace_context: TraceContext | None = None,
    ) -> AbstractContextManager[LangfuseObservation]:
        return self._observe(
            name=name,
            as_type="generation",
            model=model,
            input=input,
            metadata=metadata,
            trace_context=trace_context,
        )

    def trace_embedding(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        metadata: dict[str, Any],
        trace_context: TraceContext | None = None,
    ) -> AbstractContextManager[LangfuseObservation]:
        return self._observe(
            name=name,
            as_type="embedding",
            model=model,
            input=input,
            metadata=metadata,
            trace_context=trace_context,
        )

    @contextlib.contextmanager
    def _observe(
        self,
        *,
        name: str,
        as_type: _ObservationType,
        model: str,
        input: Any,
        metadata: dict[str, Any],
        trace_context: TraceContext | None,
    ) -> Iterator[LangfuseObservation]:
        if not self._enabled or self._client is None:
            yield _NOOP
            return
        merged = self._merge_trace_context(metadata, trace_context)
        session_id = trace_context.session_id if trace_context is not None else None
        stack = contextlib.ExitStack()
        try:
            if session_id is not None and self._session_scope is not None:
                # Session grouping is a v4 *trace attribute* set by the module-level
                # ``propagate_attributes`` function, not observation metadata: enter
                # it before opening the observation so the span inherits the session
                # and every call in one ingestion run groups together (doc 13 § 13).
                stack.enter_context(self._session_scope(session_id=session_id))
            raw_observation = stack.enter_context(
                self._client.start_as_current_observation(
                    name=name, as_type=as_type, input=input, metadata=merged, model=model
                )
            )
        except Exception:
            # A tracing-backend failure must never break the provider call — run it
            # untraced (doc 13 § 13). Tear down whatever opened first.
            _safe_close(stack)
            logger.warning("Langfuse trace start failed; proceeding untraced", exc_info=True)
            yield _NOOP
            return
        observation = _SafeObservation(raw_observation)
        try:
            yield observation
        except Exception as exc:
            # Technical failures must still surface as an ERROR observation before
            # propagating unchanged: the ``_SafeObservation`` swallows any tracing
            # error here so the provider's own exception is the one that re-raises.
            # Record the ``failed`` status dimension too, so failure rates stay
            # queryable alongside the success/rejected paths (AC: status metadata).
            observation.update(
                level="ERROR", status_message=str(exc), metadata={"status": "failed"}
            )
            raise
        finally:
            _safe_close(stack)

    @staticmethod
    def _merge_trace_context(
        metadata: dict[str, Any], trace_context: TraceContext | None
    ) -> dict[str, Any]:
        if trace_context is None:
            return metadata
        merged = dict(metadata)
        if trace_context.input_source_span_ids is not None:
            merged["input_source_span_ids"] = trace_context.input_source_span_ids
        if trace_context.input_hash is not None:
            merged["input_hash"] = trace_context.input_hash
        return merged


def build_provider_observability(settings: Settings) -> ProviderObservability:
    """Construct a ``ProviderObservability`` for the caller (Epic 9/10) to inject.

    Returns a disabled no-op when ``langfuse_enabled`` is ``False`` — the real
    ``Langfuse`` SDK (and its global OTEL tracer provider) is never even imported
    in that case. Lifecycle (``flush``/``shutdown``) belongs to the consumer that
    owns the process, not here (PLAN § Out of Scope).
    """
    if not settings.langfuse_enabled:
        return ProviderObservability(None, enabled=False)
    from langfuse import Langfuse, propagate_attributes

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        tracing_enabled=True,
    )
    # The SDK's overloaded ``start_as_current_observation`` returns a union of
    # ``_AgnosticContextManager[...]`` that mypy can't see as our narrower
    # ``LangfuseLike`` Protocol, though it satisfies it structurally at runtime.
    # ``propagate_attributes`` is a module-level function (not a client method), so
    # the session scope is wired separately from the client.
    return ProviderObservability(
        client,  # type: ignore[arg-type]
        enabled=True,
        session_scope=propagate_attributes,
    )
