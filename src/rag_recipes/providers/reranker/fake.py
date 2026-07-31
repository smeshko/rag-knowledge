"""Deterministic in-memory FakeRerankerProvider (Epic 18).

Scaffolding for Phase 18.2's wiring + degradation tests, mirroring
``FakeLLMProvider``/``FakeEmbeddingProvider``:

* with no ``scores_by_chunk_id`` it preserves input order (identity), writing each
  candidate's own ``score`` as the ``relevance_score`` — so 18.2 can assert that a
  no-op rerank leaves the final results unchanged;
* with a ``scores_by_chunk_id`` map it reorders by that score (desc), ties broken by
  ``chunk_id`` for determinism;
* the ``emit`` knob forces a single documented invariant violation
  (``"duplicate"`` / ``"unknown"`` / ``"missing"`` / ``"out_of_range_rank"``) so 18.2 can
  exercise its caller-side degradation policy — the malformed modes assume a
  **non-empty** candidate list (with no candidates there is nothing to corrupt);
* ``.calls`` records each invocation (deep-copied) for assertion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rag_recipes.providers._observability import TraceContext
from rag_recipes.providers.reranker.base import RerankerProvider
from rag_recipes.providers.reranker.types import RerankCandidate, RerankResult

__all__ = ["FakeRerankerProvider", "RerankCall"]

_EMIT_MODES = frozenset({"duplicate", "unknown", "missing", "out_of_range_rank"})


@dataclass(frozen=True)
class RerankCall:
    """A recorded ``rerank`` invocation (candidates deep-copied)."""

    query: str
    candidates: list[RerankCandidate]
    top_n: int


class FakeRerankerProvider(RerankerProvider):
    """Reranker provider returning deterministic, configurable results."""

    provider = "fake"

    def __init__(
        self,
        *,
        scores_by_chunk_id: Mapping[str, float] | None = None,
        emit: str | None = None,
    ) -> None:
        if emit is not None and emit not in _EMIT_MODES:
            raise ValueError(
                f"unknown emit mode {emit!r}; expected one of {sorted(_EMIT_MODES)}"
            )
        self._scores_by_chunk_id = dict(scores_by_chunk_id) if scores_by_chunk_id else None
        self._emit = emit
        self._calls: list[RerankCall] = []

    @property
    def calls(self) -> tuple[RerankCall, ...]:
        return tuple(self._calls)

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_n: int,
        trace_context: TraceContext | None = None,
    ) -> list[RerankResult]:
        # The Fake emits no traces; the param exists only to keep the override
        # signature-compatible with the RerankerProvider contract.
        self._calls.append(
            RerankCall(
                query=query,
                candidates=[c.model_copy(deep=True) for c in candidates],
                top_n=top_n,
            )
        )
        if self._emit is not None:
            return self._emit_malformed(candidates, top_n)
        return self._rank(candidates, top_n)

    def _relevance(self, candidate: RerankCandidate) -> float:
        scores = self._scores_by_chunk_id
        if scores is None:
            return candidate.score
        return scores.get(candidate.chunk_id, candidate.score)

    def _ordered(self, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        if self._scores_by_chunk_id is None:
            return list(candidates)  # identity: input order preserved
        # Reorder by supplied score (desc); ties broken by chunk_id (deterministic).
        return sorted(candidates, key=lambda c: (-self._relevance(c), c.chunk_id))

    def _rank(self, candidates: list[RerankCandidate], top_n: int) -> list[RerankResult]:
        ordered = self._ordered(candidates)[:top_n]
        return [
            RerankResult(
                chunk_id=c.chunk_id,
                relevance_score=self._relevance(c),
                rank=index + 1,
            )
            for index, c in enumerate(ordered)
        ]

    def _emit_malformed(
        self, candidates: list[RerankCandidate], top_n: int
    ) -> list[RerankResult]:
        base = self._rank(candidates, top_n)
        if self._emit == "duplicate" and base:
            # Re-emit the first result's chunk_id → a duplicate chunk_id.
            return [*base, base[0].model_copy(update={"rank": len(base) + 1})]
        if self._emit == "unknown":
            # A chunk_id not present among the input candidates.
            return [
                *base,
                RerankResult(chunk_id="__unknown__", relevance_score=0.0, rank=len(base) + 1),
            ]
        if self._emit == "missing":
            # Omit all but the first candidate (the rest are "missing" from the result).
            return base[:1]
        if self._emit == "out_of_range_rank" and base:
            # Unambiguously out of the 1-based [1, N] range: a sub-range 0 on the
            # first result and an over-range rank (> N) on the last, so a caller's
            # clamp/ignore policy is exercised at both bounds. Guarded on a non-empty
            # base so an empty candidate list is a safe no-op (not an IndexError).
            out = [r.model_copy(deep=True) for r in base]
            out[0] = out[0].model_copy(update={"rank": 0})
            out[-1] = out[-1].model_copy(update={"rank": len(base) + 5})
            return out
        return base
