"""RerankerProvider interface (Epic 18), mirroring ``LLMProvider``/``EmbeddingProvider``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from rag_recipes.providers._observability import TraceContext
from rag_recipes.providers.reranker.types import RerankCandidate, RerankResult

__all__ = ["RerankerProvider"]


class RerankerProvider(ABC):
    """Abstract provider that reranks chunk candidates against a query.

    Technical failures (transport, timeout, provider error) raise
    ``RerankerTechnicalError`` (``providers.errors``); the caller decides how to
    degrade (Phase 18.2 falls back to the un-reranked RRF order). ``trace_context``
    carries optional observability fields; whether a trace is emitted is the
    implementation's concern (the Fake ignores it). Implementations expose
    ``provider`` (a stable identity label) like the other provider ABCs.
    """

    #: Stable provider identity label, e.g. ``"openai"`` / ``"fake"``.
    provider: ClassVar[str]

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_n: int,
        trace_context: TraceContext | None = None,
    ) -> list[RerankResult]:
        """Rerank ``candidates`` against ``query`` and return up to ``top_n`` results.

        Each ``RerankResult.relevance_score`` is the score the caller writes onto the
        corresponding ``MergedChunk`` (Phase 18.2). See ``types.py`` for the result
        invariants and the caller's degradation policy.
        """
