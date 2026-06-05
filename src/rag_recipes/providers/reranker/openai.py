"""OpenAIRerankerProvider — an LLM-based listwise reranker (Epic 18.2).

Composes an ``OpenAILLMProvider`` and asks it (via strict ``rerank.v1`` structured
output) to order the candidate ``chunk_id``s by relevance to the query. The result
order is the ranking; the facade derives rank from list position. Any technical
failure — transport, **timeout**, rate-limit exhaustion, or a parse error — raises
``RerankerTechnicalError`` so the facade degrades to the un-reranked RRF order
(reranking can never break search).

The composed ``OpenAILLMProvider`` is built with the **short**
``rerank_request_timeout_seconds`` (not the 60s ingestion default) and no rate-limit
retries, so a slow rerank times out to a fallback rather than stalling the
synchronous search path.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rag_recipes.providers._observability import TraceContext
from rag_recipes.providers.errors import LLMTechnicalError, RerankerTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.openai import OpenAILLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from rag_recipes.providers.reranker.base import RerankerProvider
from rag_recipes.providers.reranker.schema import RERANK_SCHEMA_VERSION, RERANK_V1_SCHEMA
from rag_recipes.providers.reranker.types import RerankCandidate, RerankResult

__all__ = ["OpenAIRerankerProvider"]

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "rerank-v1"

_PROMPT = """\
You are a search reranker. Given a user query and a list of candidate passages \
(each with a chunk_id), order the candidates from most to least relevant to the \
query. Return every candidate's chunk_id exactly once in the `ranking` array, most \
relevant first, each with a relevance_score in [0, 1]. Use only the provided \
candidates; do not invent chunk_ids."""


class OpenAIRerankerProvider(RerankerProvider):
    """Listwise reranker backed by ``OpenAILLMProvider`` strict structured output."""

    provider = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        llm_provider: LLMProvider | None = None,
        request_timeout: float = 8.0,
        max_rate_limit_retries: int = 0,
    ) -> None:
        # Compose an LLM provider so retry/parse-error handling is inherited; the
        # caller resolves model/timeout from Settings (the provider never reads them).
        # Tests inject a FakeLLMProvider; production builds an OpenAILLMProvider.
        self._llm: LLMProvider = llm_provider or OpenAILLMProvider(
            api_key,
            default_model=model,
            request_timeout=request_timeout,
            max_rate_limit_retries=max_rate_limit_retries,
        )
        self._model = model

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_n: int,
        trace_context: TraceContext | None = None,
    ) -> list[RerankResult]:
        request = StructuredOutputRequest(
            provider=self.provider,
            model=self._model,
            prompt_version=_PROMPT_VERSION,
            schema_version=RERANK_SCHEMA_VERSION,
            input=_render_rerank_input(query, candidates),
            json_schema=_rerank_schema(),
        )
        start = time.monotonic()
        try:
            response = await self._llm.generate_structured_output(
                request, trace_context=trace_context
            )
        except LLMTechnicalError as exc:
            raise RerankerTechnicalError(str(exc)) from exc

        if response.parse_error is not None or response.output_json is None:
            raise RerankerTechnicalError(
                f"reranker output could not be parsed: {response.parse_error}"
            )

        results = _parse_ranking(response.output_json)
        latency_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "rerank ok provider=%s model=%s top_n=%d candidates=%d ranked=%d latency_ms=%.1f",
            self.provider,
            self._model,
            top_n,
            len(candidates),
            len(results),
            latency_ms,
        )
        return results


def _render_rerank_input(query: str, candidates: list[RerankCandidate]) -> str:
    lines = [f"Query: {query}", "", "Candidates:"]
    for candidate in candidates:
        lines.append(f"- chunk_id: {candidate.chunk_id}")
        lines.append(f"  text: {candidate.text}")
    return f"{_PROMPT}\n\n" + "\n".join(lines)


def _parse_ranking(output_json: dict[str, Any]) -> list[RerankResult]:
    """Build ``RerankResult``s from a clean-parsed (but content-untrusted) object.

    Rank is the 1-based list position; malformed entries (non-object, non-string
    chunk_id, non-numeric score) are skipped — the facade further validates ids
    against the input candidates and never drops an input chunk.
    """
    ranking = output_json.get("ranking")
    if not isinstance(ranking, list):
        return []
    results: list[RerankResult] = []
    for entry in ranking:
        if not isinstance(entry, dict):
            continue
        chunk_id = entry.get("chunk_id")
        score = entry.get("relevance_score")
        if not isinstance(chunk_id, str) or not isinstance(score, (int, float)):
            continue
        results.append(
            RerankResult(
                chunk_id=chunk_id,
                relevance_score=float(score),
                rank=len(results) + 1,
            )
        )
    return results


def _rerank_schema() -> dict[str, Any]:
    import copy

    return copy.deepcopy(RERANK_V1_SCHEMA)
