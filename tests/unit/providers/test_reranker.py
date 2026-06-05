"""Unit tests for the reranker provider boundary (Epic 18.1).

Covers the real product types (``RerankCandidate``/``RerankResult``) and
``RerankerTechnicalError``, plus the ``FakeRerankerProvider`` scaffolding that 18.2's
wiring + degradation tests depend on (deterministic identity / score ordering, the
``emit`` malformed-output modes, and ``.calls`` capture).
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.providers.errors import ProviderError, RerankerTechnicalError
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputResponse, TokenUsage
from rag_recipes.providers.reranker.base import RerankerProvider
from rag_recipes.providers.reranker.fake import FakeRerankerProvider
from rag_recipes.providers.reranker.openai import OpenAIRerankerProvider
from rag_recipes.providers.reranker.types import RerankCandidate, RerankResult
from rag_recipes.storage.enums import ChunkType


def _candidate(chunk_id: str, *, score: float, item: str = "item") -> RerankCandidate:
    return RerankCandidate(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        knowledge_item_id=item,
        chunk_type=ChunkType.RECIPE_SUMMARY,
        score=score,
        sources=["keyword"],
    )


def _candidates() -> list[RerankCandidate]:
    # Deliberately NOT in score order, so identity (input order) and score ordering
    # are distinguishable.
    return [
        _candidate("c1", score=0.1),
        _candidate("c2", score=0.9),
        _candidate("c3", score=0.5),
    ]


# --- product types + error ------------------------------------------------------


def test_rerank_candidate_and_result_construct() -> None:
    cand = _candidate("c1", score=0.4)
    assert cand.chunk_id == "c1"
    assert cand.chunk_type is ChunkType.RECIPE_SUMMARY
    result = RerankResult(chunk_id="c1", relevance_score=0.9, rank=1)
    assert result.rank == 1
    assert result.relevance_score == 0.9


def test_reranker_technical_error_is_provider_error() -> None:
    assert issubclass(RerankerTechnicalError, ProviderError)
    assert isinstance(RerankerTechnicalError("boom"), ProviderError)


def test_fake_is_a_reranker_provider() -> None:
    assert isinstance(FakeRerankerProvider(), RerankerProvider)
    assert FakeRerankerProvider.provider == "fake"


# --- fake ordering --------------------------------------------------------------


async def test_identity_preserves_input_order_with_sequential_ranks() -> None:
    fake = FakeRerankerProvider()
    results = await fake.rerank("q", _candidates(), top_n=10)
    assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]  # input order
    assert [r.rank for r in results] == [1, 2, 3]
    # identity writes each candidate's own score as the relevance score (no-op).
    assert [r.relevance_score for r in results] == [0.1, 0.9, 0.5]


async def test_scores_by_chunk_id_reorders_descending_deterministically() -> None:
    fake = FakeRerankerProvider(scores_by_chunk_id={"c1": 0.95, "c2": 0.10, "c3": 0.50})
    results = await fake.rerank("q", _candidates(), top_n=10)
    assert [r.chunk_id for r in results] == ["c1", "c3", "c2"]  # by supplied score desc
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].relevance_score == 0.95


async def test_top_n_truncates_after_ordering() -> None:
    fake = FakeRerankerProvider(scores_by_chunk_id={"c1": 0.95, "c2": 0.10, "c3": 0.50})
    results = await fake.rerank("q", _candidates(), top_n=2)
    assert [r.chunk_id for r in results] == ["c1", "c3"]
    assert [r.rank for r in results] == [1, 2]


async def test_score_ties_broken_by_chunk_id() -> None:
    fake = FakeRerankerProvider(scores_by_chunk_id={"c1": 0.5, "c2": 0.5, "c3": 0.5})
    results = await fake.rerank("q", _candidates(), top_n=10)
    assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]  # tie → chunk_id asc


async def test_calls_records_each_invocation() -> None:
    fake = FakeRerankerProvider()
    cands = _candidates()
    await fake.rerank("first", cands, top_n=5)
    await fake.rerank("second", cands, top_n=3)
    assert len(fake.calls) == 2
    assert fake.calls[0].query == "first"
    assert fake.calls[0].top_n == 5
    assert [c.chunk_id for c in fake.calls[0].candidates] == ["c1", "c2", "c3"]
    assert fake.calls[1].query == "second"
    # The call log is a deep copy — mutating the original candidates can't change it.
    cands.clear()
    assert len(fake.calls[0].candidates) == 3


# --- fake emit (malformed) modes for 18.2's degradation tests -------------------


async def test_emit_duplicate_repeats_a_chunk_id() -> None:
    fake = FakeRerankerProvider(emit="duplicate")
    results = await fake.rerank("q", _candidates(), top_n=10)
    ids = [r.chunk_id for r in results]
    assert len(ids) != len(set(ids))  # a duplicate chunk_id is present


async def test_emit_unknown_includes_an_unknown_chunk_id() -> None:
    fake = FakeRerankerProvider(emit="unknown")
    results = await fake.rerank("q", _candidates(), top_n=10)
    input_ids = {"c1", "c2", "c3"}
    assert any(r.chunk_id not in input_ids for r in results)


async def test_emit_missing_omits_candidates() -> None:
    fake = FakeRerankerProvider(emit="missing")
    results = await fake.rerank("q", _candidates(), top_n=10)
    assert len(results) < 3  # not every candidate is returned


async def test_emit_out_of_range_rank_breaks_one_based_ranks() -> None:
    fake = FakeRerankerProvider(emit="out_of_range_rank")
    results = await fake.rerank("q", _candidates(), top_n=10)
    # Out of the valid 1-based [1, N] range at both bounds: a sub-range 0 and an
    # over-range rank (> N), so 18.2's clamp/ignore policy is exercised at each end.
    assert any(r.rank < 1 for r in results)
    assert any(r.rank > len(results) for r in results)


def test_emit_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown emit mode"):
        FakeRerankerProvider(emit="nonsense")


@pytest.mark.parametrize("mode", ["duplicate", "unknown", "missing", "out_of_range_rank"])
async def test_emit_modes_are_safe_on_empty_candidates(mode: str) -> None:
    # The malformed modes must not crash on an empty candidate list (no IndexError).
    fake = FakeRerankerProvider(emit=mode)
    results = await fake.rerank("q", [], top_n=10)
    assert isinstance(results, list)
    # Nothing to corrupt → at most the synthetic "unknown" entry, never an input id.
    assert all(r.chunk_id == "__unknown__" for r in results)


# --- OpenAIRerankerProvider (real provider; FakeLLMProvider injected) ------------


async def test_openai_reranker_orders_by_model_ranking() -> None:
    canned = {
        "ranking": [
            {"chunk_id": "c3", "relevance_score": 0.9},
            {"chunk_id": "c1", "relevance_score": 0.5},
            {"chunk_id": "c2", "relevance_score": 0.2},
        ]
    }
    reranker = OpenAIRerankerProvider(
        "sk-test", model="gpt-4.1", llm_provider=FakeLLMProvider(default_output=canned)
    )
    results = await reranker.rerank("q", _candidates(), top_n=10)
    assert [r.chunk_id for r in results] == ["c3", "c1", "c2"]
    assert [r.rank for r in results] == [1, 2, 3]  # rank = list position
    assert results[0].relevance_score == 0.9


async def test_openai_reranker_parse_error_raises_technical() -> None:
    rejected = StructuredOutputResponse(
        output_json=None,
        parse_error="model output is not valid JSON",
        raw_text="{bad",
        usage=TokenUsage(input_tokens=1, output_tokens=0),
        provider="openai",
        model="gpt-4.1",
    )
    reranker = OpenAIRerankerProvider(
        "sk-test", model="gpt-4.1", llm_provider=FakeLLMProvider(default_output=rejected)
    )
    with pytest.raises(RerankerTechnicalError):
        await reranker.rerank("q", _candidates(), top_n=10)


async def test_openai_reranker_technical_failure_raises_technical() -> None:
    reranker = OpenAIRerankerProvider(
        "sk-test", model="gpt-4.1", llm_provider=FakeLLMProvider(fail_technically=True)
    )
    with pytest.raises(RerankerTechnicalError):
        await reranker.rerank("q", _candidates(), top_n=10)


async def test_openai_reranker_skips_malformed_entries() -> None:
    # A clean-parsed object whose entries are partly malformed: only well-formed
    # entries become results (the facade further validates ids).
    canned = {
        "ranking": [
            {"chunk_id": "c2", "relevance_score": 0.9},
            {"chunk_id": 123, "relevance_score": 0.5},  # non-string id → skipped
            "not-an-object",  # → skipped
            {"chunk_id": "c1"},  # missing score → skipped
        ]
    }
    reranker = OpenAIRerankerProvider(
        "sk-test", model="gpt-4.1", llm_provider=FakeLLMProvider(default_output=canned)
    )
    results = await reranker.rerank("q", _candidates(), top_n=10)
    assert [r.chunk_id for r in results] == ["c2"]


async def test_openai_reranker_logs_success(caplog: pytest.LogCaptureFixture) -> None:
    canned = {"ranking": [{"chunk_id": "c1", "relevance_score": 0.9}]}
    reranker = OpenAIRerankerProvider(
        "sk-test", model="gpt-4.1", llm_provider=FakeLLMProvider(default_output=canned)
    )
    with caplog.at_level("INFO", logger="rag_recipes.providers.reranker.openai"):
        await reranker.rerank("q", _candidates(), top_n=7)
    assert any(
        "rerank ok" in r.message and "model=gpt-4.1" in r.message and "top_n=7" in r.message
        for r in caplog.records
    )


# --- get_reranker_provider dependency -------------------------------------------


def _settings(*, enabled: bool, provider: str = "openai") -> Any:
    class _S:
        openai_api_key = "sk-test"
        rerank_model = "gpt-4.1"
        rerank_request_timeout_seconds = 8.0

        def __init__(self) -> None:
            self.reranking_enabled = enabled
            self.rerank_provider = provider

    return _S()


def test_get_reranker_provider_returns_none_when_disabled() -> None:
    from rag_recipes.api.dependencies import get_reranker_provider

    assert get_reranker_provider(settings=_settings(enabled=False)) is None


def test_get_reranker_provider_builds_openai_when_enabled() -> None:
    from rag_recipes.api.dependencies import get_reranker_provider

    provider = get_reranker_provider(settings=_settings(enabled=True, provider="openai"))
    assert isinstance(provider, OpenAIRerankerProvider)
    assert provider.provider == "openai"


def test_get_reranker_provider_rejects_unsupported_provider() -> None:
    # Defensive backstop (a Settings validator also rejects this at load): an
    # unsupported provider raises and constructs no OpenAI client.
    from rag_recipes.api.dependencies import get_reranker_provider

    with pytest.raises(ValueError, match="unsupported rerank_provider"):
        get_reranker_provider(settings=_settings(enabled=True, provider="cohere"))
