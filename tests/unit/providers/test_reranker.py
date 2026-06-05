"""Unit tests for the reranker provider boundary (Epic 18.1).

Covers the real product types (``RerankCandidate``/``RerankResult``) and
``RerankerTechnicalError``, plus the ``FakeRerankerProvider`` scaffolding that 18.2's
wiring + degradation tests depend on (deterministic identity / score ordering, the
``emit`` malformed-output modes, and ``.calls`` capture).
"""

from __future__ import annotations

import pytest

from rag_recipes.providers.errors import ProviderError, RerankerTechnicalError
from rag_recipes.providers.reranker.base import RerankerProvider
from rag_recipes.providers.reranker.fake import FakeRerankerProvider
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
    assert any(r.rank < 1 for r in results)  # ranks fall outside the 1-based range


def test_emit_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown emit mode"):
        FakeRerankerProvider(emit="nonsense")
