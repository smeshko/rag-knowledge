"""Unit tests for merge_candidates — weighted RRF with chunk-type boosts (doc 7 § 8)."""

from __future__ import annotations

import pytest

from rag_recipes.retrieval.merge import merge_candidates
from rag_recipes.retrieval.types import ChunkCandidate
from rag_recipes.storage.enums import ChunkType

_RRF_K = 60


def _cand(
    chunk_id: str,
    *,
    item_id: str = "item_1",
    chunk_type: ChunkType = ChunkType.RECIPE_FULL,
    source: str = "keyword",
    rank: int = 1,
) -> ChunkCandidate:
    return ChunkCandidate(
        chunk_id=chunk_id,
        knowledge_item_id=item_id,
        chunk_type=chunk_type,
        retrieval_source=source,
        rank=rank,
        raw_score=0.0,
    )


def _merge(keyword: list[ChunkCandidate], vector: list[ChunkCandidate], **kw: object):
    return merge_candidates(
        keyword,
        vector,
        keyword_boosts=kw.get("keyword_boosts", {ChunkType.RECIPE_TITLE: 1.40}),  # type: ignore[arg-type]
        vector_boosts=kw.get("vector_boosts", {ChunkType.RECIPE_SUMMARY: 1.20}),  # type: ignore[arg-type]
        rrf_k=_RRF_K,
        keyword_source_weight=kw.get("keyword_source_weight", 1.0),  # type: ignore[arg-type]
        vector_source_weight=kw.get("vector_source_weight", 1.0),  # type: ignore[arg-type]
    )


def test_single_leg_score_is_weight_times_boost_times_rrf() -> None:
    cand = _cand("c1", chunk_type=ChunkType.RECIPE_TITLE, source="keyword", rank=1)
    [merged] = _merge([cand], [], keyword_boosts={ChunkType.RECIPE_TITLE: 1.40})
    assert merged.score == pytest.approx(1.0 * 1.40 * (1 / (_RRF_K + 1)))
    assert merged.sources == ["keyword"]
    assert merged.knowledge_item_id == "item_1"
    assert merged.chunk_type == ChunkType.RECIPE_TITLE


def test_chunk_in_both_legs_sums_contributions() -> None:
    kw = _cand("c1", chunk_type=ChunkType.RECIPE_FULL, source="keyword", rank=2)
    vec = _cand("c1", chunk_type=ChunkType.RECIPE_FULL, source="vector", rank=5)
    [merged] = _merge(
        [kw],
        [vec],
        keyword_boosts={ChunkType.RECIPE_FULL: 0.95},
        vector_boosts={ChunkType.RECIPE_FULL: 1.10},
    )
    kw_contrib = 1.0 * 0.95 * (1 / (_RRF_K + 2))
    vec_contrib = 1.0 * 1.10 * (1 / (_RRF_K + 5))
    assert merged.score == pytest.approx(kw_contrib + vec_contrib)
    assert merged.sources == ["keyword", "vector"]


def test_per_side_boost_tables_are_applied_per_side() -> None:
    # Keyword side: title (1.40) out-scores full (0.95) at the same rank.
    title = _cand("c_title", chunk_type=ChunkType.RECIPE_TITLE, source="keyword", rank=1)
    full = _cand("c_full", chunk_type=ChunkType.RECIPE_FULL, source="keyword", rank=1)
    kw_results = _merge(
        [title, full],
        [],
        keyword_boosts={ChunkType.RECIPE_TITLE: 1.40, ChunkType.RECIPE_FULL: 0.95},
    )
    assert kw_results[0].chunk_id == "c_title"

    # Vector side: summary (1.20) out-scores title (0.90) at the same rank.
    v_summary = _cand("c_sum", chunk_type=ChunkType.RECIPE_SUMMARY, source="vector", rank=1)
    v_title = _cand("c_vt", chunk_type=ChunkType.RECIPE_TITLE, source="vector", rank=1)
    vec_results = _merge(
        [],
        [v_summary, v_title],
        vector_boosts={ChunkType.RECIPE_SUMMARY: 1.20, ChunkType.RECIPE_TITLE: 0.90},
    )
    assert vec_results[0].chunk_id == "c_sum"


def test_missing_boost_key_defaults_to_one() -> None:
    cand = _cand("c1", chunk_type=ChunkType.RECIPE_STEPS, source="keyword", rank=1)
    [merged] = _merge([cand], [], keyword_boosts={})  # no entry for RECIPE_STEPS
    assert merged.score == pytest.approx(1.0 * 1.0 * (1 / (_RRF_K + 1)))


def test_results_sorted_by_score_desc_with_id_tiebreak() -> None:
    high = _cand("c_high", source="keyword", rank=1, chunk_type=ChunkType.RECIPE_TITLE)
    low = _cand("c_low", source="keyword", rank=50, chunk_type=ChunkType.RECIPE_TITLE)
    results = _merge([low, high], [], keyword_boosts={ChunkType.RECIPE_TITLE: 1.40})
    assert [m.chunk_id for m in results] == ["c_high", "c_low"]
    assert results[0].score > results[1].score


def test_source_weight_scales_each_leg() -> None:
    cand = _cand("c1", chunk_type=ChunkType.RECIPE_FULL, source="vector", rank=1)
    [merged] = _merge(
        [],
        [cand],
        vector_boosts={ChunkType.RECIPE_FULL: 1.10},
        vector_source_weight=2.0,
    )
    assert merged.score == pytest.approx(2.0 * 1.10 * (1 / (_RRF_K + 1)))
