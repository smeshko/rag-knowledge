"""Weighted reciprocal-rank-fusion of the keyword + vector legs (doc 7 § 8).

Each candidate contributes ``source_weight * chunk_type_boost * 1/(rrf_k + rank)``.
RRF is deliberately **rank-based**, not score-based: ``ts_rank_cd`` (keyword) and
cosine similarity (vector) live on incomparable scales, so fusing by rank is the
whole point. Contributions for the same ``chunk_id`` across both legs are SUMMED
(a chunk both legs surface is stronger evidence) and the legs are recorded in
``sources``. Pure — the boost tables, weights, and ``rrf_k`` are injected so this is
unit-testable without ``Settings`` or a DB.
"""

from __future__ import annotations

from collections.abc import Mapping

from rag_recipes.retrieval.types import ChunkCandidate, MergedChunk
from rag_recipes.storage.enums import ChunkType


def _candidate_score(rank: int, boost: float, weight: float, rrf_k: int) -> float:
    """One leg's contribution for a candidate: ``weight * boost * 1/(rrf_k + rank)``."""
    return weight * boost * (1.0 / (rrf_k + rank))


def _accumulate(
    by_chunk: dict[str, MergedChunk], cand: ChunkCandidate, contribution: float
) -> None:
    existing = by_chunk.get(cand.chunk_id)
    if existing is None:
        by_chunk[cand.chunk_id] = MergedChunk(
            chunk_id=cand.chunk_id,
            knowledge_item_id=cand.knowledge_item_id,
            chunk_type=cand.chunk_type,
            score=contribution,
            sources=[cand.retrieval_source],
        )
    else:
        existing.score += contribution
        if cand.retrieval_source not in existing.sources:
            existing.sources.append(cand.retrieval_source)


def merge_candidates(
    keyword: list[ChunkCandidate],
    vector: list[ChunkCandidate],
    *,
    keyword_boosts: Mapping[ChunkType, float],
    vector_boosts: Mapping[ChunkType, float],
    rrf_k: int,
    keyword_source_weight: float,
    vector_source_weight: float,
) -> list[MergedChunk]:
    """Fuse the two legs into ``MergedChunk``s sorted by score desc (id tiebreak).

    A missing boost key defaults to ``1.0`` (defensive — a future chunk type still
    fuses, just unboosted). The keyword leg is accumulated first so a both-legs
    chunk's ``sources`` reads ``["keyword", "vector"]``.
    """
    by_chunk: dict[str, MergedChunk] = {}
    for cand in keyword:
        boost = keyword_boosts.get(cand.chunk_type, 1.0)
        _accumulate(
            by_chunk,
            cand,
            _candidate_score(cand.rank, boost, keyword_source_weight, rrf_k),
        )
    for cand in vector:
        boost = vector_boosts.get(cand.chunk_type, 1.0)
        _accumulate(
            by_chunk,
            cand,
            _candidate_score(cand.rank, boost, vector_source_weight, rrf_k),
        )
    return sorted(by_chunk.values(), key=lambda m: (-m.score, m.chunk_id))
