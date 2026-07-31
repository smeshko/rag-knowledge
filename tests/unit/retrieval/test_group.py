"""Unit tests for group_by_item — best-chunk score + capped supporting bonus (doc 7 § 9)."""

from __future__ import annotations

import pytest

from rag_recipes.retrieval.group import group_by_item
from rag_recipes.retrieval.types import MergedChunk
from rag_recipes.storage.enums import ChunkType

_BONUS = 0.05
_CAP = 0.15

_TYPES = [
    ChunkType.RECIPE_TITLE,
    ChunkType.RECIPE_SUMMARY,
    ChunkType.RECIPE_FULL,
    ChunkType.RECIPE_INGREDIENTS,
    ChunkType.RECIPE_STEPS,
]


def _chunk(item_id: str, chunk_type: ChunkType, score: float, *, cid: str = "") -> MergedChunk:
    return MergedChunk(
        chunk_id=cid or f"{item_id}-{chunk_type.value}",
        knowledge_item_id=item_id,
        chunk_type=chunk_type,
        score=score,
        sources=["keyword"],
    )


def _group(merged: list[MergedChunk]):
    return group_by_item(merged, supporting_bonus=_BONUS, supporting_bonus_cap=_CAP)


@pytest.mark.parametrize(
    "distinct,expected_bonus",
    [(1, 0.0), (2, 0.05), (3, 0.10), (4, 0.15), (5, 0.15)],
)
def test_supporting_bonus_by_distinct_types_with_cap(
    distinct: int, expected_bonus: float
) -> None:
    chunks = [_chunk("item_1", _TYPES[i], score=0.5) for i in range(distinct)]
    [result] = _group(chunks)
    # best chunk score is 0.5; bonus is 0.05*(distinct-1) capped at 0.15.
    assert result.item_score == pytest.approx(0.5 + expected_bonus)
    assert len(result.matched_chunks) == distinct


def test_item_score_is_best_chunk_plus_bonus() -> None:
    chunks = [
        _chunk("item_1", ChunkType.RECIPE_TITLE, score=0.30),
        _chunk("item_1", ChunkType.RECIPE_STEPS, score=0.80),  # best
    ]
    [result] = _group(chunks)
    # two distinct types → bonus 0.05; best is 0.80.
    assert result.item_score == pytest.approx(0.80 + 0.05)


def test_same_type_chunks_do_not_inflate_distinct_count() -> None:
    chunks = [
        _chunk("item_1", ChunkType.RECIPE_FULL, score=0.4, cid="a"),
        _chunk("item_1", ChunkType.RECIPE_FULL, score=0.6, cid="b"),
    ]
    [result] = _group(chunks)
    # only one distinct type → no bonus; best is 0.6.
    assert result.item_score == pytest.approx(0.6)


def test_results_ordered_by_item_score_desc() -> None:
    merged = [
        _chunk("item_low", ChunkType.RECIPE_FULL, score=0.4),
        _chunk("item_high", ChunkType.RECIPE_TITLE, score=0.9),
        _chunk("item_high", ChunkType.RECIPE_STEPS, score=0.5),  # +bonus
    ]
    results = _group(merged)
    assert [r.knowledge_item_id for r in results] == ["item_high", "item_low"]
    assert results[0].item_score > results[1].item_score
