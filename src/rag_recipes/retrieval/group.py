"""Group merged chunks under their parent KnowledgeItem (doc 7 § 9).

An item's score is its single best matched chunk plus a supporting-chunk bonus that
rewards matching across multiple *distinct* chunk types (a recipe whose title AND
steps both match is stronger than one that matches a single way). The bonus is
``supporting_bonus * (distinct types - 1)`` capped at ``supporting_bonus_cap`` — so
one matched type adds nothing, and many matched types saturate at the cap. Pure: the
bonus constants are injected for unit testing.
"""

from __future__ import annotations

from rag_recipes.retrieval.types import ItemResult, MergedChunk


def group_by_item(
    merged: list[MergedChunk],
    *,
    supporting_bonus: float,
    supporting_bonus_cap: float,
) -> list[ItemResult]:
    """Group ``merged`` by ``knowledge_item_id`` and score each item.

    ``item_score = max(chunk score) + min(supporting_bonus * (distinct types - 1),
    supporting_bonus_cap)``. Results are ordered by ``item_score`` desc, then best
    chunk score desc, then ``knowledge_item_id`` for a fully deterministic order.
    """
    by_item: dict[str, list[MergedChunk]] = {}
    for chunk in merged:
        by_item.setdefault(chunk.knowledge_item_id, []).append(chunk)

    results: list[ItemResult] = []
    for item_id, group in by_item.items():
        best = max(chunk.score for chunk in group)
        distinct_types = len({chunk.chunk_type for chunk in group})
        bonus = min(supporting_bonus * (distinct_types - 1), supporting_bonus_cap)
        results.append(
            ItemResult(
                knowledge_item_id=item_id,
                item_score=best + bonus,
                matched_chunks=group,
            )
        )

    results.sort(
        key=lambda r: (
            -r.item_score,
            -max(chunk.score for chunk in r.matched_chunks),
            r.knowledge_item_id,
        )
    )
    return results
