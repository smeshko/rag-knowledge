"""DTOs for the reranker provider (Epic 18).

A passage reranker scores chunk *candidates* against the query, so ``RerankCandidate``
carries the ``MergedChunk`` fields needed to rebuild order plus the fetched ``text``.

``RerankResult.relevance_score`` is **not cosmetic**: Phase 18.2 writes it onto the
corresponding ``MergedChunk.score`` to drive the final ranking. (A pure list reorder
would have no effect — ``group_by_item`` sorts items by ``max(MergedChunk.score)`` and
ignores list order — so the reranker must influence the *score*. Score replacement is
necessary but not sufficient: ``group_by_item`` also adds a supporting-chunk bonus
before sorting, so when reranking is applied 18.2 additionally makes the reranker
authoritative over item order, with the supporting bonus a tiebreaker only — see the
plan's Key decisions / 18.2.)

Result invariants a well-behaved reranker honors (the caller's degradation policy for
violations is specified and tested in 18.2):

* every ``chunk_id`` is one of the input candidates' ids (no **unknown** ids),
* ``chunk_id``s are **unique** (no duplicates),
* ``rank`` is 1-based, and
* ``relevance_score`` is finite; ties are broken deterministically by ``chunk_id``.
"""

from __future__ import annotations

from pydantic import BaseModel

from rag_recipes.storage.enums import ChunkType

__all__ = ["RerankCandidate", "RerankResult"]


class RerankCandidate(BaseModel):
    """One chunk candidate handed to the reranker (``MergedChunk`` fields + ``text``)."""

    chunk_id: str
    text: str
    knowledge_item_id: str
    chunk_type: ChunkType
    score: float
    sources: list[str]


class RerankResult(BaseModel):
    """The reranker's verdict for one chunk.

    ``relevance_score`` is the value Phase 18.2 writes onto ``MergedChunk.score``;
    ``rank`` is 1-based (matching ``ChunkCandidate.rank`` in ``retrieval/types.py``).
    """

    chunk_id: str
    relevance_score: float
    rank: int
