"""Keyword retrieval leg — Postgres full-text search over chunk text (doc 7 § 4).

Runs ``plainto_tsquery('english', :q)`` against the DB-managed ``chunks.ts_vector``
generated column (Epic 10.3; an unmapped ``GENERATED ALWAYS AS to_tsvector(...)``
column, so it is referenced by name), ranks with ``ts_rank_cd``, and returns ranked
``ChunkCandidate``s. The query text is always a bound parameter — never string
interpolated — so a query carrying SQL metacharacters is treated as search text
(DECISIONS #3).
"""

from __future__ import annotations

from sqlalchemy import ColumnClause, column, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.retrieval._sql import apply_search_predicates
from rag_recipes.retrieval.types import ChunkCandidate, FilterSet, NormalizedQuery
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem

# The FTS column is a DB-managed generated column not mapped on the ORM model
# (Epic 10.3 migration); reference it by name.
_TS_VECTOR: ColumnClause[object] = column("ts_vector")


async def keyword_search(
    session: AsyncSession,
    query: NormalizedQuery,
    filters: FilterSet,
    *,
    top_k: int,
) -> list[ChunkCandidate]:
    """Return up to ``top_k`` keyword-matched chunks ranked by ``ts_rank_cd`` desc.

    ``rank`` is the 1-based ordinal position after ``ORDER BY score DESC, Chunk.id``
    (so equal-score rows still get distinct, deterministic ranks); ``raw_score`` is
    the ``ts_rank_cd`` value; ``distance``/``similarity`` are left ``None`` (the
    vector leg's concern).
    """
    tsquery = func.plainto_tsquery("english", query.keyword)
    score = func.ts_rank_cd(_TS_VECTOR, tsquery).label("score")
    stmt = (
        select(
            Chunk.id,
            Chunk.parent_id,
            Chunk.chunk_type,
            score,
        )
        .join(KnowledgeItem, Chunk.parent_id == KnowledgeItem.id)
        .join(Document, KnowledgeItem.document_id == Document.id)
        .where(_TS_VECTOR.op("@@")(tsquery))
    )
    stmt = apply_search_predicates(stmt, filters)
    stmt = stmt.order_by(score.desc(), Chunk.id).limit(top_k)

    rows = (await session.execute(stmt)).all()
    return [
        ChunkCandidate(
            chunk_id=row.id,
            knowledge_item_id=row.parent_id,
            chunk_type=row.chunk_type,
            retrieval_source="keyword",
            rank=rank,
            raw_score=float(row.score),
        )
        for rank, row in enumerate(rows, start=1)
    ]
