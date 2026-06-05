"""Vector retrieval leg — pgvector cosine search over chunk embeddings (doc 7 § 5).

Embeds the normalized query with the injected ``EmbeddingProvider`` and searches
``chunk_embeddings`` by cosine distance (``<=>``), joined through ``chunks →
knowledge_items → documents`` under the same metadata predicates as the keyword leg
(via the shared ``_sql`` seam) PLUS the ``(embedding_provider, embedding_model)``
equality required by the embedding-model rule — comparing vectors across different
embedding spaces would silently corrupt results (doc 7). Candidates carry ``rank``,
``distance``, and ``similarity = 1 - distance``; ``raw_score`` mirrors the keyword
leg's "higher = better" (``= similarity``) so the 12.3 facade treats both uniformly.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.retrieval._sql import apply_search_predicates
from rag_recipes.retrieval.types import ChunkCandidate, FilterSet, NormalizedQuery
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem


async def vector_search(
    session: AsyncSession,
    query: NormalizedQuery,
    filters: FilterSet,
    *,
    provider: EmbeddingProvider,
    embedding_provider: str,
    embedding_model: str,
    top_k: int,
) -> list[ChunkCandidate]:
    """Return up to ``top_k`` chunks ranked by ascending cosine distance.

    An empty normalized query yields ``[]`` without embedding — ``embed_text("")``
    would produce a zero vector whose cosine distance is undefined (doc 7 caveat).
    """
    if not query.keyword:
        return []

    embedding = await provider.embed_text(query.keyword)
    # The query vector and the stored rows must come from the SAME embedding space,
    # or cosine distance compares incomparable vectors and returns plausible-but-
    # meaningless neighbours (the embedding-model rule, doc 7). The injected provider
    # carries the space that actually produced the query vector; fail fast if the
    # caller's filter labels disagree rather than silently ranking the wrong space
    # (review #1).
    if embedding.provider != embedding_provider or embedding.model != embedding_model:
        raise ValueError(
            "vector_search filter "
            f"({embedding_provider!r}, {embedding_model!r}) does not match the query "
            f"embedding's space ({embedding.provider!r}, {embedding.model!r}); the "
            "filter must target the space that produced the query vector."
        )
    distance = ChunkEmbedding.embedding_vector.cosine_distance(embedding.vector).label(
        "distance"
    )
    stmt = (
        select(Chunk.id, Chunk.parent_id, Chunk.chunk_type, distance)
        .select_from(ChunkEmbedding)
        .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
        .join(KnowledgeItem, Chunk.parent_id == KnowledgeItem.id)
        .join(Document, KnowledgeItem.document_id == Document.id)
        .where(
            ChunkEmbedding.embedding_provider == embedding_provider,
            ChunkEmbedding.embedding_model == embedding_model,
        )
    )
    stmt = apply_search_predicates(stmt, filters)
    stmt = stmt.order_by(distance.asc(), Chunk.id).limit(top_k)

    rows = (await session.execute(stmt)).all()
    candidates: list[ChunkCandidate] = []
    for rank, row in enumerate(rows, start=1):
        dist = float(row.distance)
        similarity = 1.0 - dist
        candidates.append(
            ChunkCandidate(
                chunk_id=row.id,
                knowledge_item_id=row.parent_id,
                chunk_type=row.chunk_type,
                retrieval_source="vector",
                rank=rank,
                raw_score=similarity,
                distance=dist,
                similarity=similarity,
            )
        )
    return candidates
