"""Shared, dependency-free dataclasses for the retrieval package (doc 7).

These types are the seam between the retrieval legs: the keyword leg (12.1), the
vector leg (12.2) and the hybrid facade (12.3) all import them from here, so no
leg imports across to another (DECISIONS #1). They are plain stdlib dataclasses,
not Pydantic — the HTTP request/response models are Epic 13's concern (DECISIONS
#2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_recipes.storage.enums import ChunkType


@dataclass
class SearchRequest:
    """The normalized internal search input (Epic 13 maps the HTTP body onto it)."""

    query: str
    category: str = "recipes"
    subcategory: str | None = None
    item_type: str = "recipe"
    document_ids: list[str] = field(default_factory=list)
    mode: str = "hybrid"
    limit: int = 10
    exclude_needs_review: bool = True


@dataclass(frozen=True)
class NormalizedQuery:
    """A query split into the verbatim ``original`` and the normalized ``keyword``."""

    original: str
    keyword: str


@dataclass(frozen=True)
class FilterSet:
    """The resolved, request-driven filter values applied to a search.

    The invariant constraints (status floor ``ready``, ``source_version ==
    active_source_version``, ``parent_type == knowledge_item``) are not carried
    here — they are always-on SQL predicates built inside the retrieval legs.
    ``exclude_needs_review`` is carried for symmetry/debug; user search always
    floors at ``ready``.
    """

    category: str
    item_type: str
    subcategory: str | None
    document_ids: list[str]
    exclude_needs_review: bool


@dataclass
class ChunkCandidate:
    """One retrieved chunk from either leg (doc 7 § 6).

    ``retrieval_source`` is ``"keyword"`` or ``"vector"``. ``rank`` is the 1-based
    ordinal position within that leg's result list. ``raw_score`` is the leg's
    native score (``ts_rank_cd`` for keyword). ``distance``/``similarity`` are only
    populated by the vector leg (12.2); the keyword leg leaves them ``None``.
    """

    chunk_id: str
    knowledge_item_id: str
    chunk_type: ChunkType
    retrieval_source: str
    rank: int
    raw_score: float
    distance: float | None = None
    similarity: float | None = None
