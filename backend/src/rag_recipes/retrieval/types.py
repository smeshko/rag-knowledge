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


@dataclass
class MergedChunk:
    """A chunk after reciprocal-rank-fusion of the two legs (doc 7 § 8).

    ``score`` is the boosted, weighted RRF score; for a chunk matched by both legs
    it is the SUM of the two contributions. ``sources`` lists the legs that
    contributed (``["keyword", "vector"]`` for a both-legs chunk, keyword-first).
    """

    chunk_id: str
    knowledge_item_id: str
    chunk_type: ChunkType
    score: float
    sources: list[str]


@dataclass
class ItemResult:
    """Merged chunks grouped under their parent KnowledgeItem (doc 7 § 9).

    ``item_score`` is the best matched chunk's score plus a capped supporting-chunk
    bonus for matching across multiple distinct chunk types. ``matched_chunks`` are
    the item's contributing ``MergedChunk``s.
    """

    knowledge_item_id: str
    item_score: float
    matched_chunks: list[MergedChunk]


# --- Result envelope (doc 7 § 11) — the internal shape Epic 13 maps to the API ---


@dataclass
class ResultItem:
    """The KnowledgeItem projection in a search result."""

    knowledge_item_id: str
    item_type: str
    title: str
    summary: str | None
    status: str


@dataclass
class ResultDocument:
    """The parent Document projection in a search result."""

    document_id: str
    title: str
    author: str


@dataclass
class MatchedChunkRef:
    """A matched chunk reference with its fused score."""

    chunk_id: str
    chunk_type: ChunkType
    score: float


@dataclass
class SourceCitation:
    """A source-span citation with a human-readable page label (doc 7 § 10)."""

    source_span_id: str
    label: str


@dataclass
class KnowledgeItemResult:
    """One grouped, fetched item-level result."""

    item: ResultItem
    document: ResultDocument
    item_score: float
    matched_chunks: list[MatchedChunkRef]
    source_citations: list[SourceCitation]


@dataclass
class SearchDebug:
    """Diagnostic counts for a search run (Epic 13 gates its dev-only exposure)."""

    mode: str
    normalized_query: str
    keyword_candidates: int
    vector_candidates: int
    merged_chunks: int
    grouped_items: int
    # Epic 18: whether a reranker actually reordered the candidates. Defaulted False
    # and additive; 18.2 sets it True only when a rerank is applied.
    rerank_applied: bool = False


@dataclass
class SearchResult:
    """The internal search envelope: ranked item results + a debug payload."""

    items: list[KnowledgeItemResult]
    debug: SearchDebug
