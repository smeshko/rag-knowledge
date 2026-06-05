"""Request/response Pydantic models for POST /api/v1/search (doc 6 § 7).

Pure API projections: the request tolerates the doc-6 body (``mode``/``query`` are
plain ``str`` and validated in the handler so failures land in the doc-6
``invalid_request`` envelope, not FastAPI's raw 422); the response is a projection of
the Epic 12 ``retrieval.search.SearchResult``. The reserved JSON keys ``schema`` /
``yield`` are carried by ``schema_`` / ``yield_`` with aliases (the ``schema_`` /
``yield_`` idiom from the extraction model), so the serialized keys read ``schema`` /
``yield``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Request ---


class SearchFilters(BaseModel):
    item_type: str | None = "recipe"
    document_ids: list[str] = []
    exclude_needs_review: bool = True


class SearchRequestBody(BaseModel):
    query: str
    category: str = "recipes"
    subcategory: str | None = None
    filters: SearchFilters = SearchFilters()
    limit: int | None = None
    include_debug: bool = False
    mode: str = "hybrid"


# --- Response ---


class ResultItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    item_type: str
    schema_: str = Field(alias="schema", serialization_alias="schema")
    title: str
    summary: str | None
    status: str
    confidence: dict[str, Any] | None


class DisplayProjection(BaseModel):
    title: str
    subtitle: str | None
    snippet: str | None
    badges: list[str]


class StructuredPreview(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(alias="schema", serialization_alias="schema")
    yield_: str | None = Field(alias="yield", serialization_alias="yield")
    top_ingredients: list[str]


class ResultDocument(BaseModel):
    id: str
    title: str
    author: str


class MatchedChunk(BaseModel):
    chunk_id: str
    chunk_type: str
    score: float


class SourceCitation(BaseModel):
    source_span_id: str
    label: str
    locator: dict[str, Any] | None


class KnowledgeItemResult(BaseModel):
    type: Literal["knowledge_item_result"] = "knowledge_item_result"
    item: ResultItem
    display: DisplayProjection
    structured_preview: StructuredPreview
    document: ResultDocument
    matched_chunks: list[MatchedChunk]
    source_citations: list[SourceCitation]


class RetrievalDebugInfo(BaseModel):
    """Loose projection of the Epic 12 debug payload (doc 7 § 12).

    Modeled permissively so doc-7 evolution does not break the schema.
    """

    retrieval_mode: str
    normalized_query: str
    embedding_model: str | None = None
    keyword_top_k: int | None = None
    vector_top_k: int | None = None
    keyword_candidates: int | None = None
    vector_candidates: int | None = None
    merged_candidates: int | None = None
    grouped_items: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[KnowledgeItemResult]
    debug: RetrievalDebugInfo | None = None
