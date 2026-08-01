"""Request/response Pydantic models for POST /api/v1/answers (doc 8 §§ 1, 6).

Pure API projections, parallel to ``api/schemas/search.py``. The request reuses
``SearchFilters`` and the response reuses the search ``KnowledgeItemResult`` for
``results`` — the answer envelope wraps retrieval, it does not re-model it. ``mode``
and ``style`` are plain ``str`` so invalid values land in the doc-6 ``invalid_request``
envelope via handler validation, not FastAPI's raw 422.

The response ``citations[]`` and each ``recommendations[].title`` are reconstructed
by the backend from the context pack (Phase 17.2), never trusted from the LLM — the
model supplies only ``cite_N`` references.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from rag_recipes.api.schemas.search import (
    KnowledgeItemResult,
    RetrievalDebugInfo,
    SearchFilters,
)

# --- Request (doc 8 § 1) ---


class AnswerRetrievalOptions(BaseModel):
    mode: str = "hybrid"
    limit: int | None = None


class AnswerOptions(BaseModel):
    style: str = "recommendation"
    include_results: bool = False
    include_debug: bool = False


class AnswerRequestBody(BaseModel):
    query: str
    category: str = "recipes"
    subcategory: str | None = None
    filters: SearchFilters = SearchFilters()
    retrieval: AnswerRetrievalOptions = AnswerRetrievalOptions()
    answer: AnswerOptions = AnswerOptions()


# --- Response (doc 8 § 6) ---


class AnswerBody(BaseModel):
    style: str
    text: str
    citations: list[str]


class Recommendation(BaseModel):
    knowledge_item_id: str
    title: str
    reason: str
    citation_ids: list[str]


class AnswerCitation(BaseModel):
    citation_id: str
    knowledge_item_id: str
    source_span_id: str
    label: str


class AnswerDebugInfo(BaseModel):
    """Dev-only answer diagnostics (doc 8 § 11), gated like the search debug payload."""

    retrieval_mode: str
    model: str
    prompt_version: str
    context_item_count: int
    citation_count: int
    retrieval_debug: RetrievalDebugInfo | None = None


class AnswerResponse(BaseModel):
    query: str
    answer: AnswerBody
    recommendations: list[Recommendation] = []
    citations: list[AnswerCitation] = []
    results: list[KnowledgeItemResult] = []
    warnings: list[str] = []
    debug: AnswerDebugInfo | None = None

    @model_serializer(mode="wrap")
    def _drop_absent_debug(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Serialize with the ``debug`` key absent (not null) when gated off (D4).

        Every other legitimately-null field keeps appearing — this is NOT
        ``exclude_none``. Requires ``separate_input_output_schemas=False`` on
        the app or this serializer collapses the model's OpenAPI schema.
        """
        data: dict[str, Any] = handler(self)
        if data.get("debug") is None:
            data.pop("debug", None)
        return data
