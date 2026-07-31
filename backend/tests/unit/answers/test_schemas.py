"""Unit tests for the POST /api/v1/answers request/response schemas (doc 8 §§ 1, 6)."""

from __future__ import annotations

from rag_recipes.api.schemas.answers import (
    AnswerBody,
    AnswerCitation,
    AnswerOptions,
    AnswerRequestBody,
    AnswerResponse,
    AnswerRetrievalOptions,
    Recommendation,
)
from rag_recipes.api.schemas.search import KnowledgeItemResult, SearchFilters


def test_request_defaults() -> None:
    body = AnswerRequestBody(query="cozy soups")
    assert body.query == "cozy soups"
    assert body.category == "recipes"
    assert body.subcategory is None
    assert isinstance(body.filters, SearchFilters)
    assert body.retrieval.mode == "hybrid"
    assert body.retrieval.limit is None
    assert body.answer.style == "recommendation"
    assert body.answer.include_results is False


def test_request_option_models_have_documented_defaults() -> None:
    assert AnswerRetrievalOptions().mode == "hybrid"
    assert AnswerRetrievalOptions().limit is None
    assert AnswerOptions().style == "recommendation"
    assert AnswerOptions().include_results is False


def test_request_independent_default_instances() -> None:
    # Pydantic deep-copies model defaults; mutating one body's filters must not
    # leak into another's.
    a = AnswerRequestBody(query="a")
    b = AnswerRequestBody(query="b")
    a.filters.document_ids.append("doc-1")
    assert b.filters.document_ids == []


def test_response_defaults_empty_collections() -> None:
    resp = AnswerResponse(
        query="cozy soups",
        answer=AnswerBody(style="recommendation", text="...", citations=["cite_1"]),
    )
    assert resp.recommendations == []
    assert resp.citations == []
    assert resp.results == []
    assert resp.warnings == []


def test_response_full_shape() -> None:
    resp = AnswerResponse(
        query="cozy soups",
        answer=AnswerBody(
            style="recommendation", text="A strong match.", citations=["cite_1"]
        ),
        recommendations=[
            Recommendation(
                knowledge_item_id="item_1",
                title="Tomato and White Bean Soup",
                reason="Uses white beans directly.",
                citation_ids=["cite_1"],
            )
        ],
        citations=[
            AnswerCitation(
                citation_id="cite_1",
                knowledge_item_id="item_1",
                source_span_id="span_1",
                label="Cookbook, page 42",
            )
        ],
        warnings=["heads up"],
    )
    dumped = resp.model_dump()
    assert dumped["answer"]["style"] == "recommendation"
    assert dumped["recommendations"][0]["title"] == "Tomato and White Bean Soup"
    assert dumped["citations"][0]["source_span_id"] == "span_1"
    assert dumped["warnings"] == ["heads up"]


def test_results_field_uses_search_knowledge_item_result() -> None:
    # The reused type is the search projection, not a duplicate.
    field = AnswerResponse.model_fields["results"]
    assert field.annotation == list[KnowledgeItemResult]
