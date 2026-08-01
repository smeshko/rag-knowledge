"""Unit tests for the absent-``debug`` contract under ``response_model`` (D4).

``SearchResponse`` / ``AnswerResponse`` carry a wrap serializer that drops the
``debug`` key when it is ``None`` (absent, not null) while keeping every other
legitimately-null field. The OpenAPI assertions at the bottom are the only
check that catches the serializer's schema collapse: without
``separate_input_output_schemas=False`` on the app, a ``@model_serializer``
documents both models as ``{"type": "object", "additionalProperties": true}``
with zero properties while the wire bytes stay correct.
"""

from __future__ import annotations

from rag_recipes.api.app import app
from rag_recipes.api.schemas.answers import (
    AnswerBody,
    AnswerDebugInfo,
    AnswerResponse,
)
from rag_recipes.api.schemas.search import RetrievalDebugInfo, SearchResponse


def _answer_body() -> AnswerBody:
    return AnswerBody(style="recommendation", text="t", citations=[])


def test_search_response_drops_none_debug_key() -> None:
    data = SearchResponse(query="q", results=[]).model_dump()
    assert "debug" not in data
    assert data["query"] == "q"
    assert data["results"] == []


def test_search_response_keeps_populated_debug() -> None:
    debug = RetrievalDebugInfo(retrieval_mode="hybrid", normalized_query="q")
    data = SearchResponse(query="q", results=[], debug=debug).model_dump()
    assert data["debug"]["retrieval_mode"] == "hybrid"
    # Other None fields inside debug still serialize as null (no exclude_none).
    assert data["debug"]["embedding_model"] is None


def test_answer_response_drops_none_debug_key() -> None:
    data = AnswerResponse(query="q", answer=_answer_body()).model_dump()
    assert "debug" not in data
    # Legitimately-empty/None-able siblings keep appearing.
    assert data["recommendations"] == []
    assert data["citations"] == []
    assert data["results"] == []
    assert data["warnings"] == []


def test_answer_response_keeps_populated_debug() -> None:
    debug = AnswerDebugInfo(
        retrieval_mode="hybrid",
        model="m",
        prompt_version="v1",
        context_item_count=0,
        citation_count=0,
    )
    data = AnswerResponse(query="q", answer=_answer_body(), debug=debug).model_dump()
    assert data["debug"]["model"] == "m"
    assert data["debug"]["retrieval_debug"] is None


def test_openapi_schemas_keep_full_properties_despite_serializer() -> None:
    """The wrap serializer must not untype the two most FE-critical bodies.

    Guarded by ``separate_input_output_schemas=False`` in ``api/app.py`` —
    removing that flag makes this test fail with an empty ``properties``.
    """
    schemas = app.openapi()["components"]["schemas"]
    for name in ("SearchResponse", "AnswerResponse"):
        properties = schemas[name].get("properties") or {}
        assert properties, f"{name} documented without properties (schema collapsed)"
        assert {"query", "results", "debug"} <= set(properties)
