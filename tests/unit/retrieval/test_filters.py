"""Unit tests for build_filters (doc 7 § 3)."""

from __future__ import annotations

from rag_recipes.retrieval.filters import build_filters
from rag_recipes.retrieval.types import SearchRequest


def test_applies_defaults_for_a_bare_request() -> None:
    fs = build_filters(SearchRequest(query="white beans"))
    assert fs.category == "recipes"
    assert fs.item_type == "recipe"
    assert fs.subcategory is None
    assert fs.document_ids == []
    assert fs.exclude_needs_review is True


def test_passes_through_explicit_values() -> None:
    fs = build_filters(
        SearchRequest(
            query="risotto",
            category="cookbooks",
            subcategory="italian",
            item_type="technique",
            document_ids=["doc_1", "doc_2"],
            exclude_needs_review=False,
        )
    )
    assert fs.category == "cookbooks"
    assert fs.subcategory == "italian"
    assert fs.item_type == "technique"
    assert fs.document_ids == ["doc_1", "doc_2"]
    assert fs.exclude_needs_review is False


def test_document_ids_are_copied_not_aliased() -> None:
    ids = ["doc_1"]
    fs = build_filters(SearchRequest(query="q", document_ids=ids))
    ids.append("doc_2")
    # The FilterSet snapshot is independent of later mutation of the input list.
    assert fs.document_ids == ["doc_1"]
