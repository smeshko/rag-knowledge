"""Resolve a SearchRequest's request-driven filters into a FilterSet (doc 7 § 3).

Pure function. The invariant constraints (status floor ``ready``, active-version
match, ``parent_type == knowledge_item``) are not represented here — they are
always-on SQL predicates applied inside the retrieval legs. ``build_filters`` only
resolves the knobs the request controls.
"""

from __future__ import annotations

from rag_recipes.retrieval.types import FilterSet, SearchRequest


def build_filters(request: SearchRequest) -> FilterSet:
    """Resolve defaults (``category="recipes"``, ``item_type="recipe"``) and pass
    through ``subcategory`` / ``document_ids`` / ``exclude_needs_review``."""
    return FilterSet(
        category=request.category or "recipes",
        item_type=request.item_type or "recipe",
        subcategory=request.subcategory,
        document_ids=list(request.document_ids),
        exclude_needs_review=request.exclude_needs_review,
    )
