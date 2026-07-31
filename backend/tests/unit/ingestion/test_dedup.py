"""Unit tests for the pure dedup layer in ingestion.pipeline.dedup.

No DB, no provider, no ``Settings``: ``compute_candidate_score`` runs over
``ExtractedRecipe`` candidates built directly from the 9.2 Pydantic models
(reusing the factories in ``test_validation``), and ``select_best`` runs over
hand-built ``CandidateRef``s. The score formula is pinned in ``dedup.py``'s
docstring (DECISIONS #1); these tests assert each term's contribution, the
``[0, 1]`` bounds, the missing-confidence defensiveness, and the span-coverage
cap, plus ``select_best``'s grouping and deterministic tie-break.
"""

from __future__ import annotations

import pytest

from rag_recipes.ingestion.pipeline.dedup import (
    CandidateRef,
    compute_candidate_score,
    select_best,
)
from rag_recipes.ingestion.pipeline.extraction import (
    RecipeConfidence,
    RecipeFieldConfidence,
)
from tests.unit.ingestion.test_validation import (
    _make_recipe,
    _make_structured_data,
    _recipe_confidence,
)

# --- compute_candidate_score -----------------------------------------------


def test_full_formula_combines_every_term() -> None:
    recipe = _make_recipe(
        source_span_ids=["span_001", "span_002"],
        confidence=_recipe_confidence(overall=0.8, boundary=0.6),
    )
    # 0.35*0.8 + 0.25*0.6 + 0.15*1 (ingredients) + 0.15*1 (steps) + 0.10*1 (coverage)
    score = compute_candidate_score(recipe, window_span_ids=["span_001", "span_002"])
    assert score == pytest.approx(0.28 + 0.15 + 0.15 + 0.15 + 0.10)


def test_only_confidence_terms_when_no_structure_or_spans() -> None:
    recipe = _make_recipe(
        structured_data=_make_structured_data(ingredients=[], steps=[]),
        source_span_ids=[],
        confidence=_recipe_confidence(overall=0.8, boundary=0.4),
    )
    # has_ingredients=has_steps=0, span_coverage=0 → only 0.35*0.8 + 0.25*0.4.
    score = compute_candidate_score(recipe, window_span_ids=["span_001"])
    assert score == pytest.approx(0.28 + 0.10)


def test_score_is_one_when_everything_maxed() -> None:
    recipe = _make_recipe(
        source_span_ids=["span_001"],
        confidence=_recipe_confidence(overall=1.0, boundary=1.0),
    )
    assert compute_candidate_score(recipe, window_span_ids=["span_001"]) == pytest.approx(1.0)


def test_score_is_zero_when_everything_empty() -> None:
    recipe = _make_recipe(
        structured_data=_make_structured_data(ingredients=[], steps=[]),
        source_span_ids=[],
        confidence=_recipe_confidence(overall=0.0, boundary=0.0),
    )
    assert compute_candidate_score(recipe, window_span_ids=["span_001"]) == pytest.approx(0.0)


def test_missing_confidence_defaults_to_zero() -> None:
    # A soft-failed/hand-built candidate whose confidence floats are None must not
    # crash; overall/boundary coerce to 0.0. ``model_construct`` bypasses Pydantic
    # validation to build the otherwise-impossible None shape.
    confidence = RecipeConfidence.model_construct(
        overall=None,
        boundary=None,
        fields=RecipeFieldConfidence(
            title=0.9, summary=0.9, yield_=0.9, ingredients=0.9, steps=0.9
        ),
    )
    recipe = _make_recipe(source_span_ids=["span_001"], confidence=confidence)
    # overall=boundary=0; ingredients+steps present; coverage=1 → 0.15+0.15+0.10.
    score = compute_candidate_score(recipe, window_span_ids=["span_001"])
    assert score == pytest.approx(0.15 + 0.15 + 0.10)


def test_span_coverage_caps_at_one() -> None:
    recipe = _make_recipe(
        structured_data=_make_structured_data(ingredients=[], steps=[]),
        source_span_ids=["span_001", "span_002", "span_003"],
        confidence=_recipe_confidence(overall=0.0, boundary=0.0),
    )
    # 3 cited spans over a 1-span window → coverage capped at 1.0 → 0.10.
    score = compute_candidate_score(recipe, window_span_ids=["span_001"])
    assert score == pytest.approx(0.10)


def test_span_coverage_is_partial_fraction() -> None:
    recipe = _make_recipe(
        structured_data=_make_structured_data(ingredients=[], steps=[]),
        source_span_ids=["span_001"],
        confidence=_recipe_confidence(overall=0.0, boundary=0.0),
    )
    # 1 cited span over a 2-span window → coverage 0.5 → 0.10*0.5 = 0.05.
    score = compute_candidate_score(recipe, window_span_ids=["span_001", "span_002"])
    assert score == pytest.approx(0.05)


def test_span_coverage_dedupes_repeated_citations() -> None:
    recipe = _make_recipe(
        structured_data=_make_structured_data(ingredients=[], steps=[]),
        source_span_ids=["span_001", "span_001"],
        confidence=_recipe_confidence(overall=0.0, boundary=0.0),
    )
    # Duplicate citations collapse to one unique span → 1/2 window → 0.05.
    score = compute_candidate_score(recipe, window_span_ids=["span_001", "span_002"])
    assert score == pytest.approx(0.05)


# --- select_best -----------------------------------------------------------


def _ref(item_id: str, title: str, score: float, run: str = "run_1") -> CandidateRef:
    return CandidateRef(
        item_id=item_id,
        normalized_title=title,
        candidate_score=score,
        extraction_run_id=run,
    )


def test_select_best_empty_input() -> None:
    assert select_best([]) == ([], [])


def test_same_title_keeps_highest_score() -> None:
    high = _ref("item_a", "tomato soup", 0.9)
    low = _ref("item_b", "tomato soup", 0.5)
    chosen, discarded = select_best([low, high])
    assert chosen == [high]
    assert discarded == [low]


def test_tie_breaks_on_lowest_item_id() -> None:
    b = _ref("item_b", "tomato soup", 0.7)
    a = _ref("item_a", "tomato soup", 0.7)
    chosen, discarded = select_best([b, a])
    assert chosen == [a]
    assert discarded == [b]


def test_distinct_titles_all_chosen() -> None:
    soup = _ref("item_a", "tomato soup", 0.6)
    bread = _ref("item_b", "banana bread", 0.4)
    chosen, discarded = select_best([soup, bread])
    assert set(chosen) == {soup, bread}
    assert discarded == []


def test_group_of_three_keeps_one_discards_two() -> None:
    best = _ref("item_a", "tomato soup", 0.9)
    mid = _ref("item_b", "tomato soup", 0.6)
    worst = _ref("item_c", "tomato soup", 0.3)
    chosen, discarded = select_best([mid, worst, best])
    assert chosen == [best]
    assert set(discarded) == {mid, worst}
