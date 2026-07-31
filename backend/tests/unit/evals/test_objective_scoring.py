"""Unit tests for the objective extraction-field scorers (Epic 15 Phase 15.1).

Pure value-level tests — no DB, no provider, no I/O. Every expectation is
hand-computed so the precision/recall/F1 math is asserted directly.
"""

from __future__ import annotations

from typing import Any

import pytest
from evals.scoring.objective import (
    _parse_minutes,
    score_ingredient_count,
    score_ingredients_detail,
    score_source_span_ids,
    score_step_count,
    score_times,
    score_title,
    score_yield,
)


class TestScoreTitle:
    def test_exact_match_is_also_normalized(self) -> None:
        score = score_title("Tomato Soup", "Tomato Soup")
        assert score.exact is True
        assert score.normalized is True

    def test_normalized_only_match(self) -> None:
        score = score_title("  Tomato  Soup ", "tomato soup")
        assert score.exact is False
        assert score.normalized is True

    def test_mismatch(self) -> None:
        score = score_title("Bean Stew", "Tomato Soup")
        assert score.exact is False
        assert score.normalized is False


class TestScoreYield:
    def test_case_insensitive_match(self) -> None:
        assert score_yield("Serves 4", "serves 4") is True

    def test_whitespace_trimmed_match(self) -> None:
        assert score_yield("  serves 4  ", "serves 4") is True

    def test_none_vs_none_matches(self) -> None:
        assert score_yield(None, None) is True

    def test_none_vs_value_is_a_miss(self) -> None:
        assert score_yield(None, "serves 4") is False
        assert score_yield("serves 4", None) is False

    def test_different_values_miss(self) -> None:
        assert score_yield("serves 6", "serves 4") is False


class TestParseMinutes:
    @pytest.mark.parametrize(
        ("value", "minutes"),
        [
            ("15 minutes", 15.0),
            ("15 min", 15.0),
            ("1 hr 30 min", 90.0),
            ("1 hour", 60.0),
            ("2 hours", 120.0),
            ("90", 90.0),
            ("PT45M", 45.0),
            ("PT1H30M", 90.0),
        ],
    )
    def test_parses_known_formats(self, value: str, minutes: float) -> None:
        assert _parse_minutes(value) == minutes

    def test_none_and_unparseable_return_none(self) -> None:
        assert _parse_minutes(None) is None
        assert _parse_minutes("") is None
        assert _parse_minutes("until golden") is None


class TestScoreTimes:
    def test_equivalent_formats_match(self) -> None:
        score = score_times("15 minutes", "15 min")
        assert score.match is True
        assert score.actual_minutes == 15.0
        assert score.expected_minutes == 15.0

    def test_hour_form_matches_minute_form(self) -> None:
        assert score_times("1 hr 30 min", "90 minutes").match is True

    def test_different_durations_miss(self) -> None:
        assert score_times("20 minutes", "15 minutes").match is False

    def test_missing_actual_is_a_miss(self) -> None:
        score = score_times(None, "15 minutes")
        assert score.match is False
        assert score.actual_minutes is None
        assert score.expected_minutes == 15.0

    def test_missing_or_unparseable_expected_is_a_miss_not_a_crash(self) -> None:
        assert score_times("15 minutes", None).match is False
        assert score_times("15 minutes", "until done").match is False


class TestScoreCounts:
    def test_ingredient_count(self) -> None:
        assert score_ingredient_count(3, 3) is True
        assert score_ingredient_count(2, 3) is False

    def test_step_count(self) -> None:
        assert score_step_count(4, 4) is True
        assert score_step_count(5, 4) is False


def _ingredient(
    raw_text: str,
    quantity_value: float | None,
    unit_normalized: str | None,
    item_normalized: str | None,
    preparation: str | None,
) -> dict[str, Any]:
    return {
        "raw_text": raw_text,
        "quantity_value": quantity_value,
        "unit_normalized": unit_normalized,
        "item_normalized": item_normalized,
        "preparation": preparation,
    }


_ONION = _ingredient("1 onion, diced", 1.0, None, "onion", "diced")
_BEANS = _ingredient("2 cups white beans", 2.0, "cup", "white beans", None)
_STOCK = _ingredient("4 cups vegetable stock", 4.0, "cup", "vegetable stock", None)


class TestScoreIngredientsDetail:
    def test_all_correct(self) -> None:
        breakdown = score_ingredients_detail([_ONION, _BEANS], [_ONION, _BEANS])
        assert breakdown.matched == 2
        assert breakdown.unmatched_actual == 0
        assert breakdown.unmatched_expected == 0
        assert breakdown.precision == pytest.approx(1.0)
        assert breakdown.recall == pytest.approx(1.0)
        assert breakdown.f1 == pytest.approx(1.0)
        assert all(value == pytest.approx(1.0) for value in breakdown.sub_field_accuracy.values())

    def test_one_wrong_sub_field(self) -> None:
        wrong_quantity = dict(_BEANS, quantity_value=3.0)
        breakdown = score_ingredients_detail([_ONION, wrong_quantity], [_ONION, _BEANS])
        # 9 of the 10 sub-field slots are correct on both sides.
        assert breakdown.matched == 2
        assert breakdown.precision == pytest.approx(0.9)
        assert breakdown.recall == pytest.approx(0.9)
        assert breakdown.f1 == pytest.approx(0.9)
        assert breakdown.sub_field_accuracy["quantity_value"] == pytest.approx(0.5)
        assert breakdown.sub_field_accuracy["raw_text"] == pytest.approx(1.0)
        assert breakdown.sub_field_accuracy["item_normalized"] == pytest.approx(1.0)

    def test_extra_actual_is_a_false_positive(self) -> None:
        breakdown = score_ingredients_detail([_ONION, _BEANS, _STOCK], [_ONION, _BEANS])
        # 10 correct sub-fields over 15 actual slots and 10 expected slots.
        assert breakdown.matched == 2
        assert breakdown.unmatched_actual == 1
        assert breakdown.unmatched_expected == 0
        assert breakdown.precision == pytest.approx(10 / 15)
        assert breakdown.recall == pytest.approx(1.0)
        assert breakdown.f1 == pytest.approx(0.8)

    def test_missing_expected_is_a_false_negative(self) -> None:
        breakdown = score_ingredients_detail([_ONION], [_ONION, _BEANS])
        assert breakdown.matched == 1
        assert breakdown.unmatched_actual == 0
        assert breakdown.unmatched_expected == 1
        assert breakdown.precision == pytest.approx(1.0)
        assert breakdown.recall == pytest.approx(0.5)
        assert breakdown.f1 == pytest.approx(2 / 3)

    def test_reordered_list_recovered_by_item_fallback(self) -> None:
        breakdown = score_ingredients_detail([_BEANS, _ONION], [_ONION, _BEANS])
        assert breakdown.matched == 2
        assert breakdown.unmatched_actual == 0
        assert breakdown.unmatched_expected == 0
        assert breakdown.precision == pytest.approx(1.0)
        assert breakdown.recall == pytest.approx(1.0)
        assert breakdown.f1 == pytest.approx(1.0)

    def test_absent_identity_fields_do_not_align_unrelated_ingredients(self) -> None:
        # A degraded extraction that normalized nothing must not be handed
        # credit for the sub-fields that are absent on both sides.
        salt = _ingredient("1 tsp salt", 1.0, None, None, None)
        pepper = _ingredient("1 tsp black pepper", 1.0, None, None, None)
        breakdown = score_ingredients_detail([salt], [pepper])
        assert breakdown.matched == 0
        assert breakdown.unmatched_actual == 1
        assert breakdown.unmatched_expected == 1
        assert breakdown.precision == pytest.approx(0.0)
        assert breakdown.recall == pytest.approx(0.0)
        assert breakdown.f1 == pytest.approx(0.0)

    def test_raw_text_fallback_still_aligns_when_item_is_absent(self) -> None:
        actual = _ingredient("1 tsp salt", 1.0, "teaspoon", None, None)
        expected = _ingredient("1 tsp salt", 1.0, "teaspoon", "salt", None)
        breakdown = score_ingredients_detail([actual], [expected])
        assert breakdown.matched == 1
        assert breakdown.sub_field_accuracy["item_normalized"] == pytest.approx(0.0)

    def test_both_empty(self) -> None:
        breakdown = score_ingredients_detail([], [])
        assert breakdown.matched == 0
        assert breakdown.precision == pytest.approx(1.0)
        assert breakdown.recall == pytest.approx(1.0)
        assert breakdown.f1 == pytest.approx(1.0)


class TestScoreSourceSpanIds:
    def test_identical_sets(self) -> None:
        score = score_source_span_ids(["span_a", "span_b"], ["span_b", "span_a"])
        assert score.precision == pytest.approx(1.0)
        assert score.recall == pytest.approx(1.0)
        assert score.f1 == pytest.approx(1.0)

    def test_extra_actual_lowers_precision(self) -> None:
        score = score_source_span_ids(["span_a", "span_b", "span_c"], ["span_a", "span_b"])
        assert score.precision == pytest.approx(2 / 3)
        assert score.recall == pytest.approx(1.0)
        assert score.f1 == pytest.approx(0.8)

    def test_missing_expected_lowers_recall(self) -> None:
        score = score_source_span_ids(["span_a"], ["span_a", "span_b"])
        assert score.precision == pytest.approx(1.0)
        assert score.recall == pytest.approx(0.5)
        assert score.f1 == pytest.approx(2 / 3)

    def test_both_empty_scores_perfect(self) -> None:
        score = score_source_span_ids([], [])
        assert score.precision == pytest.approx(1.0)
        assert score.recall == pytest.approx(1.0)
        assert score.f1 == pytest.approx(1.0)

    def test_empty_actual_against_expected_scores_zero(self) -> None:
        score = score_source_span_ids([], ["span_a"])
        assert score.precision == pytest.approx(0.0)
        assert score.recall == pytest.approx(0.0)
        assert score.f1 == pytest.approx(0.0)
