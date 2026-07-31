"""Objective per-field scorers for extraction evaluation (doc 12 § 5).

**Purity contract:** every function here is a pure value-level comparison —
no DB session, no provider, no ``Settings``, no file I/O. Scorers take plain
``str``/``float``/``list``/``dict`` values (the driver extracts them from the
parsed ``ExtractedRecipe`` and the fixture's ``expected.json``) and return
small frozen dataclasses the driver serialises.

Per-field metric definitions (doc 12 § 5):

- **title** — exact string equality, and equality after the production
  ``normalize_title`` normalization (the same function that populates
  ``KnowledgeItem.normalized_title`` — DECISIONS #2).
- **yield** — whitespace-trimmed, case-insensitive string match; ``None`` vs
  ``None`` is a match.
- **times** (``prep_time``/``cook_time``/``total_time``) — each side parsed to
  minutes via :func:`_parse_minutes` and compared numerically; an unparseable
  or absent side scores as a miss, never raises.
- **ingredient / step counts** — exact integer match.
- **ingredient detail** — actual↔expected ingredients aligned by position with
  a normalized-``item`` fallback (DECISIONS #3), then the five doc-12 § 5
  sub-fields (``raw_text``, ``quantity_value``, ``unit_normalized``,
  ``item_normalized``, ``preparation``) scored per matched pair and aggregated
  to precision / recall / F1 over sub-field slots, plus per-sub-field accuracy.
  An unmatched actual ingredient is a false positive; an unmatched expected
  ingredient a false negative.
- **source_span_ids** — set-membership precision / recall / F1 of the actual
  span-id set against the expected set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag_recipes.ingestion.pipeline.persist import normalize_title

__all__ = [
    "IngredientScoreBreakdown",
    "SpanScore",
    "TimeScore",
    "TitleScore",
    "score_ingredient_count",
    "score_ingredients_detail",
    "score_source_span_ids",
    "score_step_count",
    "score_times",
    "score_title",
    "score_yield",
]

INGREDIENT_SUB_FIELDS = (
    "raw_text",
    "quantity_value",
    "unit_normalized",
    "item_normalized",
    "preparation",
)


@dataclass(frozen=True)
class TitleScore:
    """Exact and normalized-title match for one recipe."""

    exact: bool
    normalized: bool


@dataclass(frozen=True)
class TimeScore:
    """Numeric (parsed-minutes) comparison for one duration field."""

    match: bool
    actual_minutes: float | None
    expected_minutes: float | None


@dataclass(frozen=True)
class SpanScore:
    """Set-membership precision/recall/F1 for span-id provenance."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class IngredientScoreBreakdown:
    """Precision/recall/F1 over ingredient sub-field slots plus per-sub-field accuracy.

    ``sub_field_accuracy`` is the fraction of *matched* pairs on which each of
    the five sub-fields agrees (``0.0`` when nothing matched).
    """

    precision: float
    recall: float
    f1: float
    matched: int
    unmatched_actual: int
    unmatched_expected: int
    sub_field_accuracy: dict[str, float]


_NUMBER_ONLY = re.compile(r"\d+(?:\.\d+)?")
_HOURS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b")
_MINUTES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b")
_ISO_DURATION = re.compile(r"pt(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?")


def _parse_minutes(value: str | None) -> float | None:
    """Parse a human/ISO duration string to minutes; ``None`` when unparseable.

    Total over the observed fixture formats — ``"15 minutes"``, ``"15 min"``,
    ``"1 hr 30 min"``, ``"1 hour"``, bare ``"90"``, and ISO-8601 ``"PT1H30M"``.
    Never raises: anything unrecognised (or ``None``/empty) parses to ``None``.
    """
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    if _NUMBER_ONLY.fullmatch(text):
        return float(text)
    iso = _ISO_DURATION.fullmatch(text)
    if iso is not None and any(iso.groups()):
        hours = float(iso.group("hours") or 0)
        minutes = float(iso.group("minutes") or 0)
        seconds = float(iso.group("seconds") or 0)
        return hours * 60 + minutes + seconds / 60
    total = 0.0
    found = False
    for match in _HOURS.finditer(text):
        total += float(match.group(1)) * 60
        found = True
    for match in _MINUTES.finditer(text):
        total += float(match.group(1))
        found = True
    return total if found else None


def _precision_recall_f1(true_positives: int, actual_total: int, expected_total: int) -> SpanScore:
    """Shared precision/recall/F1 math for the two set-style scorers.

    Both sides empty is the vacuous perfect score; any division by zero
    resolves to ``0.0``.
    """
    if actual_total == 0 and expected_total == 0:
        return SpanScore(precision=1.0, recall=1.0, f1=1.0)
    precision = true_positives / actual_total if actual_total else 0.0
    recall = true_positives / expected_total if expected_total else 0.0
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator else 0.0
    return SpanScore(precision=precision, recall=recall, f1=f1)


def score_title(actual: str, expected: str) -> TitleScore:
    """Score a title exact (raw ``==``) and after ``normalize_title`` (doc 12 § 5)."""
    return TitleScore(
        exact=actual == expected,
        normalized=normalize_title(actual) == normalize_title(expected),
    )


def _normalize_optional(value: str | None) -> str | None:
    return value.strip().casefold() if isinstance(value, str) else value


def score_yield(actual: str | None, expected: str | None) -> bool:
    """Whitespace-trimmed, case-insensitive yield match; ``None`` == ``None``."""
    return _normalize_optional(actual) == _normalize_optional(expected)


def score_times(actual: str | None, expected: str | None) -> TimeScore:
    """Compare two duration strings by parsed minutes; unparseable sides miss."""
    actual_minutes = _parse_minutes(actual)
    expected_minutes = _parse_minutes(expected)
    match = (
        actual_minutes is not None
        and expected_minutes is not None
        and actual_minutes == expected_minutes
    )
    return TimeScore(match=match, actual_minutes=actual_minutes, expected_minutes=expected_minutes)


def score_ingredient_count(actual: int, expected: int) -> bool:
    """Exact ingredient-count match."""
    return actual == expected


def score_step_count(actual: int, expected: int) -> bool:
    """Exact step-count match."""
    return actual == expected


def _sub_field_equal(field: str, actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    actual_value = actual.get(field)
    expected_value = expected.get(field)
    if isinstance(actual_value, str) or isinstance(expected_value, str):
        return _normalize_optional(actual_value) == _normalize_optional(expected_value)
    return bool(actual_value == expected_value)


def _ingredient_keys_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """True when two ingredients denote the same line (item, or raw text, agrees)."""
    return _sub_field_equal("item_normalized", actual, expected) or _sub_field_equal(
        "raw_text", actual, expected
    )


def _align_ingredients(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Pair actual↔expected indices: position first, normalized-``item`` fallback.

    A positional pair only holds when the two ingredients denote the same line
    (matching normalized ``item_normalized`` or ``raw_text``) — otherwise both
    fall through to the fallback pass, which matches remaining actuals to
    remaining expecteds one-to-one in order. This recovers reordered and
    inserted/dropped lines without a cost-matrix solver (DECISIONS #3).
    """
    pairs: list[tuple[int, int]] = []
    unmatched_actual: list[int] = []
    matched_expected: set[int] = set()
    for index, actual_ingredient in enumerate(actual):
        if index < len(expected) and _ingredient_keys_match(actual_ingredient, expected[index]):
            pairs.append((index, index))
            matched_expected.add(index)
        else:
            unmatched_actual.append(index)
    for actual_index in unmatched_actual:
        for expected_index, expected_ingredient in enumerate(expected):
            if expected_index in matched_expected:
                continue
            if _ingredient_keys_match(actual[actual_index], expected_ingredient):
                pairs.append((actual_index, expected_index))
                matched_expected.add(expected_index)
                break
    return sorted(pairs)


def score_ingredients_detail(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> IngredientScoreBreakdown:
    """Align ingredients and score the five doc-12 § 5 sub-fields to P/R/F1.

    Precision counts correct sub-fields over *all actual* sub-field slots and
    recall over *all expected* slots, so unmatched-actual ingredients dilute
    precision (false positives) and unmatched-expected dilute recall (false
    negatives). ``sub_field_accuracy`` is computed over matched pairs only.
    """
    pairs = _align_ingredients(actual, expected)
    correct_by_field = dict.fromkeys(INGREDIENT_SUB_FIELDS, 0)
    for actual_index, expected_index in pairs:
        for field in INGREDIENT_SUB_FIELDS:
            if _sub_field_equal(field, actual[actual_index], expected[expected_index]):
                correct_by_field[field] += 1
    correct_total = sum(correct_by_field.values())
    slots_per_ingredient = len(INGREDIENT_SUB_FIELDS)
    prf = _precision_recall_f1(
        correct_total,
        len(actual) * slots_per_ingredient,
        len(expected) * slots_per_ingredient,
    )
    matched = len(pairs)
    return IngredientScoreBreakdown(
        precision=prf.precision,
        recall=prf.recall,
        f1=prf.f1,
        matched=matched,
        unmatched_actual=len(actual) - matched,
        unmatched_expected=len(expected) - matched,
        sub_field_accuracy={
            field: (correct_by_field[field] / matched if matched else 0.0)
            for field in INGREDIENT_SUB_FIELDS
        },
    )


def score_source_span_ids(actual: list[str], expected: list[str]) -> SpanScore:
    """Set-membership precision/recall/F1 of actual span ids vs expected (doc 12 § 5)."""
    actual_set = set(actual)
    expected_set = set(expected)
    return _precision_recall_f1(
        len(actual_set & expected_set), len(actual_set), len(expected_set)
    )
