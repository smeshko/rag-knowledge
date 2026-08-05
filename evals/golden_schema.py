"""Shape rules for golden ``expected.json`` files (Epic 23.2).

Why this module exists: **every malformed golden degrades to a wrong number
rather than an error.** The scorers read the golden defensively — a non-dict
``structured_data`` scores zero silently (``evals/extraction.py:281-296``), a
typo'd ingredient key is read with ``.get()`` and simply never compared
(``evals/scoring/objective.py:196-201``), and an unparseable time string is
dropped from its field's mean entirely (``evals/extraction.py:485-489``). A
fixture can therefore look perfect while measuring nothing. These rules are the
only place that difference is visible.

Purity contract, mirroring ``evals/scoring/objective.py``: no ``Settings``, no
provider, no database, no filesystem. Input is already-parsed JSON; output is a
list of human-readable violations, empty when the golden is well formed. The
duration rule routes through the **production** ``score_times`` rather than
re-implementing a duration regex, so it can never disagree with the scorer it
protects.

Strictness is on *shape*, never on *content*: every scored value may be
``null`` — a recipe that states no cook time should say so honestly — but a
value that is present must be one the scorer can actually read.
"""

from __future__ import annotations

import re
from typing import Any

from evals.extraction import synthetic_span_id
from evals.scoring.objective import INGREDIENT_SUB_FIELDS, score_times

__all__ = [
    "INGREDIENT_KEYS",
    "STEP_KEYS",
    "STRUCTURED_KEYS",
    "TIME_FIELDS",
    "TOP_LEVEL_KEYS",
    "VERIFICATION_VALUES",
    "parse_verification_status",
    "validate_expected_json",
]

#: The only accepted values of a fixture's ``Verification:`` field. There is no
#: default: an absent or unrecognised value is a failure, never "assume draft"
#: and never "assume verified". A missing-status default would make a forgotten
#: fixture look finished, which is the single thing the field exists to prevent.
VERIFICATION_VALUES = frozenset({"draft", "verified"})

_VERIFICATION = re.compile(
    r"^\s*[-*]?\s*Verification:\s*(?P<status>[A-Za-z]+)\b", re.MULTILINE
)

#: Exactly the keys the two committed ``synthetic`` goldens carry. Unknown keys
#: are rejected rather than ignored: ``_sub_field_equal`` reads sub-fields with
#: ``.get()``, so an ``item_normalised`` typo would sit in the golden forever,
#: unscored and unnoticed, while the fixture reported a perfect score.
TOP_LEVEL_KEYS = frozenset({"item_type", "title", "source_span_ids", "structured_data"})
STRUCTURED_KEYS = frozenset(
    {"schema", "yield", "prep_time", "cook_time", "total_time", "ingredients", "steps"}
)
INGREDIENT_KEYS = frozenset(INGREDIENT_SUB_FIELDS)
STEP_KEYS = frozenset({"text"})

TIME_FIELDS = ("prep_time", "cook_time", "total_time")

_ITEM_TYPE = "recipe"
_SCHEMA = "recipe.v1"


def parse_verification_status(notes: str | None) -> str | None:
    """Read a fixture's ``Verification:`` status from its ``notes.md`` text.

    Returns ``"draft"``/``"verified"``, or ``None`` when the field is absent or
    carries an unrecognised value — the caller treats both as a failure. Free
    text after the keyword is allowed and ignored, so a reviewer can annotate:
    ``- Verification: verified — Ivo, 2026-08-05, corrected the yield``.

    Lives here rather than in the test because it is a rule, not a fixture: 23.5
    and 23.6 both need to ask "is this set verified?" without importing a test.
    """
    if not notes:
        return None
    match = _VERIFICATION.search(notes)
    if match is None:
        return None
    status = match.group("status").lower()
    return status if status in VERIFICATION_VALUES else None


def _key_errors(where: str, actual: Any, allowed: frozenset[str]) -> list[str]:
    """Strict key-set comparison, reported as separate missing/unknown lines."""
    if not isinstance(actual, dict):
        return [f"{where}: expected an object, got {type(actual).__name__}"]
    present = frozenset(actual)
    errors = []
    for key in sorted(allowed - present):
        errors.append(f"{where}: missing key {key!r}")
    for key in sorted(present - allowed):
        errors.append(f"{where}: unknown key {key!r} (it would be silently ignored by the scorer)")
    return errors


def _validate_ingredient(where: str, ingredient: Any) -> list[str]:
    errors = _key_errors(where, ingredient, INGREDIENT_KEYS)
    if errors or not isinstance(ingredient, dict):
        return errors

    # ``_ingredient_keys_match`` needs a non-empty identity on BOTH sides or the
    # line can never be aligned, and so can never be scored at all — it does not
    # score zero, it drops out of precision and recall entirely
    # (``objective.py:212-227``).
    identities = [
        ingredient.get(field)
        for field in ("item_normalized", "raw_text")
        if isinstance(ingredient.get(field), str) and ingredient.get(field, "").strip()
    ]
    if not identities:
        errors.append(
            f"{where}: needs a non-empty 'item_normalized' or 'raw_text' — "
            "without one the line can never be aligned, so it is dropped from "
            "the ingredient F1 rather than scored as a miss"
        )

    quantity = ingredient.get("quantity_value")
    if quantity is not None and not isinstance(quantity, (int, float)):
        errors.append(
            f"{where}: 'quantity_value' must be a number or null, got "
            f"{type(quantity).__name__} ({quantity!r})"
        )
    if isinstance(quantity, bool):  # bool is an int subclass — catch it explicitly
        errors.append(f"{where}: 'quantity_value' must be a number or null, got a bool")

    for field in ("raw_text", "unit_normalized", "item_normalized", "preparation"):
        value = ingredient.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(
                f"{where}: {field!r} must be a string or null, got {type(value).__name__}"
            )
    return errors


def _validate_step(where: str, step: Any) -> list[str]:
    errors = _key_errors(where, step, STEP_KEYS)
    if errors or not isinstance(step, dict):
        return errors
    text = step.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append(f"{where}: 'text' must be a non-empty string")
    return errors


def _validate_structured(structured: Any) -> list[str]:
    where = "structured_data"
    errors = _key_errors(where, structured, STRUCTURED_KEYS)
    if not isinstance(structured, dict):
        return errors

    schema = structured.get("schema")
    if schema != _SCHEMA:
        errors.append(f"{where}.schema: must be {_SCHEMA!r}, got {schema!r}")

    yield_value = structured.get("yield")
    if yield_value is not None and (
        not isinstance(yield_value, str) or not yield_value.strip()
    ):
        errors.append(f"{where}.yield: must be a non-empty string or null, got {yield_value!r}")

    for field in TIME_FIELDS:
        value = structured.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            errors.append(f"{where}.{field}: must be a string or null, got {type(value).__name__}")
            continue
        # score_times is the scorer's own entry point; asking it to parse the
        # golden side is exactly what the aggregate does, so this rule cannot
        # drift from the code it protects.
        if score_times(None, value).expected_minutes is None:
            errors.append(
                f"{where}.{field}: {value!r} does not parse as a duration, so the "
                "scorer drops this field from its mean entirely rather than "
                "counting it as a miss — write null instead of an unparseable value"
            )

    ingredients = structured.get("ingredients")
    if not isinstance(ingredients, list) or not ingredients:
        errors.append(f"{where}.ingredients: must be a non-empty list")
    else:
        for index, ingredient in enumerate(ingredients):
            errors += _validate_ingredient(f"{where}.ingredients[{index}]", ingredient)

    steps = structured.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append(f"{where}.steps: must be a non-empty list")
    else:
        for index, step in enumerate(steps):
            errors += _validate_step(f"{where}.steps[{index}]", step)

    return errors


def validate_expected_json(fixture_name: str, expected: Any) -> list[str]:
    """Return every shape violation in ``expected``, empty when it is well formed.

    ``fixture_name`` is needed because ``source_span_ids`` is not free-form: the
    synthetic window carries exactly one span id derived from the fixture name
    (``evals/extraction.py:210-216``, ``:238-252``). Any other value makes the
    *extraction* fail hard validation, so the fixture is never scored at all —
    a failure that surfaces as a missing row rather than a low score.
    """
    errors = _key_errors("<root>", expected, TOP_LEVEL_KEYS)
    if not isinstance(expected, dict):
        return errors

    item_type = expected.get("item_type")
    if item_type != _ITEM_TYPE:
        errors.append(f"item_type: must be {_ITEM_TYPE!r}, got {item_type!r}")

    title = expected.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append(f"title: must be a non-empty string, got {title!r}")

    expected_spans = [synthetic_span_id(fixture_name)]
    spans = expected.get("source_span_ids")
    if spans != expected_spans:
        errors.append(
            f"source_span_ids: must be exactly {expected_spans!r}, got {spans!r} — "
            "the synthetic window contains that one id, so any other value fails "
            "hard validation and the fixture is never scored"
        )

    errors += _validate_structured(expected.get("structured_data"))
    return errors
