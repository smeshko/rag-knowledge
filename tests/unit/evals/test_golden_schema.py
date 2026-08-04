"""Golden-shape rules (Epic 23.2 TASK-001).

Every test here perturbs *one* thing away from a known-good golden and asserts
the rule engine names it. The structure matters: a rule that fires on several
unrelated defects is a rule nobody can act on, and a rule that fires on a valid
golden blocks legitimate fixtures.

The recurring theme in what these rules catch is that **the scorers never raise**
— each defect below degrades to a silently wrong number, which is why shape has
to be checked separately from behaviour.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from evals.golden_schema import validate_expected_json

FIXTURES = Path("data/fixtures/synthetic_recipes")


def _golden() -> dict[str, Any]:
    """A valid golden: the committed ginger-tea fixture, read from disk.

    Built from the real file rather than a literal so this suite cannot drift
    into asserting a shape the repo does not actually contain.
    """
    return json.loads(
        (FIXTURES / "synthetic" / "ginger-tea" / "expected.json").read_text(encoding="utf-8")
    )


NAME = "ginger-tea"


def test_a_committed_golden_is_valid() -> None:
    """The baseline. If this fails, every other test here is meaningless."""
    assert validate_expected_json(NAME, _golden()) == []


# --- strict key sets --------------------------------------------------------


def test_an_unknown_top_level_key_is_rejected() -> None:
    golden = _golden()
    golden["verified"] = True
    errors = validate_expected_json(NAME, golden)
    assert any("unknown key 'verified'" in e for e in errors)


def test_a_missing_top_level_key_is_rejected() -> None:
    golden = _golden()
    del golden["title"]
    errors = validate_expected_json(NAME, golden)
    assert any("missing key 'title'" in e for e in errors)


def test_a_typod_ingredient_key_is_rejected() -> None:
    """AC2 — the defect no behavioural test can catch.

    ``_sub_field_equal`` reads sub-fields with ``.get()``
    (``objective.py:196-201``), so ``item_normalised`` is not compared, not
    reported, and not scored. The fixture reports a *perfect* ingredient F1
    while one field is silently unmeasured. Only a strict key set finds it.
    """
    golden = _golden()
    ingredient = golden["structured_data"]["ingredients"][0]
    ingredient["item_normalised"] = ingredient.pop("item_normalized")

    errors = validate_expected_json(NAME, golden)
    assert any("unknown key 'item_normalised'" in e for e in errors)
    assert any("missing key 'item_normalized'" in e for e in errors)


def test_an_unknown_step_key_is_rejected() -> None:
    golden = _golden()
    golden["structured_data"]["steps"][0]["number"] = 1
    errors = validate_expected_json(NAME, golden)
    assert any("unknown key 'number'" in e for e in errors)


# --- durations --------------------------------------------------------------


@pytest.mark.parametrize("value", ["overnight", "1-2 days", "until golden", ""])
def test_an_unparseable_time_is_rejected(value: str) -> None:
    """AC3 — an unparseable time is *dropped from the mean*, not scored zero.

    ``evals/extraction.py:485-489`` only appends to the accuracy accumulator
    when the golden side parsed. So "overnight" does not make the fixture look
    bad — it removes the fixture from that field's denominator, quietly
    improving the average over the fixtures that remain.
    """
    golden = _golden()
    golden["structured_data"]["cook_time"] = value
    errors = validate_expected_json(NAME, golden)
    assert any("cook_time" in e and "does not parse" in e for e in errors)


@pytest.mark.parametrize("value", ["15 minutes", "1 hr 30 min", "90", "PT1H30M", "1 hour"])
def test_every_format_the_scorer_accepts_is_allowed(value: str) -> None:
    """The negative control: the rule must not be stricter than the scorer."""
    golden = _golden()
    golden["structured_data"]["cook_time"] = value
    assert validate_expected_json(NAME, golden) == []


def test_a_null_time_is_allowed() -> None:
    """Strictness is on shape, never content — a recipe may state no cook time."""
    golden = _golden()
    golden["structured_data"]["cook_time"] = None
    assert validate_expected_json(NAME, golden) == []


# --- span provenance --------------------------------------------------------


@pytest.mark.parametrize(
    "spans",
    [
        [],
        ["span_eval_wrong-name"],
        ["span_eval_ginger-tea", "span_eval_ginger-tea"],
        ["ginger-tea"],
        "span_eval_ginger-tea",
    ],
)
def test_wrong_source_span_ids_are_rejected(spans: Any) -> None:
    """AC4 — a wrong span id means the fixture is never scored at all.

    The extraction fails *hard* validation, so the fixture disappears from the
    report rather than scoring low — the most misleading failure mode available.
    """
    golden = _golden()
    golden["source_span_ids"] = spans
    errors = validate_expected_json(NAME, golden)
    assert any("source_span_ids" in e for e in errors)


def test_the_span_id_is_derived_from_the_fixture_name_argument() -> None:
    """A golden valid under one name must be invalid under another."""
    assert validate_expected_json("white-bean-soup", _golden()) != []


# --- ingredient identity ----------------------------------------------------


def test_an_ingredient_without_an_identity_field_is_rejected() -> None:
    """AC5 — an unidentifiable line is dropped from F1, not counted as a miss.

    ``_ingredient_keys_match`` (``objective.py:212-227``) needs a non-empty
    ``item_normalized`` or ``raw_text`` on both sides to align a line at all.
    """
    golden = _golden()
    golden["structured_data"]["ingredients"][0]["item_normalized"] = None
    golden["structured_data"]["ingredients"][0]["raw_text"] = "   "
    errors = validate_expected_json(NAME, golden)
    assert any("non-empty" in e and "ingredients[0]" in e for e in errors)


def test_one_identity_field_is_enough() -> None:
    golden = _golden()
    golden["structured_data"]["ingredients"][0]["item_normalized"] = None
    assert validate_expected_json(NAME, golden) == []


@pytest.mark.parametrize("quantity", ["2", "two", True])
def test_a_non_numeric_quantity_is_rejected(quantity: Any) -> None:
    """A string quantity compares as a string and never equals the float 2.0."""
    golden = _golden()
    golden["structured_data"]["ingredients"][0]["quantity_value"] = quantity
    errors = validate_expected_json(NAME, golden)
    assert any("quantity_value" in e for e in errors)


def test_a_null_quantity_is_allowed() -> None:
    golden = _golden()
    golden["structured_data"]["ingredients"][0]["quantity_value"] = None
    assert validate_expected_json(NAME, golden) == []


# --- structural collapse ----------------------------------------------------


def test_a_non_dict_structured_data_is_rejected() -> None:
    """The silent-zero case: ``_golden_structured`` returns {} for a non-dict."""
    golden = _golden()
    golden["structured_data"] = []
    errors = validate_expected_json(NAME, golden)
    assert any("structured_data" in e for e in errors)


@pytest.mark.parametrize("field", ["ingredients", "steps"])
def test_an_empty_list_is_rejected(field: str) -> None:
    golden = _golden()
    golden["structured_data"][field] = []
    errors = validate_expected_json(NAME, golden)
    assert any(f"structured_data.{field}" in e for e in errors)


def test_a_blank_step_is_rejected() -> None:
    golden = _golden()
    golden["structured_data"]["steps"][0]["text"] = "  "
    errors = validate_expected_json(NAME, golden)
    assert any("steps[0]" in e for e in errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [("item_type", "note"), ("title", ""), ("title", None)],
)
def test_contract_fields_must_hold_their_canonical_values(field: str, value: Any) -> None:
    golden = _golden()
    golden[field] = value
    assert any(field in e for e in validate_expected_json(NAME, golden))


def test_a_wrong_structured_schema_is_rejected() -> None:
    golden = _golden()
    golden["structured_data"]["schema"] = "recipe.v2"
    errors = validate_expected_json(NAME, golden)
    assert any("schema" in e for e in errors)


def test_validation_does_not_mutate_its_input() -> None:
    """The engine is pure; a caller may validate then write the same object."""
    golden = _golden()
    before = copy.deepcopy(golden)
    validate_expected_json(NAME, golden)
    assert golden == before
