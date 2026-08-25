"""Pure edit layer for knowledge items (Epic 22.1).

Editing exists because approve/reject is too blunt: a recipe flagged
``no_ingredients`` because the model missed the ingredient block can otherwise
only be waved through broken or rejected outright, and reject is terminal.

This module is **pure**, on the same contract as ``ingestion/validation``: no DB
session, no provider, no ``Settings`` singleton. It takes the primitives a
``KnowledgeItem`` row carries, returns the primitives that should replace them,
and leaves persistence, guards and transactions to the API layer.

Two functions carry the epic:

- ``apply_edit`` turns a ``RecipeEdit`` plus the current row into the new
  ``title`` / ``normalized_title`` / ``summary`` / ``body_text`` /
  ``structured_data``.
- ``warnings_for_item`` re-derives the soft-validation warning codes from a
  *persisted* row, so a reviewer who fixes "no ingredients" stops seeing the flag
  that said so. It reconstructs an ``ExtractedRecipe`` and defers to
  ``validate_soft`` rather than reimplementing the rules — there is exactly one
  definition of what "too short" means.

**What an edit costs depends on the item's status, and that is the caller's
problem, not this module's.** A ``needs_review`` item has no chunks and no
embeddings (``build_chunks`` returns ``[]`` for anything that is not ``READY``),
so an edit there is a pure row rewrite and the *edited* text is what gets
chunked when the reviewer approves. Editing an indexed item additionally
requires dropping its chunks and embeddings and re-indexing from the saved text
— which the API layer does, in the edit transaction plus an
``index_knowledge_item`` job. These functions are the same either way: they
compute the new field values and nothing else.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag_recipes.ingestion.pipeline.composition import compose_body_text, normalize_title
from rag_recipes.ingestion.pipeline.extraction import (
    SCHEMA_VERSION,
    ExtractedRecipe,
    RecipeConfidence,
    RecipeV1StructuredData,
)
from rag_recipes.ingestion.validation import SoftValidationThresholds, validate_soft

__all__ = [
    "UNSET",
    "EditedItem",
    "RecipeEdit",
    "Unset",
    "apply_edit",
    "warnings_for_item",
]


class Unset(Enum):
    """Sentinel type distinguishing "field absent" from "field set to null".

    Modelled as a single-member ``Enum`` so static analysis narrows
    ``str | None | Unset`` correctly on an ``is not UNSET`` check. Clearing a
    summary is a real operation, so ``None`` cannot double as "not supplied".
    """

    TOKEN = "unset"


UNSET = Unset.TOKEN

# A human authored the line, so the machine's parse of the old line no longer
# describes it; the per-line confidences become 1.0 because a typed correction
# is not a guess. The parsed sub-fields are nulled rather than re-derived —
# re-running normalization over an edited line is future work.
_NULLED_INGREDIENT_FIELDS = (
    "quantity_text",
    "quantity_value",
    "unit_raw",
    "unit_normalized",
    "item_text",
    "item_normalized",
    "preparation",
    "notes",
)
_HUMAN_INGREDIENT_CONFIDENCE = {
    "overall": 1.0,
    "quantity": 1.0,
    "unit": 1.0,
    "item": 1.0,
    "normalization": 1.0,
}
_HUMAN_STEP_CONFIDENCE = {"overall": 1.0, "ordering": 1.0}

# Used when a persisted row is missing a confidence value entirely. 1.0 means
# "no evidence of low confidence" — a gap in the data must not invent a warning
# that the ingest pipeline never raised.
_ASSUMED_CONFIDENCE = 1.0


@dataclass(frozen=True)
class RecipeEdit:
    """The editable subset of a recipe, as submitted by a reviewer.

    Every field defaults to ``UNSET`` (absent). ``title`` cannot be nulled — the
    column is NOT NULL and an untitled recipe is not a recipe. The two lists are
    whole-array replacement, which is what a form submits and what makes add,
    remove and reorder fall out for free.

    Deliberately absent: ``confidence``, ``source_span_ids``, ``schema``,
    ``item_type`` and ``warnings`` are machine-owned provenance, not
    client-writable.
    """

    title: str | Unset = UNSET
    summary: str | None | Unset = UNSET
    yield_: str | None | Unset = UNSET
    prep_time: str | None | Unset = UNSET
    cook_time: str | None | Unset = UNSET
    total_time: str | None | Unset = UNSET
    ingredients: list[str] | Unset = UNSET
    steps: list[str] | Unset = UNSET

    def is_empty(self) -> bool:
        """True when the edit names no field at all."""
        return all(
            getattr(self, name) is UNSET
            for name in (
                "title",
                "summary",
                "yield_",
                "prep_time",
                "cook_time",
                "total_time",
                "ingredients",
                "steps",
            )
        )


@dataclass(frozen=True)
class EditedItem:
    """The column values an edit produces, ready for whole-object assignment.

    ``body_text_rebuilt`` records whether the composition actually re-ran, so a
    caller (and a test) can tell a title-only edit from one that invalidated the
    body.
    """

    title: str
    normalized_title: str
    summary: str | None
    body_text: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    body_text_rebuilt: bool = False


def _rekey_by_text(rows: list[dict[str, Any]], key: str) -> dict[str, deque[dict[str, Any]]]:
    """Index existing rows by their text so unchanged lines can be matched back.

    Matching on text rather than on list position is what makes a pure reorder
    cheap: the row moves, keeps its parse and its span provenance, and only its
    number changes. Duplicate lines are held in a queue so N identical lines in
    still map to N identical lines out.
    """
    by_text: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        text = row.get(key)
        if isinstance(text, str):
            by_text[text].append(row)
    return by_text


def _edit_ingredients(existing: list[dict[str, Any]], lines: list[str]) -> list[dict[str, Any]]:
    """Rebuild the ingredient rows from the submitted lines.

    A line that matches an existing row passes through byte-identical apart from
    its ``position``, which is renumbered from list order. A line that is new or
    changed becomes a human-authored row: parse nulled, confidences 1.0,
    ``edited: true``.
    """
    available = _rekey_by_text(existing, "raw_text")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        pool = available.get(line)
        if pool:
            rows.append({**deepcopy(pool.popleft()), "position": index})
            continue
        rows.append(
            {
                "position": index,
                "raw_text": line,
                **dict.fromkeys(_NULLED_INGREDIENT_FIELDS),
                "confidence": dict(_HUMAN_INGREDIENT_CONFIDENCE),
                "edited": True,
            }
        )
    return rows


def _edit_steps(existing: list[dict[str, Any]], lines: list[str]) -> list[dict[str, Any]]:
    """Rebuild the step rows from the submitted lines.

    Unchanged steps keep their ``source_span_ids`` and confidence; an added or
    rewritten step carries an empty ``source_span_ids``, because a human wrote it
    and claiming a page cited it would be a lie.
    """
    available = _rekey_by_text(existing, "text")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        pool = available.get(line)
        if pool:
            rows.append({**deepcopy(pool.popleft()), "step_number": index})
            continue
        rows.append(
            {
                "step_number": index,
                "text": line,
                "source_span_ids": [],
                "confidence": dict(_HUMAN_STEP_CONFIDENCE),
                "edited": True,
            }
        )
    return rows


def _rows(structured: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The list under ``key``, keeping only dict entries (JSONB is untyped)."""
    value = structured.get(key) or []
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _text(row: dict[str, Any], key: str) -> str:
    """One row's text, coerced — a null in JSONB must not raise on a join."""
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _join_lines(structured: dict[str, Any], list_key: str, text_key: str) -> str:
    """Recompose a text block from its row list."""
    return "\n".join(_text(row, text_key) for row in _rows(structured, list_key))


def apply_edit(
    *,
    title: str,
    summary: str | None,
    body_text: str,
    structured_data: dict[str, Any],
    edit: RecipeEdit,
) -> EditedItem:
    """Apply ``edit`` to a persisted item's fields and return the new values.

    Only the fields the edit names change; everything else — including
    ``structured_data`` keys this layer knows nothing about, and the ``warnings``
    list, which the caller overwrites from ``warnings_for_item`` — passes
    through.

    ``body_text`` is rebuilt **only when the ingredient or step lines actually
    change**. A title-only edit leaves it byte-identical, so LLM prose that lives
    only in ``body_text`` survives an edit that does not invalidate it. When the
    lines do change, ``ingredients_text`` / ``steps_text`` are rewritten to match
    them: those two fields take precedence over the row lists everywhere text is
    resolved, so leaving them stale would index the pre-edit ingredients under a
    corrected recipe.
    """
    new_title = title if isinstance(edit.title, Unset) else edit.title
    new_summary = summary if isinstance(edit.summary, Unset) else edit.summary

    # Deep copy, not `dict(...)`: the caller passes a SQLAlchemy-loaded JSONB
    # dict, and sharing a nested list or confidence dict between the input and
    # the result means a later in-place tweak of one silently rewrites the other
    # — including the pre-edit snapshot the caller builds from the input.
    structured: dict[str, Any] = deepcopy(structured_data)
    for attr, key in (
        ("yield_", "yield"),
        ("prep_time", "prep_time"),
        ("cook_time", "cook_time"),
        ("total_time", "total_time"),
    ):
        value = getattr(edit, attr)
        if not isinstance(value, Unset):
            structured[key] = value

    ingredients_changed = False
    steps_changed = False
    if not isinstance(edit.ingredients, Unset):
        existing = _rows(structured, "ingredients")
        rows = _edit_ingredients(existing, edit.ingredients)
        ingredients_changed = rows != existing
        structured["ingredients"] = rows
    if not isinstance(edit.steps, Unset):
        existing = _rows(structured, "steps")
        rows = _edit_steps(existing, edit.steps)
        steps_changed = rows != existing
        structured["steps"] = rows

    # Only the submitted list's own text block is refreshed. Rewriting the other
    # one would destroy content that lives in the blob but not in the rows — an
    # item flagged ``no_steps`` keeps its method prose in ``steps_text``, and an
    # ingredients-only edit must not wipe it.
    if ingredients_changed:
        structured["ingredients_text"] = _join_lines(structured, "ingredients", "raw_text")
    if steps_changed:
        structured["steps_text"] = _join_lines(structured, "steps", "text")

    new_body_text = body_text
    if ingredients_changed or steps_changed:
        new_body_text = compose_body_text(title=new_title, structured=structured)

    return EditedItem(
        title=new_title,
        normalized_title=normalize_title(new_title),
        summary=new_summary,
        body_text=new_body_text,
        structured_data=structured,
        body_text_rebuilt=ingredients_changed or steps_changed,
    )


def _confidence_value(source: dict[str, Any], key: str) -> float:
    """Read one confidence float, falling back when the row does not carry it."""
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _ASSUMED_CONFIDENCE
    return float(value)


def _int_or(value: Any, fallback: int) -> int:
    """Coerce a persisted integer, tolerating a null or a junk value.

    Every guard in this reconstruction exists because a raise here would be a
    500 on the edit endpoint. Nothing in the ingest path writes these shapes —
    they are defence against hand-edited or future rows, not a known case.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return value


def _str_or_none(value: Any) -> str | None:
    """Coerce a persisted optional string, mapping any other type to ``None``."""
    return value if isinstance(value, str) else None


def _ingredient_for_validation(row: dict[str, Any], position: int) -> dict[str, Any]:
    """Project a persisted ingredient row onto the ``recipe.v1`` model shape."""
    confidence = row.get("confidence")
    confidence = confidence if isinstance(confidence, dict) else {}
    return {
        "position": _int_or(row.get("position"), position),
        "raw_text": _text(row, "raw_text"),
        **{name: row.get(name) for name in _NULLED_INGREDIENT_FIELDS},
        "confidence": {
            name: _confidence_value(confidence, name)
            for name in ("overall", "quantity", "unit", "item", "normalization")
        },
    }


def _step_for_validation(row: dict[str, Any], step_number: int) -> dict[str, Any]:
    """Project a persisted step row onto the ``recipe.v1`` model shape."""
    confidence = row.get("confidence")
    confidence = confidence if isinstance(confidence, dict) else {}
    span_ids = row.get("source_span_ids")
    return {
        "step_number": _int_or(row.get("step_number"), step_number),
        "text": _text(row, "text"),
        "source_span_ids": [s for s in span_ids if isinstance(s, str)]
        if isinstance(span_ids, list)
        else [],
        "confidence": {
            name: _confidence_value(confidence, name) for name in ("overall", "ordering")
        },
    }


def warnings_for_item(
    *,
    title: str,
    summary: str | None,
    body_text: str,
    source_span_ids: list[str],
    structured_data: dict[str, Any],
    confidence: dict[str, Any] | None,
    thresholds: SoftValidationThresholds,
    item_type: str = "recipe",
) -> list[str]:
    """Re-derive the soft-validation warning codes for a persisted item.

    Reconstructs the ``ExtractedRecipe`` the ingest pipeline would have validated
    and runs ``validate_soft`` over it, so an edit's effect on the flags is
    decided by the same rules that raised them. Over an *unedited* row this
    reproduces exactly the codes already in ``structured_data["warnings"]``.

    ``structured_data["warnings"]`` is deliberately not read back in — the
    warnings are recomputed from the content, never carried forward, which is
    what lets a fixed flag actually clear. Confidence-derived warnings
    (``low_overall_confidence``, ``low_boundary_confidence``) therefore survive
    every content edit: they judge whether the recipe was cut out of the page
    correctly, and retyping an ingredient line does not attest to that. They are
    cleared by approving, not by fixing.
    """
    conf = confidence if isinstance(confidence, dict) else {}
    fields = conf.get("fields")
    fields = fields if isinstance(fields, dict) else {}

    extracted = ExtractedRecipe(
        item_type=item_type,
        title=title,
        summary=summary,
        body_text=body_text,
        source_span_ids=list(source_span_ids),
        structured_data=RecipeV1StructuredData.model_validate(
            {
                "schema": _str_or_none(structured_data.get("schema")) or SCHEMA_VERSION,
                "yield": _str_or_none(structured_data.get("yield")),
                "prep_time": _str_or_none(structured_data.get("prep_time")),
                "cook_time": _str_or_none(structured_data.get("cook_time")),
                "total_time": _str_or_none(structured_data.get("total_time")),
                "ingredients_text": _str_or_none(structured_data.get("ingredients_text")),
                "ingredients": [
                    _ingredient_for_validation(row, index)
                    for index, row in enumerate(_rows(structured_data, "ingredients"), start=1)
                ],
                "steps_text": _str_or_none(structured_data.get("steps_text")),
                "steps": [
                    _step_for_validation(row, index)
                    for index, row in enumerate(_rows(structured_data, "steps"), start=1)
                ],
            }
        ),
        confidence=RecipeConfidence.model_validate(
            {
                "overall": _confidence_value(conf, "overall"),
                "boundary": _confidence_value(conf, "boundary"),
                "fields": {
                    name: _confidence_value(fields, name)
                    for name in ("title", "summary", "yield", "ingredients", "steps")
                },
            }
        ),
        warnings=[],
    )
    return [warning.code for warning in validate_soft(extracted, thresholds=thresholds)]
