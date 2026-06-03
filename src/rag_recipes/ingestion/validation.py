"""Pure validation layer for extracted recipe candidates (doc 4 § Validation).

This module is **pure**: no DB session, no LLM provider, no storage, and no
``Settings`` singleton (``get_settings()``). Thresholds for soft validation are
passed in explicitly so the functions stay trivially testable without an env.

The rules here are **post-parse invariants** applied to a ``recipe.v1``
``ExtractedRecipe`` that has already passed 9.2's Pydantic schema. They are
defence-in-depth over that schema: where 9.2 enforces structure/typing only
(field presence, nesting), this layer enforces *semantics* (``item_type`` value,
the ``[0, 1]`` confidence range, span-id provenance). Some rules are therefore
unreachable through the real 9.2 parse path — they remain as intentional guards
against a future schema change or a candidate constructed by other means.

``validate_hard`` collects **all** failures (it does not short-circuit) so logs
and the caller see every reason a candidate was dropped.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from rag_recipes.ingestion.pipeline.extraction import ExtractedRecipe
from rag_recipes.ingestion.pipeline.windows import Window

__all__ = [
    "HardValidationError",
    "HardValidationFailure",
    "validate_hard",
]


@dataclass(frozen=True)
class HardValidationFailure:
    """One hard-validation rule violation (rejects the candidate; no item stored).

    ``code`` is a stable machine identifier (e.g. ``"item_type_not_recipe"``);
    ``message`` carries human detail such as the offending span id or value.
    """

    code: str
    message: str


class HardValidationError(Exception):
    """Raised when a candidate fails hard validation and must not be persisted.

    Carries the full ``failures`` list so the caller (9.4 orchestration) can log
    every reason. Application-level — caught/logged by the caller; never bubbles
    up to flip the parent ``ExtractionRun`` status (DECISIONS #3).
    """

    def __init__(self, failures: list[HardValidationFailure]) -> None:
        self.failures = failures
        codes = ", ".join(f.code for f in failures)
        super().__init__(f"hard validation failed: {codes}")


def _is_out_of_range(value: float) -> bool:
    """True when ``value`` is not a real float in ``[0.0, 1.0]``.

    ``bool`` is a subclass of ``int`` but is not a valid confidence value here,
    so ``True``/``False`` are treated as out of range regardless of numeric value.
    """
    if isinstance(value, bool):
        return True
    return not (0.0 <= value <= 1.0)


def _confidence_pairs(extracted: ExtractedRecipe) -> Iterator[tuple[str, float]]:
    """Yield ``(label, value)`` for every confidence float in the candidate.

    Driven off the typed model fields (doc 2 canonical confidence shapes) rather
    than an arbitrary dict walk, so the traversal is exhaustive and stable.
    """
    confidence = extracted.confidence
    yield "item.overall", confidence.overall
    yield "item.boundary", confidence.boundary
    fields = confidence.fields
    yield "item.fields.title", fields.title
    yield "item.fields.summary", fields.summary
    yield "item.fields.yield", fields.yield_
    yield "item.fields.ingredients", fields.ingredients
    yield "item.fields.steps", fields.steps
    for ingredient in extracted.structured_data.ingredients:
        conf = ingredient.confidence
        position = ingredient.position
        yield f"ingredient[{position}].overall", conf.overall
        yield f"ingredient[{position}].quantity", conf.quantity
        yield f"ingredient[{position}].unit", conf.unit
        yield f"ingredient[{position}].item", conf.item
        yield f"ingredient[{position}].normalization", conf.normalization
    for step in extracted.structured_data.steps:
        step_conf = step.confidence
        number = step.step_number
        yield f"step[{number}].overall", step_conf.overall
        yield f"step[{number}].ordering", step_conf.ordering


def validate_hard(extracted: ExtractedRecipe, window: Window) -> list[HardValidationFailure]:
    """Return every hard-validation failure for ``extracted`` (empty = valid).

    Hard failures reject the candidate entirely — the caller persists no
    ``KnowledgeItem`` (doc 2 § 4). Collects all failures without short-circuiting.
    """
    failures: list[HardValidationFailure] = []

    if extracted.item_type != "recipe":
        failures.append(
            HardValidationFailure(
                code="item_type_not_recipe",
                message=f"item_type={extracted.item_type!r} is not 'recipe'",
            )
        )

    if not extracted.title.strip():
        failures.append(
            HardValidationFailure(
                code="missing_title",
                message="title is empty or whitespace-only",
            )
        )

    if not extracted.source_span_ids:
        failures.append(
            HardValidationFailure(
                code="missing_source_span_ids",
                message="source_span_ids is empty",
            )
        )

    # Item-level and step-level cited spans must all exist in the input window.
    # Collect candidate ids in order, dedup, and emit one failure per offending id.
    window_ids = set(window.span_ids)
    cited_ids: list[str] = list(extracted.source_span_ids)
    for step in extracted.structured_data.steps:
        cited_ids.extend(step.source_span_ids)
    seen: set[str] = set()
    for span_id in cited_ids:
        if span_id in window_ids or span_id in seen:
            continue
        seen.add(span_id)
        failures.append(
            HardValidationFailure(
                code="source_span_not_in_window",
                message=f"source span {span_id!r} is not in the input window",
            )
        )

    for label, value in _confidence_pairs(extracted):
        if _is_out_of_range(value):
            failures.append(
                HardValidationFailure(
                    code="confidence_out_of_range",
                    message=f"{label} confidence {value!r} is outside [0.0, 1.0]",
                )
            )

    for ingredient in extracted.structured_data.ingredients:
        if not ingredient.raw_text.strip():
            failures.append(
                HardValidationFailure(
                    code="ingredient_missing_raw_text",
                    message=f"ingredient at position {ingredient.position} has blank raw_text",
                )
            )

    return failures
