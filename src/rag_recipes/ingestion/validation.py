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
    "SoftValidationThresholds",
    "SoftValidationWarning",
    "validate_hard",
    "validate_soft",
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
        # ``normalization`` only: it is the one per-row confidence with a reader
        # (``validate_soft``'s low_normalization_confidence warning). The four
        # sibling ingredient axes and both step axes were range-checked here and
        # nowhere else, so they were dropped from the schema — see
        # ``IngredientConfidence`` / ``ExtractedStep``. Steps now contribute no
        # pairs at all.
        yield (
            f"ingredient[{ingredient.position}].normalization",
            ingredient.confidence.normalization,
        )


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


@dataclass(frozen=True)
class SoftValidationWarning:
    """One soft-validation concern — the item is still stored as ``needs_review``.

    ``code`` is a stable machine identifier (e.g. ``"no_ingredients"``);
    ``message`` carries human detail.
    """

    code: str
    message: str


@dataclass(frozen=True)
class SoftValidationThresholds:
    """Threshold values for soft validation, passed in to keep ``validate_soft`` pure.

    The orchestration layer (9.4) builds this from ``Settings.extraction_*`` and
    passes it down; this module never reads ``get_settings()`` itself.
    """

    min_overall_confidence: float
    min_boundary_confidence: float
    min_normalization_confidence: float
    min_recipe_chars: int
    max_recipe_chars: int
    #: Bounds on what still counts as an *assembly* recipe — see
    #: ``_is_assembly_recipe``. A step-less candidate inside all three is judged
    #: method-free by design rather than truncated or unstructured.
    assembly_min_ingredients: int
    assembly_max_ingredients: int
    assembly_max_chars: int


def _is_assembly_recipe(
    extracted: ExtractedRecipe,
    body_len: int,
    thresholds: SoftValidationThresholds,
) -> bool:
    """True when a method-free candidate is an *assembly* recipe, not a truncated one.

    Some cookbooks — the bowl / "build your own" genre especially — print recipes
    that are a title plus a list of already-documented components and nothing
    else::

        THANKSGIVING IN A BOWL
        Mashed Potatoes (page 46), shredded leftover roasted turkey, roasted
        Brussels sprouts (page 43), ..., cranberry sauce

    That is the whole recipe as printed. ``no_steps`` and ``recipe_too_short``
    were written for books where a missing method means the extraction failed, so
    against this genre they fire on correct output: ingesting one such title sent
    31 of 167 items to ``needs_review``, where nothing is chunked and the
    retrieval floor hides them entirely.

    The discriminator is size, not the absence of steps. A *truncated* recipe —
    the head half of one that spilled across a window boundary, keeping its
    ingredient block and losing its method — carries the full component list of a
    real dish: 17 to 32 rows over a long body. An assembly recipe names a handful
    of finished components in a line or two. Measured over both books ingested so
    far, every method-free candidate with ingredients sat at 3-10 rows and
    96-260 characters, and no truncated head came close; the defaults leave the
    gap between the two populations wide.

    The *lower* bound is what keeps this from swallowing the failure it most
    resembles. A recipe whose method the model wrote into ``body_text`` as prose
    but never broke into ``steps`` is also short and step-less — the extraction
    eval's tomato-soup fixture is exactly that, one ingredient and "Chop the
    tomatoes, then simmer and blend" as its whole body — and exempting it would
    promote genuinely unstructured output into search. An assembly recipe is a
    *composition*: it names several finished components. Below three there is no
    composition to speak of, only a recipe missing its method, so the floor sits
    at the smallest real component list observed.

    Requiring ingredients at all is deliberate for the same reason: a candidate
    with neither ingredients nor steps is not an assembly recipe, it is empty,
    and ``no_ingredients`` should still speak for it.
    """
    ingredients = extracted.structured_data.ingredients
    return (
        not extracted.structured_data.steps
        and thresholds.assembly_min_ingredients
        <= len(ingredients)
        <= thresholds.assembly_max_ingredients
        and body_len <= thresholds.assembly_max_chars
    )


def validate_soft(
    extracted: ExtractedRecipe,
    *,
    thresholds: SoftValidationThresholds,
) -> list[SoftValidationWarning]:
    """Return soft-validation warnings for ``extracted`` (empty = fully clean).

    Soft failures do not drop the candidate — the caller persists it with
    ``status="needs_review"`` and attaches the warning codes (doc 4 § Soft
    validation). All warnings are collected; nothing short-circuits.
    """
    warnings: list[SoftValidationWarning] = []
    structured = extracted.structured_data
    body_len = len(extracted.body_text)
    # Both of this candidate's size-shaped rules are waived together or not at
    # all: an assembly recipe is short *because* it has no method, so clearing
    # ``no_steps`` while leaving ``recipe_too_short`` behind would still park it
    # in review for the same underlying fact.
    assembly = _is_assembly_recipe(extracted, body_len, thresholds)

    if not structured.ingredients:
        warnings.append(
            SoftValidationWarning(
                code="no_ingredients", message="structured_data has no ingredients"
            )
        )

    if not structured.steps and not assembly:
        warnings.append(
            SoftValidationWarning(code="no_steps", message="structured_data has no steps")
        )

    overall = extracted.confidence.overall
    if overall < thresholds.min_overall_confidence:
        warnings.append(
            SoftValidationWarning(
                code="low_overall_confidence",
                message=f"overall confidence {overall} < {thresholds.min_overall_confidence}",
            )
        )

    boundary = extracted.confidence.boundary
    if boundary < thresholds.min_boundary_confidence:
        warnings.append(
            SoftValidationWarning(
                code="low_boundary_confidence",
                message=f"boundary confidence {boundary} < {thresholds.min_boundary_confidence}",
            )
        )

    if body_len < thresholds.min_recipe_chars and not assembly:
        warnings.append(
            SoftValidationWarning(
                code="recipe_too_short",
                message=f"body_text length {body_len} < {thresholds.min_recipe_chars}",
            )
        )
    elif body_len > thresholds.max_recipe_chars:
        warnings.append(
            SoftValidationWarning(
                code="recipe_too_long",
                message=f"body_text length {body_len} > {thresholds.max_recipe_chars}",
            )
        )

    # Only meaningful when ingredients are present (else no_ingredients fires).
    if structured.ingredients:
        lowest = min(ing.confidence.normalization for ing in structured.ingredients)
        if lowest < thresholds.min_normalization_confidence:
            warnings.append(
                SoftValidationWarning(
                    code="low_normalization_confidence",
                    message=(
                        f"lowest ingredient normalization confidence {lowest} "
                        f"< {thresholds.min_normalization_confidence}"
                    ),
                )
            )

    return warnings
