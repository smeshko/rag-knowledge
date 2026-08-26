"""Projection of persisted warning codes into review reasons (Epic 21.1, D3).

``structured_data["warnings"]`` holds soft-validation *codes only* (persist
drops the LLM's self-reported warnings and the threshold-bearing messages), so
the API maps each code to a stable human label at projection time. Pure module:
no session, no settings — trivially unit-testable.

Reviewer aids are layered on top when the caller passes the item's persisted
``confidence`` and the current ``thresholds``: the observed score behind a
confidence reason, the bound it is measured against, and — for the ingredient
normalization rule — which rows are below it, so the UI can point at them.

Consumed by ``GET /knowledge-items/{id}`` (the review surface: search cannot
return ``needs_review`` items — they are never chunked, and retrieval floors at
``ready``).
"""

from __future__ import annotations

from typing import Any, Literal

from rag_recipes.api.schemas.knowledge_items import ReviewReason, ReviewThresholds
from rag_recipes.ingestion.validation import SoftValidationThresholds

# Human labels for the canonical `validate_soft` warning codes. The original
# threshold-bearing messages are not persisted (thresholds may have changed
# since ingestion), so these are stable, generic presentation labels.
# Coverage is drift-guarded by tests/unit/api/test_review_reasons.py.
SOFT_WARNING_MESSAGES: dict[str, str] = {
    "no_ingredients": "No ingredients were extracted.",
    "no_steps": "No preparation steps were extracted.",
    "low_overall_confidence": "Overall extraction confidence was below the threshold.",
    "low_boundary_confidence": "Recipe boundary confidence was below the threshold.",
    "recipe_too_short": "The recipe body is shorter than expected.",
    "recipe_too_long": "The recipe body is longer than expected.",
    "low_normalization_confidence": (
        "An ingredient normalization confidence was below the threshold."
    ),
}


def _score(container: Any, key: str) -> float | None:
    """A finite number under ``key`` of a mapping, else ``None`` (JSONB is opaque)."""
    if not isinstance(container, dict):
        return None
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ingredient_normalizations(structured_data: dict[str, Any]) -> list[tuple[int, float]]:
    """``(position, normalization)`` per ingredient that carries a numeric score.

    ``position`` falls back to the row's index when absent, which is also what
    the FE's row order resolves to for an unpositioned list.
    """
    ingredients = structured_data.get("ingredients")
    if not isinstance(ingredients, list):
        return []
    scored: list[tuple[int, float]] = []
    for index, ing in enumerate(ingredients):
        if not isinstance(ing, dict):
            continue
        normalization = _score(ing.get("confidence"), "normalization")
        if normalization is None:
            continue
        position = ing.get("position")
        if isinstance(position, bool) or not isinstance(position, int):
            position = index
        scored.append((position, normalization))
    return scored


def _normalization_detail(
    structured_data: dict[str, Any], threshold: float | None
) -> tuple[float | None, list[int] | None]:
    """Lowest normalization score and the positions at/below the bound.

    The rule fired on the minimum, so the minimum row is always named even when
    the threshold has since moved above it (or is unknown): a reason that
    points at nothing is what this exists to fix.
    """
    scored = _ingredient_normalizations(structured_data)
    if not scored:
        return None, None
    lowest = min(score for _, score in scored)
    below = [pos for pos, score in scored if score < threshold] if threshold is not None else []
    if not below:
        below = [pos for pos, score in scored if score == lowest]
    return lowest, below


ThresholdSource = Literal["recorded", "current"]


def thresholds_for_item(
    structured_data: dict[str, Any] | None,
    *,
    current: SoftValidationThresholds,
) -> tuple[SoftValidationThresholds, ThresholdSource]:
    """The bounds this item's warnings were judged against, and where they came from.

    ``recorded`` — read back from ``structured_data["validation_thresholds"]``,
    the snapshot persist/edit wrote when the warnings were derived. ``current`` —
    the row predates the snapshot (or it is malformed), so the live Settings
    stand in; a reason's ``threshold`` may then differ from the one that fired.
    """
    recorded = SoftValidationThresholds.from_record(
        (structured_data or {}).get("validation_thresholds")
    )
    if recorded is not None:
        return recorded, "recorded"
    return current, "current"


def build_review_thresholds(
    status: str,
    thresholds: SoftValidationThresholds | None,
    source: ThresholdSource = "current",
) -> ReviewThresholds | None:
    """The bounds a reviewer is judging against; only for ``needs_review``."""
    if status != "needs_review" or thresholds is None:
        return None
    return ReviewThresholds(
        overall=thresholds.min_overall_confidence,
        boundary=thresholds.min_boundary_confidence,
        normalization=thresholds.min_normalization_confidence,
        source=source,
    )


def build_review_reasons(
    status: str,
    structured_data: dict[str, Any],
    confidence: dict[str, Any] | None = None,
    thresholds: SoftValidationThresholds | None = None,
) -> list[ReviewReason]:
    """Project persisted warning codes into ``{code, message}`` reasons (D3).

    Populated only for ``needs_review`` items; ``[]`` otherwise. Known codes get
    their static label; anything else — an unknown string or (defensively, the
    JSONB is opaque) a non-string element — is enveloped as ``llm_warning`` with
    the raw value as the message, so free prose never lands in the ``code``
    position and a hand-seeded dict can never 500 the endpoint.

    The ``warnings`` value itself is guarded the same way: only a list is
    iterated. A degraded row holding a scalar would otherwise raise ``TypeError``
    (a 500 on the item's own detail/audit endpoint) and a bare string or mapping
    would iterate per character / per key into nonsense reasons. Anything that is
    not a list yields ``[]`` — no reasons rather than invented ones.

    ``confidence`` (the item's persisted top-level scores) and ``thresholds``
    (the current bounds) are optional: without them the reasons carry only
    ``code``/``message``. With them, the three confidence codes also carry the
    observed ``value`` and ``threshold``, and ``low_normalization_confidence``
    names the ``ingredient_positions`` to look at. Every lookup tolerates a
    malformed payload — a missing or non-numeric score just leaves the aid
    ``None``.
    """
    if status != "needs_review":
        return []
    warnings = structured_data.get("warnings")
    if not isinstance(warnings, list):
        return []
    reasons: list[ReviewReason] = []
    for warning in warnings:
        if not (isinstance(warning, str) and warning in SOFT_WARNING_MESSAGES):
            reasons.append(ReviewReason(code="llm_warning", message=str(warning)))
            continue
        reason = ReviewReason(code=warning, message=SOFT_WARNING_MESSAGES[warning])
        if warning == "low_overall_confidence":
            reason.value = _score(confidence, "overall")
            reason.threshold = thresholds.min_overall_confidence if thresholds else None
        elif warning == "low_boundary_confidence":
            reason.value = _score(confidence, "boundary")
            reason.threshold = thresholds.min_boundary_confidence if thresholds else None
        elif warning == "low_normalization_confidence":
            reason.threshold = thresholds.min_normalization_confidence if thresholds else None
            reason.value, reason.ingredient_positions = _normalization_detail(
                structured_data, reason.threshold
            )
        reasons.append(reason)
    return reasons
