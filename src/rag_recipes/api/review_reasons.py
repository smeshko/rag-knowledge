"""Projection of persisted warning codes into review reasons (Epic 21.1, D3).

``structured_data["warnings"]`` holds soft-validation *codes only* (persist
drops the LLM's self-reported warnings and the threshold-bearing messages), so
the API maps each code to a stable human label at projection time. Pure module:
no session, no settings — trivially unit-testable.

Consumed by ``GET /knowledge-items/{id}`` (the review surface: search cannot
return ``needs_review`` items — they are never chunked, and retrieval floors at
``ready``).
"""

from __future__ import annotations

from typing import Any

from rag_recipes.api.schemas.knowledge_items import ReviewReason

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


def build_review_reasons(
    status: str, structured_data: dict[str, Any]
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
    """
    if status != "needs_review":
        return []
    warnings = structured_data.get("warnings")
    if not isinstance(warnings, list):
        return []
    reasons: list[ReviewReason] = []
    for warning in warnings:
        if isinstance(warning, str) and warning in SOFT_WARNING_MESSAGES:
            reasons.append(
                ReviewReason(code=warning, message=SOFT_WARNING_MESSAGES[warning])
            )
        else:
            reasons.append(ReviewReason(code="llm_warning", message=str(warning)))
    return reasons
