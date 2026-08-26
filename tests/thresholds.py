"""One factory for the ``SoftValidationThresholds`` value object tests hand-build.

The eight-field literal was copied into seven test modules; every new bound
(the assembly trio, most recently) then had to be added in seven places — the
same drift ``evals/extraction.py`` fixed by delegating to
``thresholds_from_settings``. Tests override only what they exercise.
"""

from __future__ import annotations

from typing import Any

from rag_recipes.ingestion.validation import SoftValidationThresholds

# Mirrors the Settings defaults (config.py ``extraction_*``); tests that need a
# different bound say so explicitly.
_DEFAULTS: dict[str, Any] = {
    "min_overall_confidence": 0.5,
    "min_boundary_confidence": 0.5,
    "min_normalization_confidence": 0.5,
    "min_recipe_chars": 200,
    "max_recipe_chars": 20_000,
    "assembly_min_ingredients": 3,
    "assembly_max_ingredients": 12,
    "assembly_max_chars": 400,
}


def thresholds(**overrides: Any) -> SoftValidationThresholds:
    """``SoftValidationThresholds`` at the Settings defaults, with ``overrides``."""
    unknown = set(overrides) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown threshold field(s): {sorted(unknown)}")
    return SoftValidationThresholds(**{**_DEFAULTS, **overrides})
