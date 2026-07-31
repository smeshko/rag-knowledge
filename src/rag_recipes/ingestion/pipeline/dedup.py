"""Pure deduplication / candidate-scoring layer (doc 4 § Deduplication).

This module is **pure**: no DB session, no LLM provider, no storage, and no
``Settings`` singleton. It scores extracted recipe candidates and picks one
winner per recipe across a document's overlapping windows. The orchestration
layer (``ingestion/jobs.py``) persists every candidate via 9.3, wraps each
persisted row in a ``CandidateRef``, then calls ``select_best`` and prunes the
losers (persist-then-prune, DECISIONS #6).

``candidate_score`` is a confidence-weighted quality fraction in ``[0, 1]``
(DECISIONS #1), pinned here verbatim so the Epic 15 eval reports can reference
it:

    candidate_score =
        0.35 * confidence.overall
      + 0.25 * confidence.boundary
      + 0.15 * has_ingredients          # 1.0 if structured_data.ingredients else 0.0
      + 0.15 * has_steps                # 1.0 if structured_data.steps else 0.0
      + 0.10 * span_coverage            # min(1.0, |set(source_span_ids)| / |window_span_ids|)

The weights sum to 1.0, so every term is in ``[0, 1]`` and the score is bounded
``[0, 1]``. Missing/``None`` confidence fields coerce to ``0.0`` — defensive
against a soft-failed-but-persisted candidate; 9.3 hard validation already
rejects out-of-range confidence on every persisted row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_recipes.ingestion.pipeline.extraction import ExtractedRecipe

__all__ = [
    "CandidateRef",
    "compute_candidate_score",
    "select_best",
]

# candidate_score weights (DECISIONS #1). Sum to 1.0 → score bounded [0, 1].
_W_OVERALL = 0.35
_W_BOUNDARY = 0.25
_W_HAS_INGREDIENTS = 0.15
_W_HAS_STEPS = 0.15
_W_SPAN_COVERAGE = 0.10


def _as_float(value: float | None) -> float:
    """Coerce a possibly-missing confidence value to a float, defaulting to 0.0.

    9.3 hard validation rejects out-of-range confidence before persistence, so a
    persisted candidate's confidence is trustworthy; this guard only stops dedup
    from crashing on a soft-failed-but-persisted (or otherwise hand-built)
    candidate whose confidence field is ``None``.
    """
    return float(value) if value is not None else 0.0


def compute_candidate_score(
    extracted: ExtractedRecipe, *, window_span_ids: Sequence[str]
) -> float:
    """Return the ``candidate_score`` for ``extracted`` in ``[0, 1]`` (DECISIONS #1).

    Combines the model's self-reported overall/boundary confidence with
    structural-presence flags and source-span coverage per the weighted formula
    pinned in the module docstring. ``window_span_ids`` is the originating
    window's span ids, used to compute ``span_coverage``. The score is transient:
    nothing is persisted.
    """
    confidence = extracted.confidence
    overall = _as_float(confidence.overall)
    boundary = _as_float(confidence.boundary)

    structured = extracted.structured_data
    has_ingredients = 1.0 if structured.ingredients else 0.0
    has_steps = 1.0 if structured.steps else 0.0

    cited = len(set(extracted.source_span_ids))
    span_coverage = min(1.0, cited / max(1, len(window_span_ids)))

    return (
        _W_OVERALL * overall
        + _W_BOUNDARY * boundary
        + _W_HAS_INGREDIENTS * has_ingredients
        + _W_HAS_STEPS * has_steps
        + _W_SPAN_COVERAGE * span_coverage
    )


@dataclass(frozen=True)
class CandidateRef:
    """A persisted recipe candidate, reduced to what dedup needs (DECISIONS #3).

    Built *after* ``persist_knowledge_item`` so it carries the real
    ``KnowledgeItem.id``, the ``normalized_title`` 9.3 computed (grouped on, not
    re-derived — keeps dedup decoupled from 9.3's ``normalize_title``), the
    transient ``candidate_score``, and the originating ``extraction_run_id``.
    ``select_best`` returns these so the orchestrator can log each discard and
    delete losers by ``item_id``.
    """

    item_id: str
    normalized_title: str
    candidate_score: float
    extraction_run_id: str


def select_best(
    candidates: list[CandidateRef],
) -> tuple[list[CandidateRef], list[CandidateRef]]:
    """Pick one winner per ``normalized_title`` group; return ``(chosen, discarded)``.

    Groups by ``normalized_title`` (DECISIONS #2), keeps the single
    highest-scoring candidate per group, and returns every other candidate as
    discarded. Ties break deterministically on the lowest ``item_id`` so the same
    input always yields the same winner. Empty input returns ``([], [])``.
    """
    groups: dict[str, list[CandidateRef]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.normalized_title, []).append(candidate)

    chosen: list[CandidateRef] = []
    discarded: list[CandidateRef] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda c: (-c.candidate_score, c.item_id))
        chosen.append(ordered[0])
        discarded.extend(ordered[1:])
    return chosen, discarded
