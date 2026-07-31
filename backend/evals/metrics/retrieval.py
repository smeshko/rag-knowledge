"""Pure ``pytrec_eval`` wrapper for retrieval metrics (Epic 16, doc 12 § 7).

Purity contract: this module imports only ``pytrec_eval`` and the stdlib — no
I/O, no DB, no HTTP, no ``Settings``. Callers assemble the qrels/run dicts and
serialize the results.

Aggregation rule (DECISIONS #5): aggregates are means over the **qrels**
query-id set, so a query that returned nothing still counts as ``0.0`` for
every measure. ``pytrec_eval`` returns a query present in the run with an
empty ``{}`` map with all measures ``0.0`` and omits only a query whose key is
entirely absent — the qrels denominator is correct for both cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytrec_eval

__all__ = [
    "RETRIEVAL_METRICS",
    "RetrievalMetrics",
    "build_run_dict",
    "compute_retrieval_metrics",
]

#: The literal ``trec_eval`` measure names (DECISIONS #2): NDCG@10 (headline),
#: Recall@5, Recall@10, and reciprocal rank (MRR when averaged).
RETRIEVAL_METRICS = {"ndcg_cut_10", "recall_5", "recall_10", "recip_rank"}


@dataclass(frozen=True)
class RetrievalMetrics:
    """Per-query and aggregate values for the four retrieval measures."""

    per_query: dict[str, dict[str, float]]
    aggregate: dict[str, float]


def build_run_dict(ranked_item_ids: list[str]) -> dict[str, float]:
    """Turn one query's ranked id list into pytrec_eval's ``{doc_id: score}``.

    Scores are rank-based and strictly descending (``score = len - position``,
    DECISIONS #3), so the *ranking* — not any raw fused item score — is what
    the metrics evaluate, and distinct positions can never tie.
    """
    total = len(ranked_item_ids)
    return {
        item_id: float(total - position) for position, item_id in enumerate(ranked_item_ids)
    }


def compute_retrieval_metrics(
    qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]
) -> RetrievalMetrics:
    """Evaluate ``run`` against ``qrels`` for :data:`RETRIEVAL_METRICS`.

    ``qrels`` relevance values are coerced to ``int`` (pytrec_eval rejects
    non-int relevance); binary ``1`` today and graded ``0/1/2/3`` later both
    pass through unchanged.
    """
    coerced = {
        query_id: {item_id: int(relevance) for item_id, relevance in judged.items()}
        for query_id, judged in qrels.items()
    }
    evaluator = pytrec_eval.RelevanceEvaluator(coerced, RETRIEVAL_METRICS)
    evaluated = evaluator.evaluate(run)
    per_query = {
        query_id: {
            measure: float(evaluated.get(query_id, {}).get(measure, 0.0))
            for measure in sorted(RETRIEVAL_METRICS)
        }
        for query_id in coerced
    }
    query_count = len(per_query)
    aggregate = {
        measure: (
            sum(measures[measure] for measures in per_query.values()) / query_count
            if query_count
            else 0.0
        )
        for measure in sorted(RETRIEVAL_METRICS)
    }
    return RetrievalMetrics(per_query=per_query, aggregate=aggregate)
