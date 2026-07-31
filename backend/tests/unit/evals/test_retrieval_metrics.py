"""Tests for the pytrec_eval retrieval-metrics wrapper (Epic 16 Phase 16.1).

Every metric value is cross-checked against a tiny hand-computed qrels/run
pair (DECISIONS #4) — the library is not trusted blindly and outputs are not
merely snapshotted. Pure dict-in/dict-out: no DB, no HTTP, no providers.

Hand math for the oracle below (binary relevance, one relevant item each):

- ``q1`` ranks its relevant item 3rd → MRR ``1/3``; Recall@5 = Recall@10 = 1;
  NDCG@10 = ``(1/log2(4)) / (1/log2(2))`` = 0.5.
- ``q2`` ranks its relevant item 8th of 10 → MRR ``1/8``; Recall@5 = 0,
  Recall@10 = 1; NDCG@10 = ``1/log2(9)`` ≈ 0.3155.
- ``q3`` has an empty run → all four measures 0.0 (and still counts in the
  aggregate denominator — DECISIONS #5).
"""

from __future__ import annotations

import math

import pytest
from evals.metrics.retrieval import (
    RETRIEVAL_METRICS,
    build_run_dict,
    compute_retrieval_metrics,
)

_Q2_NDCG = 1.0 / math.log2(9)

QRELS = {
    "q1": {"item_a": 1},
    "q2": {"item_b": 1},
    "q3": {"item_c": 1},
}

RUN = {
    # item_a at rank 3.
    "q1": build_run_dict(["item_x", "item_y", "item_a"]),
    # item_b at rank 8 of a 10-long run.
    "q2": build_run_dict(
        ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "item_b", "r9", "r10"]
    ),
    # Empty run: pytrec_eval returns this query with all-zero measures (it
    # omits only an entirely absent key); either way it must score 0.0.
    "q3": {},
}


def test_measure_names_are_the_literal_trec_eval_strings() -> None:
    assert {"ndcg_cut_10", "recall_5", "recall_10", "recip_rank"} == RETRIEVAL_METRICS


def test_per_query_values_match_hand_computed_oracle() -> None:
    metrics = compute_retrieval_metrics(QRELS, RUN)
    q1 = metrics.per_query["q1"]
    assert q1["recip_rank"] == pytest.approx(1 / 3)
    assert q1["recall_5"] == pytest.approx(1.0)
    assert q1["recall_10"] == pytest.approx(1.0)
    assert q1["ndcg_cut_10"] == pytest.approx(0.5)

    q2 = metrics.per_query["q2"]
    assert q2["recip_rank"] == pytest.approx(1 / 8)
    assert q2["recall_5"] == pytest.approx(0.0)
    assert q2["recall_10"] == pytest.approx(1.0)
    assert q2["ndcg_cut_10"] == pytest.approx(_Q2_NDCG)


def test_empty_run_query_scores_zero_everywhere() -> None:
    metrics = compute_retrieval_metrics(QRELS, RUN)
    assert metrics.per_query["q3"] == {measure: 0.0 for measure in RETRIEVAL_METRICS}


def test_aggregate_divides_by_qrels_query_count() -> None:
    # The empty-run q3 contributes 0.0 to every mean: denominator is the qrels
    # key set (3), never the count of queries pytrec_eval happened to score.
    metrics = compute_retrieval_metrics(QRELS, RUN)
    assert metrics.aggregate["ndcg_cut_10"] == pytest.approx((0.5 + _Q2_NDCG + 0.0) / 3)
    assert metrics.aggregate["recall_5"] == pytest.approx(1 / 3)
    assert metrics.aggregate["recall_10"] == pytest.approx(2 / 3)
    assert metrics.aggregate["recip_rank"] == pytest.approx((1 / 3 + 1 / 8 + 0.0) / 3)


def test_absent_run_key_also_counts_as_zero_in_aggregate() -> None:
    # A query whose key is entirely missing from the run dict is omitted from
    # pytrec_eval's output — the qrels-denominator aggregation must still count
    # it as 0.0 (DECISIONS #5 covers both the empty-{} and absent-key cases).
    run = {"q1": RUN["q1"], "q2": RUN["q2"]}  # q3 key absent entirely
    metrics = compute_retrieval_metrics(QRELS, run)
    assert metrics.per_query["q3"] == {measure: 0.0 for measure in RETRIEVAL_METRICS}
    assert metrics.aggregate["recall_10"] == pytest.approx(2 / 3)


def test_int_relevance_values_are_accepted_and_coerced() -> None:
    qrels = {"q1": {"item_a": True}}  # bool is an int subtype; must coerce cleanly
    metrics = compute_retrieval_metrics(qrels, {"q1": build_run_dict(["item_a"])})
    assert metrics.per_query["q1"]["recip_rank"] == pytest.approx(1.0)


def test_build_run_dict_preserves_order_with_strictly_descending_scores() -> None:
    run = build_run_dict(["a", "b", "c"])
    assert run == {"a": 3.0, "b": 2.0, "c": 1.0}
    scores = list(run.values())
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)  # distinct positions never tie


def test_build_run_dict_empty_input() -> None:
    assert build_run_dict([]) == {}
