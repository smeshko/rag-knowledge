"""Unit tests for the optional rerank step in the retrieval facade (Epic 18.2).

``_apply_rerank`` is pure (no I/O) — tested directly with synthetic ``MergedChunk``s
and reranker results, then ``group_by_item`` is run to assert the *final* item order
follows reranker rank (with the supporting bonus suppressed). ``_maybe_rerank``'s
DB seam (``_fetch_chunk_texts``) is monkeypatched so the orchestration + degradation
+ fallback logic run in memory against a ``FakeRerankerProvider``.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.providers.errors import RerankerTechnicalError
from rag_recipes.providers.reranker.base import RerankerProvider
from rag_recipes.providers.reranker.fake import FakeRerankerProvider
from rag_recipes.providers.reranker.types import RerankResult
from rag_recipes.retrieval import search as search_module
from rag_recipes.retrieval.group import group_by_item
from rag_recipes.retrieval.search import _apply_rerank, _maybe_rerank
from rag_recipes.retrieval.types import MergedChunk
from rag_recipes.storage.enums import ChunkType


def _chunk(
    chunk_id: str,
    *,
    item: str,
    score: float,
    chunk_type: ChunkType = ChunkType.RECIPE_SUMMARY,
) -> MergedChunk:
    return MergedChunk(
        chunk_id=chunk_id,
        knowledge_item_id=item,
        chunk_type=chunk_type,
        score=score,
        sources=["keyword"],
    )


def _result(chunk_id: str, *, rank: int) -> RerankResult:
    return RerankResult(chunk_id=chunk_id, relevance_score=1.0 / rank, rank=rank)


def _ordered_items(merged: list[MergedChunk], *, rerank_applied: bool) -> list[str]:
    bonus = 0.0 if rerank_applied else 0.05
    cap = 0.0 if rerank_applied else 0.15
    grouped = group_by_item(merged, supporting_bonus=bonus, supporting_bonus_cap=cap)
    return [g.knowledge_item_id for g in grouped]


# --- _apply_rerank (pure) -------------------------------------------------------


def test_apply_rerank_final_order_follows_reranker_rank() -> None:
    # RRF ranks item_a's chunk first; the reranker ranks item_b's chunk first.
    merged = [_chunk("a1", item="item_a", score=0.9), _chunk("b1", item="item_b", score=0.1)]
    top = merged[:50]
    results = [_result("b1", rank=1), _result("a1", rank=2)]
    out, applied = _apply_rerank(merged, top, results)
    assert applied is True
    assert _ordered_items(out, rerank_applied=True) == ["item_b", "item_a"]


def test_apply_rerank_tail_higher_rrf_stays_behind_reranked_head() -> None:
    # Only the first chunk is reranked (top_n=1); the tail chunk has a much higher
    # RRF score but must still fall behind the reranked head (dynamic base dominates).
    merged = [_chunk("a1", item="item_a", score=0.1), _chunk("b1", item="item_b", score=99.0)]
    top = merged[:1]  # only a1 is a candidate
    results = [_result("a1", rank=1)]
    out, applied = _apply_rerank(merged, top, results)
    assert applied is True
    # a1's effective score = max(0.1, 99.0)+1 = 100 > b1's RRF 99 → item_a wins.
    assert _ordered_items(out, rerank_applied=True) == ["item_a", "item_b"]


def test_apply_rerank_equal_original_scores_use_reranker_order_not_id() -> None:
    # Two items with equal RRF scores: without rerank, group_by_item tiebreaks by
    # knowledge_item_id (item_a first). The reranker ranks item_b first → it wins.
    merged = [_chunk("a1", item="item_a", score=5.0), _chunk("b1", item="item_b", score=5.0)]
    assert _ordered_items(merged, rerank_applied=False) == ["item_a", "item_b"]
    out, applied = _apply_rerank(
        merged, merged[:50], [_result("b1", rank=1), _result("a1", rank=2)]
    )
    assert applied is True
    assert _ordered_items(out, rerank_applied=True) == ["item_b", "item_a"]


def test_apply_rerank_beats_supporting_bonus() -> None:
    # item_y has two distinct chunk types (a supporting bonus that would let it win
    # under normal grouping); item_x has one chunk the reranker ranks #1. With the
    # bonus suppressed on the reranked path, the reranker's top item wins.
    merged = [
        _chunk("x1", item="item_x", score=1.0, chunk_type=ChunkType.RECIPE_TITLE),
        _chunk("y1", item="item_y", score=1.0, chunk_type=ChunkType.RECIPE_TITLE),
        _chunk("y2", item="item_y", score=0.9, chunk_type=ChunkType.RECIPE_STEPS),
    ]
    # Without rerank, item_y's bonus (2 distinct types) lifts it above item_x.
    assert _ordered_items(merged, rerank_applied=False)[0] == "item_y"
    out, applied = _apply_rerank(
        merged, merged[:50], [_result("x1", rank=1), _result("y1", rank=2), _result("y2", rank=3)]
    )
    assert applied is True
    assert _ordered_items(out, rerank_applied=True)[0] == "item_x"  # reranker wins


def test_apply_rerank_all_invalid_is_noop_baseline() -> None:
    merged = [_chunk("a1", item="item_a", score=0.9), _chunk("b1", item="item_b", score=0.1)]
    original = [(c.chunk_id, c.score) for c in merged]
    out, applied = _apply_rerank(merged, merged[:50], [_result("unknown_x", rank=1)])
    assert applied is False
    assert [(c.chunk_id, c.score) for c in out] == original  # RRF scores untouched


def test_apply_rerank_ignores_unknown_and_dedupes_dropping_no_chunk() -> None:
    merged = [_chunk("a1", item="item_a", score=0.5), _chunk("b1", item="item_b", score=0.4)]
    results = [
        _result("b1", rank=1),
        _result("__unknown__", rank=2),  # ignored
        _result("b1", rank=3),  # duplicate → ignored (keep-first)
        _result("a1", rank=4),
    ]
    out, applied = _apply_rerank(merged, merged[:50], results)
    assert applied is True
    assert {c.chunk_id for c in out} == {"a1", "b1"}  # no input chunk dropped
    assert _ordered_items(out, rerank_applied=True) == ["item_b", "item_a"]


def test_apply_rerank_omitted_chunk_keeps_rrf_behind_head() -> None:
    merged = [
        _chunk("a1", item="item_a", score=0.5),
        _chunk("b1", item="item_b", score=0.4),
    ]
    # The reranker only returns a1; b1 is omitted → keeps its RRF score, behind a1.
    out, applied = _apply_rerank(merged, merged[:50], [_result("a1", rank=1)])
    assert applied is True
    b1 = next(c for c in out if c.chunk_id == "b1")
    assert b1.score == 0.4  # RRF preserved
    assert _ordered_items(out, rerank_applied=True) == ["item_a", "item_b"]


# --- _maybe_rerank (DB seam monkeypatched) --------------------------------------


def _settings(**over: Any) -> Any:
    class _S:
        reranking_enabled = True
        rerank_top_n = 50
        rerank_max_chars_per_candidate = 2000

    s = _S()
    for key, value in over.items():
        setattr(s, key, value)
    return s


def _patch_texts(monkeypatch: pytest.MonkeyPatch, texts: dict[str, str]) -> None:
    async def fake_fetch(session: Any, chunk_ids: list[str]) -> dict[str, str]:
        return {cid: texts.get(cid, "") for cid in chunk_ids}

    monkeypatch.setattr(search_module, "_fetch_chunk_texts", fake_fetch)


@pytest.mark.asyncio
async def test_maybe_rerank_truncates_candidate_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_texts(monkeypatch, {"a1": "x" * 5000})
    fake = FakeRerankerProvider()
    merged = [_chunk("a1", item="item_a", score=0.5)]
    await _maybe_rerank(
        None, "q", merged, reranker=fake, settings=_settings(rerank_max_chars_per_candidate=100)
    )
    assert len(fake.calls[0].candidates[0].text) == 100  # truncated to the cap


@pytest.mark.asyncio
async def test_maybe_rerank_technical_error_falls_back_to_rrf(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_texts(monkeypatch, {"a1": "t"})

    class _Boom(RerankerProvider):
        provider = "boom"

        async def rerank(
            self, query: str, candidates: Any, *, top_n: int, trace_context: Any = None
        ) -> Any:
            raise RerankerTechnicalError("kaboom")

    merged = [_chunk("a1", item="item_a", score=0.5), _chunk("b1", item="item_b", score=0.9)]
    original = [(c.chunk_id, c.score) for c in merged]
    with caplog.at_level("WARNING", logger="rag_recipes.retrieval.search"):
        out, applied = await _maybe_rerank(
            None, "q", merged, reranker=_Boom(), settings=_settings()
        )
    assert applied is False
    assert [(c.chunk_id, c.score) for c in out] == original  # RRF intact
    assert any("falling back to RRF order" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_maybe_rerank_empty_merged_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRerankerProvider()
    out, applied = await _maybe_rerank(None, "q", [], reranker=fake, settings=_settings())
    assert applied is False
    assert out == []
    assert fake.calls == ()  # never called


@pytest.mark.asyncio
async def test_maybe_rerank_reorders_via_fake_score_map(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_texts(monkeypatch, {"a1": "ta", "b1": "tb"})
    # Fake ranks b1 above a1.
    fake = FakeRerankerProvider(scores_by_chunk_id={"a1": 0.1, "b1": 0.9})
    merged = [_chunk("a1", item="item_a", score=0.9), _chunk("b1", item="item_b", score=0.1)]
    out, applied = await _maybe_rerank(None, "q", merged, reranker=fake, settings=_settings())
    assert applied is True
    assert _ordered_items(out, rerank_applied=True) == ["item_b", "item_a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unknown", "duplicate", "missing", "out_of_range_rank"])
async def test_maybe_rerank_emit_modes_drop_no_input_chunk(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    # Each malformed reranker-output mode degrades safely: no input chunk is dropped.
    _patch_texts(monkeypatch, {"a1": "ta", "b1": "tb", "c1": "tc"})
    fake = FakeRerankerProvider(emit=mode)
    merged = [
        _chunk("a1", item="item_a", score=0.9),
        _chunk("b1", item="item_b", score=0.5),
        _chunk("c1", item="item_c", score=0.1),
    ]
    out, _applied = await _maybe_rerank(None, "q", merged, reranker=fake, settings=_settings())
    assert {c.chunk_id for c in out} == {"a1", "b1", "c1"}  # every input chunk survives
