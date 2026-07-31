"""Tests for the on-disk judge cache (Epic 15 Phase 15.2).

Every test constructs the cache with a ``tmp_path`` root — never the repo's
``evals/reports/.judge_cache/``.
"""

from __future__ import annotations

from pathlib import Path

from evals.judge_cache import JudgeCache
from evals.judges import JudgeRating


def _rating(judge_version: str = "v1", model: str = "fake-model") -> JudgeRating:
    return JudgeRating(
        judge_name="summary_quality",
        judge_version=judge_version,
        rating="pass",
        critique="Captures the dish faithfully.",
        metadata={"provider": "fake", "model": model, "prompt_version": "summary_quality-v1"},
    )


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = JudgeCache(root=tmp_path)
    rating = _rating()
    cache.put(rating, fixture_id="bean-stew")
    assert cache.get("bean-stew", "summary_quality", "v1", "fake-model") == rating


def test_get_misses_when_nothing_stored(tmp_path: Path) -> None:
    cache = JudgeCache(root=tmp_path)
    assert cache.get("bean-stew", "summary_quality", "v1", "fake-model") is None


def test_each_key_element_is_significant(tmp_path: Path) -> None:
    cache = JudgeCache(root=tmp_path)
    cache.put(_rating(), fixture_id="bean-stew")
    assert cache.get("tomato-soup", "summary_quality", "v1", "fake-model") is None
    assert cache.get("bean-stew", "boundary_correctness", "v1", "fake-model") is None
    assert cache.get("bean-stew", "summary_quality", "v2", "fake-model") is None
    assert cache.get("bean-stew", "summary_quality", "v1", "other-model") is None


def test_version_bump_invalidates(tmp_path: Path) -> None:
    # A prompt edit bumps the front-matter version (the prompts' contract
    # note), so the old rating must miss and the new version cache separately.
    cache = JudgeCache(root=tmp_path)
    cache.put(_rating(judge_version="v1"), fixture_id="bean-stew")
    assert cache.get("bean-stew", "summary_quality", "v2", "fake-model") is None
    cache.put(_rating(judge_version="v2"), fixture_id="bean-stew")
    assert cache.get("bean-stew", "summary_quality", "v1", "fake-model") is not None
    assert cache.get("bean-stew", "summary_quality", "v2", "fake-model") is not None
