"""Tests for the on-disk judge cache (Epic 15 Phase 15.2, Epic 20 Phase 20.1).

Every test constructs the cache with a ``tmp_path`` root — never the repo's
``evals/reports/.judge_cache/``. The key is the nine-part ``JudgeCacheKey``;
the significance matrix below proves every part participates in addressing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from evals.judge_cache import JudgeCache, JudgeCacheKey
from evals.judges import JudgeRating


def _rating(judge_version: str = "v1", model: str = "fake-model") -> JudgeRating:
    return JudgeRating(
        judge_name="summary_quality",
        judge_version=judge_version,
        rating="pass",
        critique="Captures the dish faithfully.",
        metadata={"provider": "fake", "model": model, "prompt_version": "summary_quality-v1"},
    )


def _key(**overrides: str) -> JudgeCacheKey:
    defaults = {
        "fixture_set": "smoke",
        "fixture_id": "bean-stew",
        "fixture_content_hash": "a" * 64,
        "extraction_prompt_version": "recipe-extraction-v1",
        "artifact_hash": "b" * 64,
        "judge_name": "summary_quality",
        "judge_version": "v1",
        "provider": "fake",
        "model": "fake-model",
    }
    defaults.update(overrides)
    return JudgeCacheKey(**defaults)


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = JudgeCache(root=tmp_path)
    rating = _rating()
    cache.put(rating, key=_key())
    assert cache.get(_key()) == rating


def test_get_misses_when_nothing_stored(tmp_path: Path) -> None:
    cache = JudgeCache(root=tmp_path)
    assert cache.get(_key()) is None


def test_every_key_part_is_significant(tmp_path: Path) -> None:
    # Changing any single one of the nine parts must miss — an insignificant
    # part would let a stale rating (other set, edited fixture, bumped prompt,
    # different artifact, other judge/version/provider/model) be replayed
    # silently.
    cache = JudgeCache(root=tmp_path)
    cache.put(_rating(), key=_key())
    assert cache.get(_key()) is not None
    changed = {
        "fixture_set": "smoke2",
        "fixture_id": "tomato-soup",
        "fixture_content_hash": "c" * 64,
        "extraction_prompt_version": "recipe-extraction-v2",
        "artifact_hash": "d" * 64,
        "judge_name": "boundary_correctness",
        "judge_version": "v2",
        "provider": "other-vendor",
        "model": "other-model",
    }
    for part, value in changed.items():
        assert cache.get(_key(**{part: value})) is None, part


def test_two_providers_at_the_same_model_name_do_not_share_a_rating(
    tmp_path: Path,
) -> None:
    """The Epic 23.3 case, spelled out rather than left to the matrix.

    Once the judge is separately configurable, two vendors can serve the same
    model name — an OpenAI-compatible endpoint advertises whatever id it likes.
    Without ``provider`` in the key, an Anthropic judge and a DeepSeek judge
    would silently share ratings, which is the exact cross-vendor contamination
    the split exists to remove.
    """
    cache = JudgeCache(root=tmp_path)
    cache.put(_rating(), key=_key(provider="anthropic", model="shared-model"))

    assert cache.get(_key(provider="deepseek", model="shared-model")) is None


def test_the_same_provider_still_hits_its_own_rating(tmp_path: Path) -> None:
    """The negative control for the test above.

    A cache that never hit at all would satisfy the isolation assertion while
    silently re-paying for every judge call.
    """
    cache = JudgeCache(root=tmp_path)
    rating = _rating()
    cache.put(rating, key=_key(provider="anthropic", model="shared-model"))

    assert cache.get(_key(provider="anthropic", model="shared-model")) == rating


def test_version_bump_invalidates(tmp_path: Path) -> None:
    # A prompt edit bumps the front-matter version (the prompts' contract
    # note), so the old rating must miss and the new version cache separately.
    cache = JudgeCache(root=tmp_path)
    cache.put(_rating(judge_version="v1"), key=_key())
    assert cache.get(_key(judge_version="v2")) is None
    cache.put(_rating(judge_version="v2"), key=_key(judge_version="v2"))
    assert cache.get(_key(judge_version="v1")) is not None
    assert cache.get(_key(judge_version="v2")) is not None


def test_entries_for_different_artifacts_coexist(tmp_path: Path) -> None:
    # DECISIONS #10: the artifact hash is a key *part*, so ratings for two
    # extractions of the same fixture land in two files — put never clobbers
    # the other artifact's entry (the property the read-guard design lacked).
    cache = JudgeCache(root=tmp_path)
    first = _rating()
    second = _rating().model_copy(update={"critique": "Second artifact."})
    cache.put(first, key=_key(artifact_hash="b" * 64))
    cache.put(second, key=_key(artifact_hash="e" * 64))
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert cache.get(_key(artifact_hash="b" * 64)) == first
    assert cache.get(_key(artifact_hash="e" * 64)) == second


def test_filename_stays_under_name_max_for_a_worst_case_key(tmp_path: Path) -> None:
    # Slugged parts truncate to 32 chars, the joined prefix to 120, plus the
    # 32-hex digest: a ~160-char ceiling comfortably inside NAME_MAX (255).
    cache = JudgeCache(root=tmp_path)
    key = _key(
        fixture_set="a-very-long-fixture-set-name-that-keeps-going-and-going",
        fixture_id="an-extremely-long-fixture-directory-name-with-many-words",
        fixture_content_hash="f" * 64,
        artifact_hash="0" * 64,
        model="anthropic/claude-sonnet-4-6-with-an-improbably-long-suffix",
    )
    path = cache.put(_rating(), key=key)
    assert len(path.name) < 255
    assert cache.get(key) is not None


def test_put_stores_under_the_key_model_not_rating_metadata(tmp_path: Path) -> None:
    # One source of truth per key part (DECISIONS #2): the old derivation from
    # rating.metadata["model"] is gone, so a mismatched metadata model must not
    # move the file.
    cache = JudgeCache(root=tmp_path)
    cache.put(_rating(model="something-else-entirely"), key=_key(model="fake-model"))
    assert cache.get(_key(model="fake-model")) is not None
    assert cache.get(_key(model="something-else-entirely")) is None


def test_key_is_frozen(tmp_path: Path) -> None:
    key = _key()
    try:
        key.model = "mutated"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("JudgeCacheKey must be frozen")
