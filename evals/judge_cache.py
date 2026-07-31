"""On-disk cache for judge ratings (Epic 15 Phase 15.2, DECISIONS #3).

Storage-only — this module knows nothing about LLMs or prompts. One JSON file
per key under ``evals/reports/.judge_cache/`` (covered by the existing
``evals/reports/*`` gitignore rule), keyed by the epic's tuple
``(fixture_id, judge_name, judge_version, model)`` where ``fixture_id`` is
``RecipeFixture.name`` and ``model`` is the provider's ``default_model``.

Invalidation rides the key: a prompt edit MUST bump the prompt's front-matter
version (the contract note in each prompt file), which changes ``judge_version``
and therefore misses the cache. The cache root is injectable so tests point it
at ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from evals.judges import JudgeRating

__all__ = ["CACHE_ROOT", "JudgeCache"]

CACHE_ROOT = Path(__file__).resolve().parents[1] / "evals" / "reports" / ".judge_cache"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


class JudgeCache:
    """Filesystem cache of ``JudgeRating``s; a hit skips the LLM call entirely."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = CACHE_ROOT if root is None else root

    def _path(self, fixture_id: str, judge_name: str, judge_version: str, model: str) -> Path:
        key = (fixture_id, judge_name, judge_version, model)
        # Readable slug prefix + exact-tuple hash suffix, so two keys that slug
        # identically can never share a file.
        digest = hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()[:12]
        name = "__".join(_slug(part) for part in key)
        return self._root / f"{name}--{digest}.json"

    def get(
        self, fixture_id: str, judge_name: str, judge_version: str, model: str
    ) -> JudgeRating | None:
        """Return the cached rating for the key, or ``None`` on a miss."""
        path = self._path(fixture_id, judge_name, judge_version, model)
        if not path.is_file():
            return None
        return JudgeRating.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, rating: JudgeRating, *, fixture_id: str) -> Path:
        """Store a rating under its key; the model comes from ``rating.metadata``."""
        model = rating.metadata["model"]
        path = self._path(fixture_id, rating.judge_name, rating.judge_version, model)
        self._root.mkdir(parents=True, exist_ok=True)
        path.write_text(rating.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path
