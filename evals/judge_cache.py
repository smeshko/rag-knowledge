"""On-disk cache for judge ratings (Epic 15 Phase 15.2, Epic 20 Phase 20.1).

Storage-only — this module knows nothing about LLMs, prompts, or the extraction
pipeline (it must never import ``rag_recipes``; callers pass every key part in
as a plain string). One JSON file per :class:`JudgeCacheKey` under
``evals/reports/.judge_cache/`` (covered by the existing ``evals/reports/*``
gitignore rule).

The nine-part key is ``(fixture_set, fixture_id, fixture_content_hash,
extraction_prompt_version, artifact_hash, judge_name, judge_version, provider,
model)``:

- ``fixture_set`` + ``fixture_id`` — same-named fixtures in different sets can
  never collide.
- ``fixture_content_hash`` — ``RecipeFixture.content_hash()``; a
  ``source.md``/``expected.json`` edit invalidates the rating.
- ``extraction_prompt_version`` — the extraction ``PROMPT_VERSION`` the run
  used; a prompt bump invalidates the rating.
- ``artifact_hash`` — hash of the exact serialized artifact that was judged
  (DECISIONS #10): a rating is replayed only for the byte-identical artifact it
  was produced from, and ratings for different artifacts of the same fixture
  coexist as separate files instead of clobbering each other.
- ``judge_name``/``judge_version``/``provider``/``model`` — the judge identity;
  a prompt edit MUST bump the prompt's front-matter version (the contract note
  in each prompt file). ``provider`` joined the key in Epic 23.3, when the judge
  became separately configurable: two providers can serve the same model name —
  an OpenAI-compatible endpoint advertises whatever model id it likes — so
  ``(judge_name, judge_version, model)`` stopped identifying who produced a
  rating. Without this part, an Anthropic judge and a DeepSeek judge would
  silently share ratings, which is the precise cross-vendor contamination the
  split exists to remove.

File naming: a bounded readable slug prefix (each part truncated to 32 chars,
the joined prefix to 120) plus a 32-hex digest of the full key tuple — a hard
ceiling of ~160 characters, comfortably inside ``NAME_MAX`` (255). Collisions
between distinct keys are astronomically unlikely, not impossible. The cache
root is injectable so tests point it at ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

from evals.judges import JudgeRating

__all__ = ["CACHE_ROOT", "JudgeCache", "JudgeCacheKey", "slugify_key_part"]

CACHE_ROOT = Path(__file__).resolve().parents[1] / "evals" / "reports" / ".judge_cache"


def slugify_key_part(value: str) -> str:
    """Map an arbitrary string to a filesystem-safe slug (``[A-Za-z0-9._-]``).

    Public because the judge-cache filenames and the alignment-record composite
    ids must slug identically — two independent slug functions would drift.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


@dataclasses.dataclass(frozen=True)
class JudgeCacheKey:
    """The full nine-part cache key; every part is significant.

    All parts are caller-supplied opaque strings — this module never computes a
    hash or reads a version itself (storage-only contract). The single
    construction site is ``evals.extraction.build_judge_cache_key``.

    Adding ``provider`` in Epic 23.3 changed every cache path (``_path`` hashes
    the whole tuple), orphaning previously cached ratings. Deliberate: a cache
    that can serve one vendor's rating for another vendor's request is worse
    than a cold one, and ratings are cheap to regenerate.
    """

    fixture_set: str
    fixture_id: str
    fixture_content_hash: str
    extraction_prompt_version: str
    artifact_hash: str
    judge_name: str
    judge_version: str
    provider: str
    model: str


class JudgeCache:
    """Filesystem cache of ``JudgeRating``s; a hit skips the LLM call entirely."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = CACHE_ROOT if root is None else root

    def _path(self, key: JudgeCacheKey) -> Path:
        # astuple, not the dataclass itself: json.dumps cannot serialize it.
        parts = dataclasses.astuple(key)
        digest = hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()[:32]
        prefix = "__".join(slugify_key_part(part)[:32] for part in parts)[:120]
        return self._root / f"{prefix}--{digest}.json"

    def get(self, key: JudgeCacheKey) -> JudgeRating | None:
        """Return the cached rating for the key, or ``None`` on a miss."""
        path = self._path(key)
        if not path.is_file():
            return None
        return JudgeRating.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, rating: JudgeRating, *, key: JudgeCacheKey) -> Path:
        """Store a rating under ``key`` — every key part has exactly one source.

        ``key.model`` addresses the file even if ``rating.metadata`` disagrees;
        the old ``rating.metadata["model"]`` derivation is gone (DECISIONS #2).
        """
        path = self._path(key)
        self._root.mkdir(parents=True, exist_ok=True)
        path.write_text(rating.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path
