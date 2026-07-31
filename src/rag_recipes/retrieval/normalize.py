"""Query normalization (doc 7 § 2).

Three steps only — trim, collapse internal whitespace, lowercase — preserving the
verbatim original for display. Punctuation is deliberately NOT stripped: tokenization
is ``plainto_tsquery``'s job at the SQL layer, so ``"WHITE beans!!"`` normalizes to
``"white beans!!"`` and the FTS query reduces it to the lexemes ``white & bean``.
Over-normalizing here would risk changing query intent (doc 7 § 2).
"""

from __future__ import annotations

from rag_recipes.retrieval.types import NormalizedQuery


def normalize_query(raw: str) -> NormalizedQuery:
    """Return a ``NormalizedQuery`` with the verbatim ``original`` and a trimmed,
    single-spaced, lowercased ``keyword``. Pure function, no I/O; idempotent on an
    already-normalized input."""
    keyword = " ".join(raw.split()).lower()
    return NormalizedQuery(original=raw, keyword=keyword)
