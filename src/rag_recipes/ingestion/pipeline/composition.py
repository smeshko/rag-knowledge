"""Shared text composition over a ``recipe.v1`` ``structured_data`` payload.

Pure module: no session, no provider, no ``Settings``. It exists so the two
places that turn a persisted knowledge item into text — ``pipeline/chunking``
when it builds the ``recipe_ingredients`` / ``recipe_steps`` / ``recipe_full``
chunks, and ``ingestion/editing`` when it rebuilds ``body_text`` after a
reviewer edit — cannot drift apart. A drift there is invisible until search and
display disagree about what a recipe says.

The resolution rules are doc 5's per-type source rules:

- ingredients → ``structured_data["ingredients_text"]`` when non-empty, else the
  ``ingredients`` list's ``raw_text`` fields joined on newlines.
- steps       → ``structured_data["steps_text"]`` when non-empty, else the
  ``steps`` list's ``text`` fields joined on newlines.
- body        → title + ingredients + steps blocks joined by a blank line,
  skipping any block that is empty or whitespace-only.
"""

from __future__ import annotations

import unicodedata
from typing import Any

__all__ = [
    "compose_body_text",
    "is_present",
    "join_blocks",
    "normalize_title",
    "resolve_ingredients_text",
    "resolve_steps_text",
]


def normalize_title(title: str) -> str:
    """Deterministically normalize a title for dedup/lookup (doc 2 § 4).

    Unicode-NFC → lowercase → collapse internal whitespace runs to a single
    space → strip. Pure, locale-independent, and idempotent
    (``"Tomato and White Bean Soup"`` → ``"tomato and white bean soup"``).

    Lives here rather than in ``pipeline/persist`` (which re-exports it for its
    existing callers) so the pure edit layer can reach it without importing a
    module that pulls in a session and ``Settings``.
    """
    folded = unicodedata.normalize("NFC", title).lower()
    return " ".join(folded.split())


def is_present(text: object) -> bool:
    """A source text counts only when it is a string with non-whitespace content.

    Typed against ``object`` rather than ``str | None`` because the callers read
    unwrapped ``JSONB``, where a number or a list is structurally possible; a
    non-string is simply absent rather than an exception.
    """
    return isinstance(text, str) and bool(text.strip())


def _row_texts(structured: dict[str, Any], list_key: str, text_key: str) -> str:
    """Join one field across a JSONB row list, tolerating junk entries.

    ``structured_data`` is unwrapped ``JSONB``: nothing at the database level
    stops a hand-edited row from holding a scalar where a list belongs, or a
    null where a string does. Anything that is not a string is skipped rather
    than raised on — a degraded row must not take down chunking or an edit.
    """
    rows = structured.get(list_key)
    if not isinstance(rows, list):
        return ""
    texts = [
        row.get(text_key)
        for row in rows
        if isinstance(row, dict) and isinstance(row.get(text_key), str)
    ]
    return "\n".join(text for text in texts if isinstance(text, str))


def resolve_ingredients_text(structured: dict[str, Any]) -> str:
    """``ingredients_text`` when present, else the ingredient list's ``raw_text``."""
    text = structured.get("ingredients_text")
    if is_present(text):
        return text  # type: ignore[return-value]
    return _row_texts(structured, "ingredients", "raw_text")


def resolve_steps_text(structured: dict[str, Any]) -> str:
    """``steps_text`` when present, else the step list's ``text`` fields."""
    text = structured.get("steps_text")
    if is_present(text):
        return text  # type: ignore[return-value]
    return _row_texts(structured, "steps", "text")


def join_blocks(blocks: list[str]) -> str:
    """Concatenate the non-empty text blocks separated by a blank line."""
    return "\n\n".join(block for block in blocks if is_present(block))


def compose_body_text(*, title: str, structured: dict[str, Any]) -> str:
    """Compose the canonical body text for an item from its title and payload.

    This is exactly what ``build_chunks`` falls back to when an item carries no
    ``body_text``, and exactly what an edit rebuilds ``body_text`` to when the
    ingredient or step lines change — one implementation, so the ``recipe_full``
    chunk and the stored ``body_text`` cannot say different things.
    """
    return join_blocks(
        [
            title,
            resolve_ingredients_text(structured),
            resolve_steps_text(structured),
        ]
    )
