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

from typing import Any

__all__ = [
    "compose_body_text",
    "is_present",
    "join_blocks",
    "resolve_ingredients_text",
    "resolve_steps_text",
]


def is_present(text: str | None) -> bool:
    """A source text counts only when it has non-whitespace content."""
    return bool(text and text.strip())


def resolve_ingredients_text(structured: dict[str, Any]) -> str:
    """``ingredients_text`` when present, else the ingredient list's ``raw_text``."""
    text = structured.get("ingredients_text")
    if is_present(text):
        return text  # type: ignore[return-value]
    ingredients = structured.get("ingredients") or []
    return "\n".join(item.get("raw_text", "") for item in ingredients)


def resolve_steps_text(structured: dict[str, Any]) -> str:
    """``steps_text`` when present, else the step list's ``text`` fields."""
    text = structured.get("steps_text")
    if is_present(text):
        return text  # type: ignore[return-value]
    steps = structured.get("steps") or []
    return "\n".join(step.get("text", "") for step in steps)


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
