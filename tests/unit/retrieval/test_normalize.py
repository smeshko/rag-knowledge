"""Unit tests for normalize_query (doc 7 § 2)."""

from __future__ import annotations

import pytest

from rag_recipes.retrieval.normalize import normalize_query


def test_trims_collapses_and_lowercases_preserving_original() -> None:
    result = normalize_query("  Cozy soup with WHITE beans!! ")
    # Three steps: trim, collapse internal whitespace, lowercase. Punctuation kept.
    assert result.keyword == "cozy soup with white beans!!"
    # The verbatim original is preserved for display.
    assert result.original == "  Cozy soup with WHITE beans!! "


def test_collapses_runs_of_internal_whitespace() -> None:
    result = normalize_query("white\t\tbeans   and\nrice")
    assert result.keyword == "white beans and rice"


def test_is_idempotent_on_already_normalized_input() -> None:
    once = normalize_query("white beans and rice")
    twice = normalize_query(once.keyword)
    assert once.keyword == twice.keyword == "white beans and rice"


@pytest.mark.parametrize("raw", ["", "   ", "\t\n  \n"])
def test_empty_or_whitespace_only_yields_empty_keyword(raw: str) -> None:
    result = normalize_query(raw)
    assert result.keyword == ""
    assert result.original == raw


def test_does_not_strip_punctuation() -> None:
    # Punctuation reduction is plainto_tsquery's job, not normalize_query's.
    assert normalize_query("white beans!!").keyword == "white beans!!"
