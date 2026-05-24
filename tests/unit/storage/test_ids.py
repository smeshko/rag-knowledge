"""Unit tests for rag_recipes.storage.ids."""

from __future__ import annotations

import re

from rag_recipes.storage.ids import new_id

_ULID_RE = re.compile(r"^[a-z]+_[0-9A-HJKMNP-TV-Z]{26}$")


def test_prefix_is_preserved() -> None:
    value = new_id("asset")
    assert value.startswith("asset_")


def test_id_has_ulid_shape() -> None:
    value = new_id("doc")
    assert _ULID_RE.match(value), value


def test_ids_are_unique() -> None:
    ids = {new_id("span") for _ in range(10_000)}
    assert len(ids) == 10_000
