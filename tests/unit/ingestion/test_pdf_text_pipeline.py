"""Unit tests for the pure hashing helpers in ingestion.pipeline.pdf_text."""

from __future__ import annotations

import hashlib

from rag_recipes.ingestion.pipeline.pdf_text import (
    EmptyPdfError,
    _sha256_json,
    _sha256_text,
)


def test_sha256_json_normalises_key_order() -> None:
    assert _sha256_json({"a": 1, "b": 2}) == _sha256_json({"b": 2, "a": 1})


def test_sha256_json_is_canonical_form() -> None:
    # The helper serialises with sort_keys + compact separators, so the digest
    # equals the hash of that exact canonical byte string regardless of how the
    # input dict was constructed.
    canonical = b'{"a":1,"b":2}'
    assert _sha256_json({"b": 2, "a": 1}) == hashlib.sha256(canonical).hexdigest()


def test_sha256_text_uses_utf8() -> None:
    assert _sha256_text("café") == hashlib.sha256("café".encode()).hexdigest()


def test_empty_pdf_error_subclass() -> None:
    assert issubclass(EmptyPdfError, ValueError)
