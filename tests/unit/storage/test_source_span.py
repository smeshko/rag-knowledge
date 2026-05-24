"""Unit tests for rag_recipes.storage.models.source_span."""

from __future__ import annotations

import pytest
from sqlalchemy.dialects.postgresql import JSONB

from rag_recipes.storage.enums import SourceType
from rag_recipes.storage.models import Document, SourceSpan


def _make_doc(source_type: str = SourceType.PDF) -> Document:
    return Document(source_type=source_type)


def _make_span(**overrides: object) -> SourceSpan:
    defaults: dict[str, object] = {
        "source_version": 1,
        "source_type": SourceType.PDF,
        "locator": {"page": 1},
        "locator_hash": "lh",
        "text": "hello",
        "text_hash": "th",
    }
    defaults.update(overrides)
    return SourceSpan(**defaults)  # type: ignore[arg-type]


def test_tablename() -> None:
    assert SourceSpan.__tablename__ == "source_spans"


def test_table_columns_match_doc_2() -> None:
    expected = [
        "id",
        "document_id",
        "source_version",
        "source_type",
        "locator",
        "locator_hash",
        "text",
        "text_hash",
        "created_at",
    ]
    assert list(SourceSpan.__table__.columns.keys()) == expected


def test_no_updated_at_column() -> None:
    assert "updated_at" not in SourceSpan.__table__.columns


def test_locator_is_jsonb() -> None:
    assert isinstance(SourceSpan.__table__.columns["locator"].type, JSONB)


def test_document_id_is_foreign_key() -> None:
    col = SourceSpan.__table__.columns["document_id"]
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "documents.id"


def test_unique_constraint_on_document_version_locator() -> None:
    """One locator per (document_id, source_version) — see doc 2 §3 + Epic 8.

    Without this constraint, arq retries or concurrent ingestion runs can
    persist duplicate page spans for the same document/version and corrupt
    downstream extraction windows and citations.
    """
    from sqlalchemy import UniqueConstraint

    uniques = [c for c in SourceSpan.__table__.constraints if isinstance(c, UniqueConstraint)]
    matching = [
        c
        for c in uniques
        if {col.name for col in c.columns} == {"document_id", "source_version", "locator_hash"}
    ]
    assert len(matching) == 1, (
        "expected exactly one UniqueConstraint on (document_id, source_version, locator_hash)"
    )
    assert matching[0].name == "uq_source_spans_document_version_locator"


def test_default_id_uses_span_prefix() -> None:
    default = SourceSpan.__table__.columns["id"].default
    assert default is not None
    value = default.arg(None)
    assert value.startswith("span_")


def test_source_type_mismatch_after_doc_attach_raises() -> None:
    span = _make_span()
    doc = _make_doc()
    doc.source_type = "_other"  # bypass enum coercion on parent
    with pytest.raises(ValueError, match="source_type"):
        span.document = doc


def test_source_type_mismatch_after_source_type_change_raises() -> None:
    doc = _make_doc()
    doc.source_type = "_other"
    span = SourceSpan(
        source_version=1,
        locator={"page": 1},
        locator_hash="lh",
        text="x",
        text_hash="th",
    )
    span.document = doc
    with pytest.raises(ValueError, match="source_type"):
        span.source_type = SourceType.PDF


def test_matching_source_type_passes() -> None:
    doc = _make_doc()
    span = _make_span()
    span.document = doc
    assert span.document is doc
