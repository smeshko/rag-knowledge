"""Unit tests for rag_recipes.storage.models.document."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.orm.attributes import set_committed_value

from rag_recipes.storage.enums import DocumentStatus, SourceType
from rag_recipes.storage.models import Document, SourceAsset


def _make_asset(source_type: str = SourceType.PDF) -> SourceAsset:
    return SourceAsset(
        source_type=source_type,
        original_filename="x.pdf",
        storage_provider="s3",
        storage_key="k",
        content_hash="h",
        upload_status="uploaded",
    )


def test_tablename() -> None:
    assert Document.__tablename__ == "documents"


def test_table_columns_match_doc_2() -> None:
    expected = [
        "id",
        "asset_id",
        "category",
        "subcategory",
        "title",
        "author",
        "source_type",
        "language",
        "active_source_version",
        "status",
        "last_reprocess_mode",
        "last_reprocess_reason",
        "created_at",
        "updated_at",
    ]
    assert list(Document.__table__.columns.keys()) == expected


def test_asset_id_is_foreign_key_and_unique() -> None:
    col = Document.__table__.columns["asset_id"]
    assert col.unique is True
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "source_assets.id"


def test_nullable_columns() -> None:
    cols = Document.__table__.columns
    assert cols["subcategory"].nullable is True
    assert cols["language"].nullable is True
    assert cols["active_source_version"].nullable is True
    assert cols["last_reprocess_mode"].nullable is True
    assert cols["last_reprocess_reason"].nullable is True


def test_reprocess_columns_default_to_none() -> None:
    doc = Document()
    assert doc.last_reprocess_mode is None
    assert doc.last_reprocess_reason is None


def test_default_id_uses_doc_prefix() -> None:
    default = Document.__table__.columns["id"].default
    assert default is not None
    value = default.arg(None)
    assert value.startswith("doc_")


def test_status_enum_uses_document_status_enum_type() -> None:
    col_type = Document.__table__.columns["status"].type
    assert isinstance(col_type, sa.Enum)
    assert col_type.name == "document_status_enum"
    assert col_type.native_enum is True
    assert col_type.enum_class is DocumentStatus


def test_source_type_enum_reuses_existing_type() -> None:
    col_type = Document.__table__.columns["source_type"].type
    assert isinstance(col_type, sa.Enum)
    assert col_type.name == "source_type_enum"
    assert col_type.enum_class is SourceType


def test_source_spans_relationship_does_not_delete_orphans() -> None:
    """SourceSpan rows are immutable/append-only (doc 2 §3 + PLAN line 79).

    A delete-orphan cascade would DELETE spans when they are removed from
    Document.source_spans (e.g. a reprocess flow swapping in a new version),
    even though ExtractionRun.input_source_span_ids still references them by
    id in a JSONB array with no FK protection. Cascade/ON DELETE policy is
    explicitly deferred to Phase 2.3 (PLAN line 36).
    """
    rel = sa.inspect(Document).relationships["source_spans"]
    assert rel.cascade.delete_orphan is False
    assert rel.cascade.delete is False


def test_source_type_mismatch_after_asset_attach_raises() -> None:
    doc = Document(source_type=SourceType.PDF)
    mismatch = _make_asset()
    # Seed a sentinel via the committed state to bypass the enum coercer.
    set_committed_value(mismatch, "source_type", "_other")
    with pytest.raises(ValueError, match="source_type"):
        doc.asset = mismatch


def test_source_type_mismatch_after_source_type_change_raises() -> None:
    asset = _make_asset()
    set_committed_value(asset, "source_type", "_other")
    doc = Document()
    doc.asset = asset
    with pytest.raises(ValueError, match="source_type"):
        doc.source_type = SourceType.PDF


def test_matching_source_type_passes() -> None:
    asset = _make_asset()
    doc = Document(source_type=SourceType.PDF)
    doc.asset = asset
    assert doc.asset is asset
    assert doc.source_type == SourceType.PDF
