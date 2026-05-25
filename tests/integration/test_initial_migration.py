"""Integration tests for the initial-schema migration mechanics.

Covers object presence in the catalog (tables, extension, enums, indexes,
uniqueness constraints) and a downgrade/upgrade round-trip that proves the
migration reverses cleanly while leaving the `vector` extension installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TABLES = {
    "source_assets",
    "documents",
    "source_spans",
    "extraction_runs",
    "knowledge_items",
    "chunks",
    "chunk_embeddings",
}
EXPECTED_ENUMS = {
    "source_type_enum",
    "upload_status_enum",
    "document_status_enum",
    "extraction_run_status_enum",
    "knowledge_item_status_enum",
    "chunk_type_enum",
    "chunk_parent_type_enum",
}
EXPECTED_INDEXES = {
    "ix_chunks_document_id",
    "ix_chunks_parent_type_parent_id",
    "ix_knowledge_items_document_id",
    "ix_knowledge_items_status",
    "ix_source_spans_document_id_source_version",
}
EXPECTED_UNIQUES = {
    "uq_source_assets_content_hash",
    "uq_documents_asset_id",
    "uq_source_spans_document_version_locator",
    "uq_chunk_embeddings_chunk_provider_model",
    # Back the denormalization-enforcing composite FKs.
    "uq_extraction_runs_id_document",
    "uq_knowledge_items_id_document",
}


async def _table_set(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE'"
            )
        )
        return {row[0] for row in rows}


async def _enum_set(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT typname FROM pg_type WHERE typtype='e' AND typname LIKE '%_enum'")
        )
        return {row[0] for row in rows}


async def test_alembic_upgrade_head_creates_all_seven_tables(db_session: AsyncSession) -> None:
    result = await db_session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE'"
        )
    )
    actual = {row[0] for row in result}
    assert EXPECTED_TABLES.issubset(actual), f"missing: {EXPECTED_TABLES - actual}"


async def test_pgvector_extension_installed(db_session: AsyncSession) -> None:
    result = await db_session.execute(
        text("SELECT extname FROM pg_extension WHERE extname = :n"),
        {"n": "vector"},
    )
    assert result.scalar_one() == "vector"


async def test_enum_types_defined(db_session: AsyncSession) -> None:
    result = await db_session.execute(
        text("SELECT typname FROM pg_type WHERE typtype='e' AND typname LIKE '%_enum'")
    )
    actual = {row[0] for row in result}
    assert EXPECTED_ENUMS.issubset(actual), f"missing: {EXPECTED_ENUMS - actual}"


async def test_explicit_indexes_present(db_session: AsyncSession) -> None:
    result = await db_session.execute(
        text("SELECT indexname FROM pg_indexes WHERE schemaname='public'")
    )
    actual = {row[0] for row in result}
    assert EXPECTED_INDEXES.issubset(actual), f"missing: {EXPECTED_INDEXES - actual}"


async def test_unique_constraints_present(db_session: AsyncSession) -> None:
    result = await db_session.execute(
        text(
            "SELECT conname FROM pg_constraint "
            "WHERE contype='u' AND connamespace='public'::regnamespace"
        )
    )
    actual = {row[0] for row in result}
    assert EXPECTED_UNIQUES.issubset(actual), f"missing: {EXPECTED_UNIQUES - actual}"


async def test_downgrade_base_then_upgrade_head_roundtrips(
    postgres_test_dsn: str, test_engine: AsyncEngine
) -> None:
    """Bypasses db_session: it issues DDL (drop/create tables), which cannot run
    inside the SAVEPOINT db_session opens. Restores the schema before returning."""
    cfg = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", postgres_test_dsn)

    tables_before = await _table_set(test_engine)
    enums_before = await _enum_set(test_engine)
    assert EXPECTED_TABLES.issubset(tables_before)
    assert EXPECTED_ENUMS.issubset(enums_before)

    await asyncio.to_thread(alembic_command.downgrade, cfg, "base")

    assert (await _table_set(test_engine)).isdisjoint(EXPECTED_TABLES)
    assert (await _enum_set(test_engine)).isdisjoint(EXPECTED_ENUMS)
    async with test_engine.connect() as conn:
        ext = (
            await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'"))
        ).first()
        assert ext is not None, "vector extension must survive downgrade"

    await asyncio.to_thread(alembic_command.upgrade, cfg, "head")

    assert await _table_set(test_engine) == tables_before
    assert await _enum_set(test_engine) == enums_before
