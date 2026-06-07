"""phase 19.2 extraction batches

Revision ID: f1a2b3c4d5e6
Revises: 77b2b79ace95
Create Date: 2026-06-07

Adds the two Anthropic-batch tracking tables (Epic 19.2): `extraction_batches`
(one row per submitted/being-submitted batch) and `extraction_batch_items` (the
per-window registration rows, whose id doubles as the Anthropic `custom_id`). The
partial-unique index on `(document_id, source_version, input_hash)` for
non-terminal statuses is the load-bearing registration-idempotency guard.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "77b2b79ace95"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the two extraction-batch tables, their enum types, and indexes."""
    op.create_table(
        "extraction_batches",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("provider_batch_id", sa.Text(), nullable=True),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column(
            "processing_status",
            sa.Enum(
                "submitting",
                "submitted",
                "in_progress",
                "ended",
                "failed",
                name="extraction_batch_status_enum",
            ),
            nullable=False,
        ),
        sa.Column("request_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "extraction_batch_items",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("source_version", sa.Integer(), nullable=False),
        sa.Column("input_hash", sa.Text(), nullable=False),
        sa.Column(
            "input_source_span_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("request_input", sa.Text(), nullable=False),
        sa.Column(
            "request_schema",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "submitting",
                "submitted",
                "succeeded",
                "rejected",
                "errored",
                "expired",
                "canceled",
                name="extraction_batch_item_status_enum",
            ),
            nullable=False,
        ),
        sa.Column("batch_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.ForeignKeyConstraint(["batch_id"], ["extraction_batches.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_extraction_batch_items_status",
        "extraction_batch_items",
        ["status"],
    )
    op.create_index(
        "ix_extraction_batch_items_document_source_version",
        "extraction_batch_items",
        ["document_id", "source_version"],
    )
    op.create_index(
        "uq_extraction_batch_items_doc_version_hash_active",
        "extraction_batch_items",
        ["document_id", "source_version", "input_hash"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'submitting', 'submitted')"),
    )


def downgrade() -> None:
    """Drop both tables and their enum types."""
    op.drop_index(
        "uq_extraction_batch_items_doc_version_hash_active",
        table_name="extraction_batch_items",
    )
    op.drop_index(
        "ix_extraction_batch_items_document_source_version",
        table_name="extraction_batch_items",
    )
    op.drop_index(
        "ix_extraction_batch_items_status",
        table_name="extraction_batch_items",
    )
    op.drop_table("extraction_batch_items")
    op.drop_table("extraction_batches")
    sa.Enum(name="extraction_batch_item_status_enum").drop(op.get_bind())
    sa.Enum(name="extraction_batch_status_enum").drop(op.get_bind())
