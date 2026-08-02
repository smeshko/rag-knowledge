"""phase 22.1 knowledge item edit audit

Revision ID: b4e07c9a51d2
Revises: 2401f662743c
Create Date: 2026-08-02 09:10:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b4e07c9a51d2'
down_revision: Union[str, Sequence[str], None] = '2401f662743c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Phase 22.1: the audit trail for in-place editing of a needs_review item.

    ``pre_edit_snapshot`` stores the original extraction (title, summary,
    body_text, structured_data, confidence) captured on the FIRST edit only, so
    it stays the model's output rather than the previous revision — an undo path,
    and the ground truth the extraction evals must keep scoring.
    ``edited_at`` timestamps the most recent edit.

    Both are nullable with no backfill: null means "never edited", which is
    exactly true of every pre-existing row.
    """
    op.add_column(
        "knowledge_items",
        sa.Column("pre_edit_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "knowledge_items",
        sa.Column("edited_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema.

    Drops both columns. Any captured snapshots are lost, which is the accepted
    cost of a downgrade — the edited content itself stays on the row.
    """
    op.drop_column("knowledge_items", "edited_at")
    op.drop_column("knowledge_items", "pre_edit_snapshot")
