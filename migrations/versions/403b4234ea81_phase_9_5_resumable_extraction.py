"""phase 9.5 resumable extraction

Revision ID: 403b4234ea81
Revises: e81558759f9b
Create Date: 2026-06-04 13:31:09.223036

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '403b4234ea81'
down_revision: Union[str, Sequence[str], None] = 'e81558759f9b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Phase 9.5: add the EXTRACTING staging label to the knowledge-item status
    enum and the two columns the resumable extraction loop needs —
    documents.last_progress_at (heartbeat) and knowledge_items.candidate_score
    (persisted dedup signal).
    """
    # DECISIONS #5: add the enum value in its own statement and reference it
    # nowhere else in this migration. PG forbids *using* a freshly-added enum
    # value in the same transaction that adds it; merely adding it (and never
    # using it here) is allowed on PG 12+, so this stays inside Alembic's
    # transaction. IF NOT EXISTS makes the statement idempotent.
    op.execute(
        "ALTER TYPE knowledge_item_status_enum ADD VALUE IF NOT EXISTS 'extracting'"
    )
    op.add_column(
        "documents",
        sa.Column("last_progress_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "knowledge_items",
        sa.Column("candidate_score", sa.Double(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema.

    Drops only the two columns. Postgres cannot remove a value from an enum
    type, so 'extracting' is left in knowledge_item_status_enum — inert with
    the columns gone and no rows written in that status (DECISIONS #5).
    """
    op.drop_column("knowledge_items", "candidate_score")
    op.drop_column("documents", "last_progress_at")
