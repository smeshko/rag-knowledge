"""favourites

Revision ID: c5f1a9b73de4
Revises: b4e07c9a51d2
Create Date: 2026-08-25 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c5f1a9b73de4'
down_revision: Union[str, Sequence[str], None] = 'b4e07c9a51d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    The reader's star on a recipe, as its own table rather than a
    ``knowledge_items.favourited_at`` column — see the model docstring: a column
    would bump ``knowledge_items.updated_at``, which
    ``sweep_stuck_indexing_items`` reads as "this item made lifecycle progress".

    ``knowledge_item_id`` is the whole primary key (single-user app), which is
    what makes the ``PUT`` idempotent via ``ON CONFLICT DO NOTHING``.
    ``ON DELETE CASCADE`` means neither delete path — per-recipe or per-book —
    needs to learn about this table.
    """
    op.create_table(
        "knowledge_item_favourites",
        sa.Column("knowledge_item_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_item_id"],
            ["knowledge_items.id"],
            name="fk_knowledge_item_favourites_knowledge_item",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "knowledge_item_id", name="pk_knowledge_item_favourites"
        ),
    )
    # Backs the GET /favourites ordering (newest star first).
    op.create_index(
        "ix_knowledge_item_favourites_created_at",
        "knowledge_item_favourites",
        [sa.text("created_at DESC")],
    )


def downgrade() -> None:
    """Downgrade schema.

    Drops the table, and with it every star. Nothing else references it, so
    this is a clean reversal — the recipes themselves are untouched.
    """
    op.drop_index(
        "ix_knowledge_item_favourites_created_at",
        table_name="knowledge_item_favourites",
    )
    op.drop_table("knowledge_item_favourites")
