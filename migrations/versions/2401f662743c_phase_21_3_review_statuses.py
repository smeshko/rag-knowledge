"""phase 21.3 review statuses

Revision ID: 2401f662743c
Revises: a7c9e1d3b5f2
Create Date: 2026-08-01 13:16:28.940126

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '2401f662743c'
down_revision: Union[str, Sequence[str], None] = 'a7c9e1d3b5f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Phase 21.3: add the two review-decision labels to the knowledge-item
    status enum — 'rejected' (terminal reject) and 'indexing' (transitional
    approve state consumed by the index_knowledge_item job).

    Phase 9.5 precedent (DECISIONS #5 there, D2 here): each value is added in
    its own statement and referenced nowhere else in this migration. PG forbids
    *using* a freshly-added enum value in the same transaction that adds it;
    merely adding it is allowed on PG 12+, so both statements stay inside
    Alembic's transaction. IF NOT EXISTS makes them idempotent.
    """
    op.execute(
        "ALTER TYPE knowledge_item_status_enum ADD VALUE IF NOT EXISTS 'rejected'"
    )
    op.execute(
        "ALTER TYPE knowledge_item_status_enum ADD VALUE IF NOT EXISTS 'indexing'"
    )


def downgrade() -> None:
    """Downgrade schema.

    Documented no-op (D2): Postgres cannot remove a value from an enum type,
    so 'rejected' and 'indexing' intentionally remain installed — inert once
    the consuming code is downgraded and no rows are written in those
    statuses. The migration is downgrade-clean, not value-reversible.
    """
