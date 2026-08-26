"""manual source type

Revision ID: d8b3c60fae91
Revises: c5f1a9b73de4
Create Date: 2026-08-25 15:20:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd8b3c60fae91'
down_revision: Union[str, Sequence[str], None] = 'c5f1a9b73de4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    One new value on ``source_type_enum``: ``manual``, the source a hand-typed
    recipe carries. It marks the handwritten shelf — a Document with no PDF
    behind it, whose SourceAsset row exists only to satisfy
    ``documents.asset_id``.

    ``ADD VALUE`` is transactional on PostgreSQL 12+, so this needs no
    ``COMMIT`` escape hatch; ``IF NOT EXISTS`` keeps a re-run harmless. The
    value is only *added* here and first *used* by a later transaction (the
    create endpoint's shelf bootstrap), which is what the pre-15 restriction on
    using a freshly-added label actually forbids.

    Nothing to backfill: every existing row is ``pdf``.
    """
    op.execute("ALTER TYPE source_type_enum ADD VALUE IF NOT EXISTS 'manual'")


def downgrade() -> None:
    """Downgrade schema.

    Deliberately a no-op. PostgreSQL cannot drop a value from an enum type, and
    the alternative — rebuild the type, rewrite both columns that use it — would
    fail anyway for any row that reached ``manual``, i.e. exactly the
    installations where the downgrade would matter. An unused extra label is
    inert; the initial-migration round-trip test drops the type wholesale.
    """
