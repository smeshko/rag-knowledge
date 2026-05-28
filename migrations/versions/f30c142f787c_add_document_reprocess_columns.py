"""add document reprocess columns

Revision ID: f30c142f787c
Revises: 3243a5a83cfe
Create Date: 2026-05-28 11:58:44.319038

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f30c142f787c'
down_revision: Union[str, Sequence[str], None] = '3243a5a83cfe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('documents', sa.Column('last_reprocess_mode', sa.Text(), nullable=True))
    op.add_column('documents', sa.Column('last_reprocess_reason', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('documents', 'last_reprocess_reason')
    op.drop_column('documents', 'last_reprocess_mode')
