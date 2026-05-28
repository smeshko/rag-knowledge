"""create ingestion_failures

Revision ID: e81558759f9b
Revises: f30c142f787c
Create Date: 2026-05-28 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e81558759f9b'
down_revision: Union[str, Sequence[str], None] = 'f30c142f787c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'ingestion_failures',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('document_id', sa.Text(), nullable=False),
        sa.Column(
            'failed_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'last_status',
            postgresql.ENUM(
                'queued',
                'extracting_text',
                'creating_source_spans',
                'extracting_items',
                'validating_items',
                'creating_chunks',
                'embedding_chunks',
                'indexing',
                'ready',
                'needs_review',
                'failed',
                name='document_status_enum',
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column(
            'metadata_json',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_ingestion_failures_document_id',
        'ingestion_failures',
        ['document_id'],
    )
    op.create_index(
        'ix_ingestion_failures_failed_at',
        'ingestion_failures',
        ['failed_at'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_ingestion_failures_failed_at', table_name='ingestion_failures')
    op.drop_index('ix_ingestion_failures_document_id', table_name='ingestion_failures')
    op.drop_table('ingestion_failures')
