"""phase 19.3 batch item retry/audit columns

Revision ID: a7c9e1d3b5f2
Revises: f1a2b3c4d5e6
Create Date: 2026-06-07

Adds the retry/audit columns 19.3 needs on `extraction_batch_items`:
`submit_attempts` (bounds re-submission of expired/transient-errored windows;
server_default 0 so existing rows backfill), plus nullable `result_type` /
`error_message` recording the provider outcome at ingestion.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c9e1d3b5f2"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "extraction_batch_items",
        sa.Column(
            "submit_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "extraction_batch_items",
        sa.Column("result_type", sa.Text(), nullable=True),
    )
    op.add_column(
        "extraction_batch_items",
        sa.Column("error_message", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("extraction_batch_items", "error_message")
    op.drop_column("extraction_batch_items", "result_type")
    op.drop_column("extraction_batch_items", "submit_attempts")
