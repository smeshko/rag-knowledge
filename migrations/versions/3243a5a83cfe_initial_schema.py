"""initial schema

Revision ID: 3243a5a83cfe
Revises:
Create Date: 2026-05-25 07:45:10.767080

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '3243a5a83cfe'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.create_table('source_assets',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('source_type', sa.Enum('pdf', name='source_type_enum'), nullable=False),
    sa.Column('original_filename', sa.Text(), nullable=False),
    sa.Column('storage_provider', sa.Text(), nullable=False),
    sa.Column('storage_key', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.Text(), nullable=False),
    sa.Column('upload_status', sa.Enum('uploading', 'uploaded', 'upload_failed', 'deleted', name='upload_status_enum'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('content_hash', name='uq_source_assets_content_hash')
    )
    op.create_table('documents',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('asset_id', sa.Text(), nullable=False),
    sa.Column('category', sa.Text(), nullable=False),
    sa.Column('subcategory', sa.Text(), nullable=True),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('author', sa.Text(), nullable=False),
    sa.Column('source_type', sa.Enum('pdf', name='source_type_enum', create_type=False), nullable=False),
    sa.Column('language', sa.Text(), nullable=True),
    sa.Column('active_source_version', sa.Integer(), nullable=True),
    sa.Column('status', sa.Enum('queued', 'extracting_text', 'creating_source_spans', 'extracting_items', 'validating_items', 'creating_chunks', 'embedding_chunks', 'indexing', 'ready', 'needs_review', 'failed', name='document_status_enum'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['asset_id'], ['source_assets.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('asset_id', name='uq_documents_asset_id')
    )
    op.create_table('source_spans',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('document_id', sa.Text(), nullable=False),
    sa.Column('source_version', sa.Integer(), nullable=False),
    sa.Column('source_type', sa.Enum('pdf', name='source_type_enum', create_type=False), nullable=False),
    sa.Column('locator', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('locator_hash', sa.Text(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('text_hash', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'source_version', 'locator_hash', name='uq_source_spans_document_version_locator')
    )
    op.create_table('extraction_runs',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('document_id', sa.Text(), nullable=False),
    sa.Column('source_version', sa.Integer(), nullable=False),
    sa.Column('provider', sa.Text(), nullable=False),
    sa.Column('model', sa.Text(), nullable=False),
    sa.Column('prompt_version', sa.Text(), nullable=False),
    sa.Column('schema_version', sa.Text(), nullable=False),
    sa.Column('input_source_span_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('input_hash', sa.Text(), nullable=False),
    sa.Column('status', sa.Enum('running', 'success', 'failed', 'rejected', name='extraction_run_status_enum'), nullable=False),
    sa.Column('output_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('id', 'document_id', 'source_version', name='uq_extraction_runs_id_document_version')
    )
    op.create_table('knowledge_items',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('document_id', sa.Text(), nullable=False),
    sa.Column('extraction_run_id', sa.Text(), nullable=False),
    sa.Column('source_version', sa.Integer(), nullable=False),
    sa.Column('item_type', sa.Text(), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('normalized_title', sa.Text(), nullable=False),
    sa.Column('summary', sa.Text(), nullable=True),
    sa.Column('body_text', sa.Text(), nullable=False),
    sa.Column('source_span_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('structured_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('confidence', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('status', sa.Enum('ready', 'needs_review', 'superseded', name='knowledge_item_status_enum'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.ForeignKeyConstraint(['extraction_run_id'], ['extraction_runs.id'], ),
    sa.ForeignKeyConstraint(['extraction_run_id', 'document_id', 'source_version'], ['extraction_runs.id', 'extraction_runs.document_id', 'extraction_runs.source_version'], name='fk_knowledge_items_extraction_run_document_version'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('id', 'document_id', name='uq_knowledge_items_id_document')
    )
    op.create_table('chunks',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('document_id', sa.Text(), nullable=False),
    sa.Column('parent_type', sa.Enum('knowledge_item', name='chunk_parent_type_enum'), nullable=False),
    sa.Column('parent_id', sa.Text(), nullable=False),
    sa.Column('chunk_type', sa.Enum('recipe_full', 'recipe_title', 'recipe_summary', 'recipe_ingredients', 'recipe_steps', name='chunk_type_enum'), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('text_hash', sa.Text(), nullable=False),
    sa.Column('source_span_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.ForeignKeyConstraint(['parent_id'], ['knowledge_items.id'], ),
    sa.ForeignKeyConstraint(['parent_id', 'document_id'], ['knowledge_items.id', 'knowledge_items.document_id'], name='fk_chunks_parent_document'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('chunk_embeddings',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('chunk_id', sa.Text(), nullable=False),
    sa.Column('embedding_provider', sa.Text(), nullable=False),
    sa.Column('embedding_model', sa.Text(), nullable=False),
    sa.Column('embedding_dimensions', sa.Integer(), nullable=False),
    sa.Column('embedding_vector', Vector(1536), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['chunk_id'], ['chunks.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('chunk_id', 'embedding_provider', 'embedding_model', name='uq_chunk_embeddings_chunk_provider_model')
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_parent_type_parent_id", "chunks", ["parent_type", "parent_id"])
    op.create_index("ix_knowledge_items_document_id", "knowledge_items", ["document_id"])
    op.create_index("ix_knowledge_items_status", "knowledge_items", ["status"])
    op.create_index(
        "ix_source_spans_document_id_source_version",
        "source_spans",
        ["document_id", "source_version"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    # vector extension is a superuser cluster object; leave it installed.
    op.drop_index("ix_source_spans_document_id_source_version", table_name="source_spans")
    op.drop_index("ix_knowledge_items_status", table_name="knowledge_items")
    op.drop_index("ix_knowledge_items_document_id", table_name="knowledge_items")
    op.drop_index("ix_chunks_parent_type_parent_id", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    # Tables in reverse FK order; their inline unique constraints drop with them.
    op.drop_table('chunk_embeddings')
    op.drop_table('chunks')
    op.drop_table('knowledge_items')
    op.drop_table('extraction_runs')
    op.drop_table('source_spans')
    op.drop_table('documents')
    op.drop_table('source_assets')
    # Enum types — explicit, because op.drop_table does not cascade-drop them.
    for enum_name in (
        "chunk_type_enum",
        "chunk_parent_type_enum",
        "knowledge_item_status_enum",
        "extraction_run_status_enum",
        "document_status_enum",
        "upload_status_enum",
        "source_type_enum",
    ):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
