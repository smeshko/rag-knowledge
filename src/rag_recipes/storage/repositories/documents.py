"""Data-access primitives for ``SourceAsset`` + ``Document``.

The repository owns no transaction control: callers (the upload route)
flush/commit/rollback explicitly. Methods that insert ``session.add`` +
``await session.flush()`` so server defaults and IDs are populated before
the caller takes the next step.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_source_asset_by_content_hash(
        self, content_hash: str
    ) -> SourceAsset | None:
        result = await self._session.execute(
            select(SourceAsset).where(SourceAsset.content_hash == content_hash)
        )
        return result.scalar_one_or_none()

    async def get_document_by_asset_id(self, asset_id: str) -> Document | None:
        # Explicit query — never `await`-trigger the lazy SourceAsset.document
        # relationship, which is a default-lazy load and breaks on AsyncSession.
        result = await self._session.execute(
            select(Document).where(Document.asset_id == asset_id)
        )
        return result.scalar_one_or_none()

    async def add_source_asset(
        self,
        *,
        id: str,
        source_type: SourceType,
        original_filename: str,
        storage_provider: str,
        storage_key: str,
        content_hash: str,
        upload_status: UploadStatus,
    ) -> SourceAsset:
        asset = SourceAsset(
            id=id,
            source_type=source_type,
            original_filename=original_filename,
            storage_provider=storage_provider,
            storage_key=storage_key,
            content_hash=content_hash,
            upload_status=upload_status,
        )
        self._session.add(asset)
        await self._session.flush()
        return asset

    async def add_document(
        self,
        *,
        asset_id: str,
        category: str,
        subcategory: str | None,
        title: str,
        author: str,
        source_type: SourceType,
        language: str | None,
        active_source_version: int | None,
        status: DocumentStatus,
    ) -> Document:
        # Set source_type as a scalar column; do not attach the asset
        # relationship — that would trigger the cross-validator against a
        # possibly-half-built graph (see PLAN Risks).
        document = Document(
            asset_id=asset_id,
            category=category,
            subcategory=subcategory,
            title=title,
            author=author,
            source_type=source_type,
            language=language,
            active_source_version=active_source_version,
            status=status,
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def get_document(self, document_id: str) -> Document | None:
        return await self._session.get(Document, document_id)
