"""Declarative base + pgvector re-export shared by ORM models and Alembic."""

from __future__ import annotations

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


__all__ = ["Base", "Vector"]
