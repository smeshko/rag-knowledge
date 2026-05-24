"""Prefixed-ULID primary-key generation for ORM models."""

from __future__ import annotations

from ulid import ULID


def new_id(prefix: str) -> str:
    return f"{prefix}_{ULID()}"
