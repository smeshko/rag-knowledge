"""FastAPI dependency factories for settings, DB sessions, and Redis."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.config import Settings
from rag_recipes.config import get_settings as _get_settings


def get_settings() -> Settings:
    return _get_settings()


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


def get_redis(request: Request) -> Redis:
    redis_client: Redis = request.app.state.redis
    return redis_client
