"""arq Redis pool + enqueue helpers for the ingestion worker.

`_build_redis_settings` is intentionally implemented on top of
`redis.asyncio.connection.parse_url` — the same parser `Settings`'s
validators use — rather than `RedisSettings.from_dsn`. arq 0.28's
`from_dsn` uses `urlparse().password`, which silently leaves
percent-encoded passwords (`p%40ss`) undecoded and ignores unix-socket
query-string passwords. The project's `Settings` validators accept the
decoded form, so the worker pool must agree or auth diverges between API
and worker. See `.claude/plans/epic-07-phase-7-1-arq-worker/DECISIONS.md`
§9.
"""

from __future__ import annotations

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from redis.asyncio.connection import SSLConnection
from redis.asyncio.connection import parse_url as redis_parse_url

from rag_recipes.config import Settings


def _build_redis_settings(settings: Settings) -> RedisSettings:
    parsed = redis_parse_url(settings.redis_url)
    # redis-py signals TLS via `connection_class=SSLConnection` rather than an
    # explicit `ssl` key, so translate that into the boolean arq expects.
    connection_class = parsed.get("connection_class")
    use_ssl = isinstance(connection_class, type) and issubclass(connection_class, SSLConnection)
    return RedisSettings(
        host=str(parsed.get("host", "localhost")),
        port=int(parsed.get("port", 6379)),
        password=parsed.get("password"),
        database=int(parsed.get("db", 0)),
        ssl=use_ssl,
    )


async def create_arq_pool(settings: Settings) -> ArqRedis:
    return await create_pool(_build_redis_settings(settings))
