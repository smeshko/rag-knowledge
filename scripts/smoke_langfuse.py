"""Submit a trace via the Langfuse SDK and verify it lands in the public API.

Used by ``just setup`` (and standalone via ``just smoke-langfuse``) to prove
that the local Langfuse stack ingests traces end to end: web → Redis → worker
→ ClickHouse. A green container healthcheck only covers the worker process and
its Postgres connection — not the full ingestion path. Phase 1.6 adds this
script so a broken pipeline fails the bootstrap loudly instead of producing
a quietly-empty UI.

When ``LANGFUSE_ENABLED`` is false the script prints a notice and exits 0.

Knob: ``POLL_TIMEOUT_S`` (default 30) gives ~2x headroom over the empirical
<15 s cold-start ingestion latency on Apple Silicon. Bump it if cold caches
on slower hardware bite.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime

import httpx
from dotenv import dotenv_values
from langfuse import Langfuse

from rag_recipes.config import get_settings

POLL_TIMEOUT_S = 30
POLL_INTERVAL_S = 1


def main() -> int:
    settings = get_settings()
    if not settings.langfuse_enabled:
        print("Langfuse disabled — skipping smoke")
        return 0

    # Settings reads .env via pydantic-settings but does not populate
    # os.environ, so the init project id (env-resident only) needs a
    # second read via python-dotenv (transitive through pydantic-settings).
    env_values = {**dotenv_values(".env"), **os.environ}
    init_project_id = env_values.get("LANGFUSE_INIT_PROJECT_ID", "<unknown-project>")

    host = settings.langfuse_host
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key

    name = f"smoke-setup-{datetime.now(UTC).isoformat(timespec='seconds')}"
    trace_id = Langfuse.create_trace_id()

    lf = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    try:
        with lf.start_as_current_observation(
            name=name,
            as_type="span",
            trace_context={"trace_id": trace_id},
        ) as span:
            span.update(input={"smoke": "phase-1.6"}, output={"ok": True})
        lf.flush()
    except httpx.RequestError as exc:
        print(
            f"Langfuse host unreachable at {host} ({type(exc).__name__}) "
            "— is the stack up? (docker compose ps)"
        )
        return 1

    api_url = f"{host}/api/public/traces/{trace_id}"
    ui_url = f"{host}/project/{init_project_id}/traces/{trace_id}"

    started = time.monotonic()
    last_status: int | None = None
    last_body: str = ""
    with httpx.Client(auth=(public_key, secret_key), timeout=5.0) as client:
        for _ in range(POLL_TIMEOUT_S):
            try:
                response = client.get(api_url)
            except httpx.RequestError as exc:
                # Covers ConnectError, TimeoutException, ReadError, etc.
                # Without this, a hung langfuse-web yields an unhandled
                # traceback inside `just setup` instead of the diagnostic.
                last_body = f"{type(exc).__name__}: {exc}"
                time.sleep(POLL_INTERVAL_S)
                continue
            last_status = response.status_code
            last_body = response.text[:200]
            if response.status_code == 200:
                elapsed = time.monotonic() - started
                print(f"Smoke trace visible (in {elapsed:.1f}s):")
                print(f"  API: {api_url}")
                print(f"  UI:  {ui_url}")
                return 0
            if response.status_code in {401, 403}:
                print(
                    "Auth failed — LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
                    "don't match the seeded project"
                )
                return 1
            # 404 = trace not yet materialised, keep polling. Other statuses
            # (5xx, unexpected 4xx) also poll — last_status/last_body surface
            # the failing response in the post-loop diagnostic.
            time.sleep(POLL_INTERVAL_S)

    elapsed = time.monotonic() - started
    print(
        f"Smoke trace did not materialise in {elapsed:.1f}s "
        f"(last status: {last_status}, body: {last_body!r}). "
        "Worker may be down — check `docker compose logs langfuse-worker` "
        "(also tail langfuse-clickhouse and redis)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
