# Epic 1 — Foundation & Project Scaffolding

**Status**: Ready for dev

## Overview

Establish the project skeleton, tooling, infrastructure, and developer setup so all subsequent epics have a stable foundation. By the end of this epic, a fresh clone + a single bootstrap command brings up Postgres+pgvector, Redis, and Langfuse locally; the FastAPI app runs and serves `/health`; Alembic is initialized; and `pytest`, `ruff`, and `mypy` all run green on an empty codebase.

## Architecture references

- [01 — Big Picture](../../architecture/01-big-picture.md) — system goal and starting decisions
- [11 — Configuration and Providers](../../architecture/11-configuration-and-providers.md) — config and secrets separation; required env vars
- [13 — Implementation Decisions](../../architecture/13-implementation-decisions.md) — full stack picks
  - Topic 1: Python + FastAPI
  - Topic 7: Docker Compose with `pgvector/pgvector:pg17` + `redis:7-alpine` + Langfuse
  - Topic 8: Layer-driven `src/rag_recipes/` layout
  - Topic 9: pytest, ruff, mypy

## Dependencies

None. This epic unblocks everything else.

## Out of scope

- SQLAlchemy ORM models or migrations (Epic 2)
- Provider interfaces and implementations (Epics 3–5)
- Any business logic — the FastAPI app exposes only `/health` at the end of this epic

---

## Phase 1.1 — Initial scaffold

**Goal**: Create the layer-driven `src/` directory tree, `pyproject.toml` with all chosen dependencies, and tooling configuration that runs green on an empty codebase.

### What to build

- **`pyproject.toml`** — project metadata, runtime + dev dependencies (justified in doc 13):
  - Runtime: `fastapi`, `uvicorn[standard]`, `sqlalchemy[asyncio]`, `alembic`, `pgvector`, `asyncpg`, `arq`, `redis`, `pydantic`, `pydantic-settings`, `openai`, `pymupdf`, `typer`, `langfuse`, `pytrec_eval-terrier`
  - Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `httpx`, `polyfactory`, `ruff`, `mypy`, plus type stubs as needed
- **`src/rag_recipes/`** package tree per doc 13 topic 8:

  ```text
  src/rag_recipes/
  ├── __init__.py
  ├── api/
  │   ├── __init__.py
  │   ├── app.py                  # populated in 1.3
  │   ├── routes/__init__.py
  │   ├── schemas/__init__.py
  │   └── dependencies.py
  ├── domain/__init__.py
  ├── ingestion/__init__.py
  ├── retrieval/__init__.py
  ├── providers/
  │   ├── __init__.py
  │   ├── file_storage/__init__.py
  │   ├── pdf_extractor/__init__.py
  │   ├── llm/__init__.py
  │   └── embeddings/__init__.py
  ├── storage/__init__.py
  └── config.py                   # populated in 1.3
  ```

- **`tests/`** structure: `unit/`, `integration/`, `fixtures/` with `__init__.py` files
- **`evals/`** structure: `scripts/`, `reports/`, `baselines/`, `README.md` placeholder
- **`data/`** structure: `fixtures/`, `storage/` (both gitignored except committed fixtures)
- **`.gitignore`** — Python artifacts, IDE files, `.env`, `data/storage/`, `data/fixtures/private/`, `evals/reports/*` (keep `evals/baselines/`)
- **`.env.example`** — all config vars planned (empty values), see doc 11 for the full list
- **Tooling config** (in `pyproject.toml` where possible):
  - `[tool.ruff]` — line length, target Python version, lint rule selection
  - `[tool.mypy]` — strict mode for `rag_recipes.domain.*` and `rag_recipes.providers.*`, relaxed elsewhere
  - `[tool.pytest.ini_options]` — `asyncio_mode = "auto"`, test paths, marker registration

### Acceptance criteria

- [ ] `uv sync` (or equivalent) succeeds in a fresh clone
- [ ] `uv run ruff check src/ tests/` passes
- [ ] `uv run ruff format --check src/ tests/` passes
- [ ] `uv run mypy src/` passes against empty modules
- [ ] `uv run pytest` runs and collects zero tests successfully
- [ ] All directories above exist with the expected `__init__.py` files

### Validation

`uv sync && uv run ruff check src/ tests/ && uv run mypy src/ && uv run pytest` — all green.

---

## Phase 1.2 — Docker Compose stack (Postgres + Redis)

**Goal**: A single `docker compose up -d` brings up the app's local infra — Postgres with pgvector and Redis — both healthy and reachable. The Langfuse self-hosted stack is delivered separately in [Phase 1.5](#phase-15--langfuse-self-hosted-observability-stack); see the note below.

### What to build

- **`docker-compose.yml`** with services:
  - `postgres` — image `pgvector/pgvector:pg17`, port `5432`, named volume `postgres-data`, healthcheck on `pg_isready`, env vars for user/password/db read from the project `.env` (dev credentials, NOT production secrets)
  - `redis` — image `redis:7-alpine`, port `6379`, healthcheck on `redis-cli ping`, no persistence (ephemeral)
  - A one-line TODO comment marking where the Phase 1.5 Langfuse services will slot in
- **`.dockerignore`** to keep build context small
- **`.env.example`** updated with the Postgres infra vars (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`)
- Dev credentials live in the project's `.env` (single config surface shared with the future Python app — no separate `.env.docker`)

> **Why Langfuse is split out:** doc 13 §13 originally described Langfuse self-hosting as "two extra containers", but Langfuse v3 actually requires a heavier stack (separate Langfuse Postgres, ClickHouse, S3-compatible blob storage, web, worker, plus several secrets — five+ containers). Bundling it into Phase 1.2 would have ballooned the slice; the bring-up is now Phase 1.5, after the rest of Epic 1 is in place and before Epic 1 closes.

### Acceptance criteria

- [ ] `docker compose up -d` starts both services without errors
- [ ] `docker compose ps` shows `postgres` and `redis` as healthy
- [ ] `docker compose exec postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"` succeeds
- [ ] `docker compose exec redis redis-cli ping` returns `PONG`
- [ ] `docker compose down -v` removes containers and the `postgres-data` volume cleanly

### Validation

Fresh `docker compose up -d`, wait until healthchecks pass, run the two connectivity checks above, then `docker compose down -v` to confirm clean teardown.

---

## Phase 1.3 — FastAPI app shell, config, Alembic init

**Goal**: A runnable FastAPI app exposing `/health`, a typed config layer reading env + `.env`, and Alembic initialized for future migrations.

### What to build

- **`src/rag_recipes/config.py`** — `pydantic-settings` `Settings` class. Fields (defaults from doc 11):
  - `database_url`, `redis_url`
  - `openai_api_key`, `llm_model`, `embedding_model`, `embedding_dimensions`
  - `langfuse_host`, `langfuse_public_key`, `langfuse_secret_key`, `langfuse_enabled`
  - `personal_api_token` (used in Epic 6)
  - `debug_endpoints_enabled`
  - `local_storage_root`
  - `pdf_window_size_pages`, `pdf_overlap_pages` (default 3 and 1 per doc 3)
  - Search defaults: `search_default_limit`, `search_keyword_top_k`, `search_vector_top_k`, `search_rrf_k`, plus the chunk-type boosts from doc 7
- **`src/rag_recipes/api/app.py`** — FastAPI instance, lifespan context manager initializing SQLAlchemy async engine and Redis client; `/api/v1/health` returns `{"status": "ok"}`
- **`src/rag_recipes/api/dependencies.py`** — DI factories for DB session, Redis client, settings
- **`src/rag_recipes/storage/session.py`** — async SQLAlchemy engine + `sessionmaker` factory; import `pgvector.sqlalchemy.Vector` so future migrations can reference it
- **`alembic.ini`** at repo root
- **`migrations/env.py`** wired for async SQLAlchemy using the `Settings`
- **`migrations/versions/`** empty
- App fails fast with a clear error when required env vars (e.g., `database_url`, `openai_api_key`) are missing

### Acceptance criteria

- [ ] `uv run uvicorn rag_recipes.api.app:app --reload` starts the app
- [ ] `curl http://localhost:8000/api/v1/health` returns `{"status": "ok"}`
- [ ] `uv run alembic current` succeeds (no migrations yet)
- [ ] `uv run alembic revision --autogenerate -m "smoke"` creates an empty revision; delete it after verifying
- [ ] Starting the app with a missing required env var produces a clear, actionable error
- [ ] mypy still passes (the new modules are typed)

### Validation

Run the app, hit `/health`, run `alembic current` — all without errors.

---

## Phase 1.4 — Developer setup (justfile + README quickstart)

**Goal**: A new developer can clone the repo and be running the backend in under five minutes with a single command.

### What to build

- **`justfile`** with recipes:
  - `setup` — `uv sync`; copy `.env.example` to `.env` if missing; `docker compose up -d`; wait for Postgres healthy; `alembic upgrade head`
  - `dev` — runs the FastAPI app with `--reload` and (no-op until Epic 7) the `arq` worker
  - `dev-api` — only the FastAPI server
  - `dev-worker` — only the arq worker (placeholder until Epic 7)
  - `test` — `uv run pytest`
  - `test-unit` — `uv run pytest tests/unit`
  - `test-integration` — `uv run pytest tests/integration`
  - `lint` — `uv run ruff check src/ tests/ && uv run mypy src/`
  - `format` — `uv run ruff format src/ tests/`
  - `migrate` — `uv run alembic upgrade head`
  - `migration MSG` — `uv run alembic revision --autogenerate -m "{{MSG}}"`
  - `down` — `docker compose down`
  - `clean` — `docker compose down -v` (destructive; remove all volumes)
- **`README.md`** quickstart section:
  - Prerequisites (Docker, `uv` or Python 3.12+, `just`, OpenAI API key)
  - Bootstrap: `cp .env.example .env`, fill `OPENAI_API_KEY`, then `just setup`
  - Run: `just dev`
  - Test: `just test`
  - Pointers to `docs/architecture/` and `docs/implementation/EPICS.md`

### Acceptance criteria

- [ ] `just setup` on a fresh clone brings up all services and applies migrations end-to-end
- [ ] `just dev` runs the API at `localhost:8000`
- [ ] `just test`, `just lint`, `just migrate` all succeed
- [ ] `just down` stops services without removing volumes
- [ ] `just clean` removes volumes (with a confirmation prompt if practical)
- [ ] README quickstart is accurate end-to-end (manually verified by re-cloning into a fresh directory)

### Validation

On a fresh clone with an empty `.env`: copy example, set `OPENAI_API_KEY`, run `just setup`, then `just dev`, then `curl localhost:8000/api/v1/health` — all green within five minutes.

---

## Phase 1.5 — Langfuse self-hosted observability stack

**Goal**: Extend the compose stack with the Langfuse v3 self-hosted services so LLM call tracing can be wired up in a later epic. By the end of this phase, the Langfuse UI is reachable locally and an org / project / user / API keys are auto-provisioned on first boot via `LANGFUSE_INIT_*` env vars, so the app-side `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are already valid the moment `.env` is in place.

### Background

Doc 13 §13 originally anticipated Langfuse as "two extra containers", but Langfuse v3 self-hosting requires a heavier stack: a separate Langfuse Postgres (distinct from the app DB), ClickHouse for event analytics, S3-compatible blob storage (MinIO) for large payloads, a worker for background event processing, the web UI, and a handful of secrets (NextAuth, encryption, salt). Phase 1.2 was scoped to the lean app-infra stack (Postgres + Redis) so the rest of Epic 1 could move forward; Phase 1.5 picks the Langfuse work up as its own slice. Consumption of Langfuse (instrumenting LLM calls) is owned by Epic 5.

### What to build

- **Extend `docker-compose.yml`** with the Langfuse v3 services per the [Langfuse self-hosting guide](https://langfuse.com/self-hosting/docker-compose). Image tags follow the repo's major-version pin convention (matching `pgvector/pgvector:pg17` and `redis:7-alpine`):
  - `langfuse-postgres` — Langfuse's own metadata DB on `postgres:16-alpine`, separate from the app `postgres`
  - `langfuse-clickhouse` — event / analytics store on `clickhouse/clickhouse-server:24-alpine`
  - `langfuse-minio` — S3-compatible blob storage on a dated MinIO release tag (e.g. `minio/minio:RELEASE.2025-04-22T22-12-26Z`)
  - `langfuse-worker` — background event-processing worker on `langfuse/langfuse-worker:3`
  - `langfuse-web` — Langfuse UI on `langfuse/langfuse:3`, host port `3001` → container `3000` (avoids the common clash with Next.js / Vite dev servers on `3000`; mirrors the Phase 1.2 `5433 → 5432` Postgres remap precedent)
  - Healthchecks on each stateful service; named volumes for `langfuse-postgres-data`, `langfuse-clickhouse-data`, `langfuse-minio-data`
  - Replace the Phase 1.2 TODO comment with the actual services
- **`.env.example` additions** for the Langfuse infra vars (the app-side `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_ENABLED` already exist from Phase 1.1 with empty values — Phase 1.5 fills them in so they match the seeded project). `.env.example` ships dev-safe values for every Langfuse secret (not empty), continuing the Phase 1.4 `POSTGRES_PASSWORD=postgres` convention; the file makes clear these are local-only dev values:
  - `LANGFUSE_POSTGRES_USER`, `LANGFUSE_POSTGRES_PASSWORD`, `LANGFUSE_POSTGRES_DB`
  - `LANGFUSE_CLICKHOUSE_USER`, `LANGFUSE_CLICKHOUSE_PASSWORD`
  - `LANGFUSE_MINIO_ROOT_USER`, `LANGFUSE_MINIO_ROOT_PASSWORD`
  - `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_ENCRYPTION_KEY` (must be 64-char hex), `LANGFUSE_SALT`
  - **Auto-bootstrap block** — `LANGFUSE_INIT_ORG_ID` / `_NAME`, `LANGFUSE_INIT_PROJECT_ID` / `_NAME` / `_PUBLIC_KEY` / `_SECRET_KEY`, `LANGFUSE_INIT_USER_EMAIL` / `_NAME` / `_PASSWORD`. These pre-create the org / project / user on first boot of an empty Langfuse Postgres; the resulting `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `_SECRET_KEY` feed back into the app-side `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`. The bootstrap is idempotent only on emptiness — rotating these values requires `just clean` to drop `langfuse-postgres-data`.
- **README quickstart addition**: a short note that the seeded admin credentials and auto-provisioned API keys live in `.env`, and that the UI is at `http://localhost:3001`.
- **`just setup` tail echo**: add a one-line tail in `just setup` pointing the developer at `http://localhost:3001` with a hint that the seeded credentials are in `.env`.

### Acceptance criteria

- [ ] `docker compose up -d` brings up the Langfuse services healthy alongside `postgres` and `redis`
- [ ] `docker compose ps` shows `langfuse-postgres`, `langfuse-clickhouse`, `langfuse-minio`, `langfuse-worker`, `langfuse-web` as healthy
- [ ] Langfuse UI is reachable at `http://localhost:3001`
- [ ] An org / project / user / API keys are auto-provisioned on first boot via `LANGFUSE_INIT_*`; `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in `.env.example` match the seeded values
- [ ] `docker compose down -v` cleanly removes all Langfuse containers and named volumes
- [ ] No app-side code changes — the Langfuse client integration is owned by Epic 5

### Validation

Fresh `docker compose up -d`, wait for healthchecks, open `http://localhost:3001`, log in with the seeded admin credentials, confirm the seeded project is present and that the keys in `.env` match the project keys in the UI, then `docker compose down -v` to confirm clean teardown.

---

## Epic-level acceptance criteria

- [ ] All five phases complete and merged
- [ ] Fresh-clone bootstrap to a working `/health` endpoint is a single command sequence (`just setup && just dev`)
- [ ] `just test`, `just lint`, `just migrate` all run green on the empty codebase
- [ ] All services (Postgres+pgvector, Redis, Langfuse) accessible locally
- [ ] No business logic — this is purely infrastructure
- [ ] Status in [`EPICS.md`](../EPICS.md) updated to `Done`; Epic 2 unblocked
