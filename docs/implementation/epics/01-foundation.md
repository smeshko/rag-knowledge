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

## Phase 1.2 — Docker Compose stack

**Goal**: A single `docker compose up -d` brings up all services required for local development.

### What to build

- **`docker-compose.yml`** with services:
  - `postgres` — image `pgvector/pgvector:pg17`, port `5432`, named volume, healthcheck on `pg_isready`, env vars for user/password/db (dev credentials, NOT production secrets)
  - `redis` — image `redis:7-alpine`, port `6379`, healthcheck on `redis-cli ping`
  - `langfuse-postgres`, `langfuse-clickhouse`, `langfuse-web` — per the [Langfuse self-hosting guide](https://langfuse.com/self-hosting/docker-compose); Langfuse UI exposed on port `3000`
  - Named volumes for all stateful services
- **`.dockerignore`** to keep build context small
- Dev credentials live in the compose file or a separate `.env.docker` — distinct from the app's `.env`

### Acceptance criteria

- [ ] `docker compose up -d` starts all services without errors
- [ ] `docker compose ps` shows all services as healthy
- [ ] `psql -h localhost -p 5432 -U postgres -d postgres` from the host succeeds
- [ ] `redis-cli -h localhost ping` returns `PONG`
- [ ] Langfuse UI is accessible at `http://localhost:3000`
- [ ] `docker compose down -v` removes containers and volumes cleanly

### Validation

Fresh `docker compose up -d`, wait until healthchecks pass, run the three connectivity checks above, then `docker compose down -v` to confirm clean teardown.

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

## Phase 1.4 — Developer setup (Makefile + README quickstart)

**Goal**: A new developer can clone the repo and be running the backend in under five minutes with a single command.

### What to build

- **`Makefile`** with targets (use `.PHONY` for all):
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
  - `migration MSG="..."` — `uv run alembic revision --autogenerate -m "$(MSG)"`
  - `down` — `docker compose down`
  - `clean` — `docker compose down -v` (destructive; remove all volumes)
- **`README.md`** quickstart section:
  - Prerequisites (Docker, `uv` or Python 3.12+, OpenAI API key)
  - Bootstrap: `cp .env.example .env`, fill `OPENAI_API_KEY`, then `make setup`
  - Run: `make dev`
  - Test: `make test`
  - Pointers to `docs/architecture/` and `docs/implementation/EPICS.md`

### Acceptance criteria

- [ ] `make setup` on a fresh clone brings up all services and applies migrations end-to-end
- [ ] `make dev` runs the API at `localhost:8000`
- [ ] `make test`, `make lint`, `make migrate` all succeed
- [ ] `make down` stops services without removing volumes
- [ ] `make clean` removes volumes (with a confirmation prompt if practical)
- [ ] README quickstart is accurate end-to-end (manually verified by re-cloning into a fresh directory)

### Validation

On a fresh clone with an empty `.env`: copy example, set `OPENAI_API_KEY`, run `make setup`, then `make dev`, then `curl localhost:8000/api/v1/health` — all green within five minutes.

---

## Epic-level acceptance criteria

- [ ] All four phases complete and merged
- [ ] Fresh-clone bootstrap to a working `/health` endpoint is a single command sequence (`make setup && make dev`)
- [ ] `make test`, `make lint`, `make migrate` all run green on the empty codebase
- [ ] All services (Postgres+pgvector, Redis, Langfuse) accessible locally
- [ ] No business logic — this is purely infrastructure
- [ ] Status in [`EPICS.md`](../EPICS.md) updated to `Done`; Epic 2 unblocked
