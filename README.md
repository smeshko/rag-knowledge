# rag-recipes

Hybrid-search RAG service over a personal recipe library, ingested from PDF cookbooks.

## Prerequisites

- Docker (Desktop or Engine) running
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)
- [`just`](https://github.com/casey/just) ≥ 1.17
- An OpenAI API key

## Bootstrap

```sh
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...
just setup
```

`just setup` syncs dependencies, brings up Postgres + Redis + the Langfuse self-hosted stack (waits for healthchecks), applies Alembic migrations, and then submits a Langfuse trace via `scripts/smoke_langfuse.py` to verify the full ingestion path (web → Redis → worker → ClickHouse). The smoke typically adds ~2–5 s on a warm stack and up to ~30 s on cold-cache first boot. First-time `just setup` pulls ~2 GB of images and can take several minutes; the running stack adds ~1 GB of RAM (ClickHouse + Langfuse) on top of the baseline — bump Docker Desktop's memory if needed.

## Local infrastructure

The dev compose stack publishes every service on `127.0.0.1` only — the committed dev credentials (`postgres:postgres`, `redis:redis`, `dev@rag-recipes.local`) are deliberately weak and loopback binding is what keeps them safe.

- `postgres` on `127.0.0.1:5435` (user `postgres`, password `postgres`).
- `redis` on `127.0.0.1:6379` with `requirepass`; the password is `REDIS_PASSWORD` (default `redis`) — it must also be present in the `REDIS_URL` DSN (Settings rejects a credential-less DSN).
- Langfuse UI on `127.0.0.1:3002`; MinIO API / console on `127.0.0.1:9090` / `127.0.0.1:9091`.

When upgrading an existing checkout, add `REDIS_PASSWORD=redis` to your `.env` and update `REDIS_URL` to `redis://:redis@localhost:6379/0`; `docker compose up` fails fast with `refresh your .env from .env.example` if the password is missing.

## Langfuse

The self-hosted Langfuse UI runs at <http://localhost:3002> after `just setup`. Log in with `dev@rag-recipes.local` / `devdevdev` (seeded in `.env`).

API keys are auto-provisioned via the `LANGFUSE_INIT_*` block in `.env`; the existing `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` values already match the seeded project — no copy-paste required.

Re-seeding (e.g. after editing `LANGFUSE_INIT_PROJECT_*` values) requires `just clean` first: `LANGFUSE_INIT_*` only takes effect against an empty Langfuse Postgres.

`just smoke-langfuse` re-runs the bootstrap ingestion check on demand — it's the easiest way to debug a misconfigured stack without re-running the full `just setup`.

## Run

```sh
just dev
```

The API is gated by a personal bearer token and **fails closed**: when `PERSONAL_API_TOKEN` is empty, every request — including `/api/v1/health` — returns `401 unauthorized`. Set the variable in `.env` and send it as a header to use the API locally:

```sh
curl -H "Authorization: Bearer $PERSONAL_API_TOKEN" http://localhost:8004/api/v1/health
# {"status":"ok"}
curl -H "Authorization: Bearer $PERSONAL_API_TOKEN" http://localhost:8004/api/v1/documents
```

## LLM providers

`LLM_PROVIDER` selects the extraction/answer backend through the provider registry: `openai` (default), `anthropic`, `deepseek`, or `claude_cli`. Each reads its own model and key fields — see the matching block in `.env.example`.

`claude_cli` is the odd one out: it shells out to the local `claude` binary in `-p` mode, so extraction draws your claude.ai subscription quota instead of API billing.

- **Auth is `claude /login`, not a key.** Whoever runs the worker must have a logged-in CLI on `PATH` (or set `CLAUDE_CLI_BINARY`). `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` are stripped from the subprocess env so an exported key can never silently bill these calls to the API.
- **Pin an exact model.** `CLAUDE_CLI_MODEL` is part of the extraction cache key; an alias like `opus` that drifts to a new version invalidates every cached window.
- **Windows extract sequentially**, roughly 35 s each on `claude-opus-5`, against about `pages / 2` windows per book. Budget hours per cookbook and keep `DOCUMENT_JOB_TIMEOUT_SECONDS` above that; the per-batch heartbeat keeps `sweep_stuck_jobs` from reaping a healthy long run.
- **Quota exhaustion fails the document** rather than retrying — arq only re-drives on cancellation. Resume with `POST /documents/{id}/reprocess`; the extraction cache means already-extracted windows are not re-sent.
- **`POST /documents/batch` returns 409** under this provider — the batch path is Anthropic-only. Ingest through `POST /documents`.

## Test

```sh
just test         # full suite
just test-unit    # unit tests only
just lint         # ruff + mypy
```

## Useful recipes

`just --list` shows every available recipe. Highlights:

- `just migrate` — apply pending Alembic migrations
- `just migration "describe change"` — autogenerate a new revision
- `just down` — stop the compose stack (volumes preserved)
- `just clean` — stop the stack **and** delete data volumes (prompts before destroying data)

## Further reading

- Architecture overview: [`docs/architecture/01-big-picture.md`](docs/architecture/01-big-picture.md)
- Epic-by-epic implementation plan: [`docs/implementation/EPICS.md`](docs/implementation/EPICS.md)
