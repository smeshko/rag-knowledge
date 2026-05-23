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

`just setup` syncs dependencies, brings up Postgres + Redis + the Langfuse self-hosted stack (waits for healthchecks), and applies Alembic migrations. First-time `just setup` pulls ~2 GB of images and can take several minutes; the running stack adds ~1 GB of RAM (ClickHouse + Langfuse) on top of the baseline — bump Docker Desktop's memory if needed.

## Langfuse

The self-hosted Langfuse UI runs at <http://localhost:3001> after `just setup`. Log in with `dev@rag-recipes.local` / `devdevdev` (seeded in `.env`).

API keys are auto-provisioned via the `LANGFUSE_INIT_*` block in `.env`; the existing `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` values already match the seeded project — no copy-paste required.

Re-seeding (e.g. after editing `LANGFUSE_INIT_PROJECT_*` values) requires `just clean` first: `LANGFUSE_INIT_*` only takes effect against an empty Langfuse Postgres.

## Run

```sh
just dev
```

Then hit the health endpoint:

```sh
curl http://localhost:8000/api/v1/health
# {"status":"ok"}
```

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
