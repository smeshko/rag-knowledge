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

`just setup` syncs dependencies, brings up Postgres + Redis (waits for healthchecks), and applies Alembic migrations.

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
