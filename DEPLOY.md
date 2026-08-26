# Deploying Stove to recipes.ivot.dev

Home-server deployment: the whole app is published through a cloudflared
tunnel as `https://recipes.ivot.dev`, gated by Cloudflare Access. **The
backend is not exposed** — it is reachable only via Caddy, on loopback.

## Shape

```
browser
  │  https://recipes.ivot.dev
  ▼
Cloudflare edge ── Access policy (email allow-list) ── denies anonymous requests here
  │
  ▼  cloudflared tunnel (this machine)
  │  http://127.0.0.1:8090      ← the ONLY origin for this hostname
  ▼
Caddy (deploy/Caddyfile, bound to 127.0.0.1)
  ├── /api/*  → 127.0.0.1:8004   + Authorization: Bearer <token> injected here
  └── /*      → the frontend's dist/  (SPA, index.html fallback)
                     │
                     ▼
              uvicorn :8004  ─ postgres 127.0.0.1:5435
              arq worker     ─ redis    127.0.0.1:6379
```

Why a single hostname: the frontend and the API share an origin, so there is
no CORS and no preflight, and Access's redirect-to-login never breaks an XHR
(it would, across origins — the browser sees an opaque CORS failure instead of
a 302). It also means the token is injected server-side and never ships in the
bundle.

## Ports

| Port | Owner | Exposed? |
|---|---|---|
| 8090 | Caddy (front) | yes — the tunnel origin for `recipes.ivot.dev` |
| 8004 | API (uvicorn) | no — loopback only, absent from the tunnel ingress |
| 5435 | Postgres | no |
| 6379 | Redis | no |
| 3002 / 9090 / 9091 | Langfuse, MinIO (opt-in) | no |

Make sure nothing else on the machine is bound to these before starting. In
particular, **never put the API's port in the tunnel ingress** — anything
routed there is published without Access in front, and the API's own bearer
check is the only thing behind it.

## Files

- `deploy/Caddyfile` — the tunnel origin. Machine-specific paths come from the
  environment: `RAG_RECIPES_FE_DIST` (the frontend build) and
  `RAG_RECIPES_LOG_DIR` (its log). Nothing in the file is host-specific.
- `deploy/start-recipes.sh` — starts (or confirms) postgres + redis, the API,
  the worker and Caddy; idempotent; writes `recipes-{be,worker,caddy}.{log,pid}`
  into `RAG_RECIPES_LOG_DIR`. Its header lists every variable it reads and
  the defaults.

## Secrets

One secret, in two places that must agree:

- `PERSONAL_API_TOKEN` in `rag-knowledge/.env` — what the API checks.
- `RAG_RECIPES_TOKEN` in Caddy's environment — what it injects.
  `start-recipes.sh` takes it from the environment, or falls back to the macOS
  login keychain item `rag-recipes-token`.

Rotate both together:

```sh
TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
security add-generic-password -a "$USER" -s rag-recipes-token -w "$TOKEN" -U
# then set PERSONAL_API_TOKEN=$TOKEN in rag-knowledge/.env and restart the API + Caddy
```

For local development the frontend's Vite proxy plays Caddy's role, injecting
`$RAG_RECIPES_TOKEN` from the shell. Export it before `just dev`:

```sh
export RAG_RECIPES_TOKEN="$(security find-generic-password -a "$USER" -s rag-recipes-token -w)"
```

## Deploying a change

```sh
# frontend
cd rag-recipes-fe && just build          # Caddy serves dist/ directly; no restart needed

# backend
kill "$(cat "${RAG_RECIPES_LOG_DIR:-$HOME/Developer/logs}/recipes-be.pid")"
rag-knowledge/deploy/start-recipes.sh    # idempotent; restarts only what is down
```

Migrations: `cd rag-knowledge && uv run alembic upgrade head`.

## Cloudflare-side limits worth knowing

- **100 MB upload cap** on the free plan. A cookbook PDF above that is
  rejected at the edge, before it reaches Caddy, as HTTP 413. This only binds
  when you upload *through the tunnel*. `POST /api/v1/documents` is the only
  ingestion entry point in the codebase (there is no CLI ingest command), but
  the API runs on the same machine as the PDFs, so a bulk or oversized import
  can go straight to loopback and skip Cloudflare entirely:

  ```sh
  TOKEN=$(security find-generic-password -a "$USER" -s rag-recipes-token -w)
  curl -H "Authorization: Bearer $TOKEN" \
    -F "file=@/path/to/cookbook.pdf" \
    -F "title=Cookbook title" -F "author=Author" \
    http://127.0.0.1:8004/api/v1/documents
  ```

  The dropzone in the UI is the convenience path for when you are away from
  home, not the only way in.
- **100 s origin timeout** (error 524). `/answers` and `/menus` are
  synchronous LLM calls; a slow generation can trip this. Ingestion is
  unaffected — uploads return immediately and the arq worker does the
  extraction out of band.
- Access sessions are per-application: the policy on this hostname affects
  only this hostname.

## If it breaks

| Symptom | Cause |
|---|---|
| 502 from the edge | Caddy is down — `tail $RAG_RECIPES_LOG_DIR/recipes-caddy.log` |
| App loads, every API call 401 | Caddy started without `RAG_RECIPES_TOKEN`, or it drifted from `.env` |
| 400 on every request | Caddy site address regressed to a host-matching form; it must stay `:8090` + `bind 127.0.0.1`, because cloudflared forwards `Host: recipes.ivot.dev` |
| Uploads never finish | arq worker is down — `tail $RAG_RECIPES_LOG_DIR/recipes-worker.log` |
| Whole site unreachable | machine asleep or off the network; `caffeinate` only covers sleep |
