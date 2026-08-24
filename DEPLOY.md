# Deploying Stove to recipes.ivot.dev

Home-server deployment: the whole app is published through the existing
`adw-webhook` cloudflared tunnel as `https://recipes.ivot.dev`, gated by
Cloudflare Access. **The backend is not exposed** — it is reachable only via
Caddy, on loopback.

## Shape

```
browser
  │  https://recipes.ivot.dev
  ▼
Cloudflare edge ── Access policy (email allow-list) ── denies anonymous requests here
  │
  ▼  tunnel: adw-webhook
cloudflared (this machine, ~/.cloudflared/config.yml)
  │  http://127.0.0.1:8090      ← the ONLY origin for this hostname
  ▼
Caddy (deploy/Caddyfile, bound to 127.0.0.1)
  ├── /api/*  → 127.0.0.1:8004   + Authorization: Bearer <token> injected here
  └── /*      → rag-recipes-fe/dist  (SPA, index.html fallback)
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

## Ports on this machine

| Port | Owner | Exposed? |
|---|---|---|
| 8090 | Caddy (recipes front) | yes — the tunnel origin for `recipes.ivot.dev` |
| 8004 | rag-recipes API (uvicorn) | no — loopback only, absent from the tunnel ingress |
| 5435 | recipes Postgres | no |
| 6379 | recipes Redis | no |
| 3002 / 9090 / 9091 | Langfuse, MinIO (opt-in) | no |

Two collisions were designed around, both live:

- **8001** is `webhook.ivot.dev` in the tunnel (adw). Anything listening there
  is published to the internet with no Access in front, so the API moved to
  **8004**.
- **5433** is `life-organizer-db`, started earlier by `start-services.sh`, so
  the recipes Postgres moved to **5435**.

## Secrets

One secret, in two places that must agree:

- `PERSONAL_API_TOKEN` in `rag-knowledge/.env` — what the API checks.
- login keychain item `rag-recipes-token` — what `start-services.sh` reads and
  hands to Caddy as `RAG_RECIPES_TOKEN`.

Rotate both together:

```sh
TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
security add-generic-password -a "$USER" -s rag-recipes-token -w "$TOKEN" -U
# then set PERSONAL_API_TOKEN=$TOKEN in rag-knowledge/.env and restart both
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
kill "$(cat ~/Developer/logs/recipes-be.pid)"
~/Developer/start-services.sh            # idempotent; restarts only what is down
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
- **100 s origin timeout** (error 524). `/answers` is a synchronous LLM call;
  a slow generation can trip this. Ingestion is unaffected — uploads return
  immediately and the arq worker does the extraction out of band.
- Access sessions are per-application; the policy on `recipes.ivot.dev` does
  not affect `webhook.ivot.dev`, `ssh.ivot.dev` or the other hostnames.

## If it breaks

| Symptom | Cause |
|---|---|
| 502 from the edge | Caddy is down — `tail ~/Developer/logs/recipes-caddy.log` |
| App loads, every API call 401 | Caddy started without `RAG_RECIPES_TOKEN`, or it drifted from `.env` |
| 400 on every request | Caddy site address regressed to a host-matching form; it must stay `:8090` + `bind 127.0.0.1`, because cloudflared forwards `Host: recipes.ivot.dev` |
| Uploads never finish | arq worker is down — `tail ~/Developer/logs/recipes-worker.log` |
| Whole site unreachable | laptop asleep or off the network; `caffeinate` only covers sleep |
