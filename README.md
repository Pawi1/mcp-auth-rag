# MCP Auth Starter

> **mcp 2.0.** `main` now targets the mcp 2.0 `Server` API (constructor-based
> `on_list_tools`/`on_call_tool` handlers instead of the old
> `@server.list_tools()`/`@server.call_tool()` decorators). If you need the
> mcp 1.x-pinned version, check out the
> [`legacy`](https://github.com/Pawi1/mcp-auth-starter/tree/legacy) branch.
> Migrating your own fork? See [MIGRATING.md](MIGRATING.md).

[![Tests](https://github.com/Pawi1/mcp-auth-starter/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Pawi1/mcp-auth-starter/actions/workflows/tests.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A minimal, working example of the part of building an MCP server that's
annoying to get right: **OAuth 2.0 with Client ID Metadata Documents (and
Dynamic Client Registration, RFC 7591, as a fallback) + JWT bearer tokens,
served over Streamable HTTP** — so Claude.ai (or any other OAuth-aware MCP
client) can add your server as a connector with a normal browser login. No
manual token pasting, no bypassing OAuth with a "just give me a token"
shortcut that quietly stops working the moment you add real revocation.

This is *not* a framework — it's ~1200 lines of plain Starlette you're meant
to read, fork, and build on. There's exactly one demo tool (`whoami`) to
prove the auth chain works end to end. Your actual tools go in `app/server.py`.

## What's in here

| File | What it does |
|---|---|
| `app/main.py` | Starlette app, `/mcp` endpoint + auth gate, lifespan, CLI (`--setup`, `--adduser`) |
| `app/oauth.py` | Full OAuth 2.0 flow: discovery, client registration (CIMD + DCR), authorize/login/token (+ refresh grant with rotation), revocation, rate limiting |
| `app/auth.py` | JWT verification (signature, expiry, audience) |
| `app/users.py` | User accounts (argon2 password hashing), login attempt + tool-call audit logging |
| `app/context.py` | `ContextVar` carrying the authenticated user into your tool handlers |
| `app/server.py` | MCP tool definitions — `whoami`, `rag_search`, `rag_list_documents` |
| `app/config.py` | Config loader (`config.json` + env var overrides for secrets) |
| `app/rag_ingest.py` | PDF/DOCX/TXT/MD parsing + hierarchical chunking |
| `app/rag_embed.py` | Embedding + reranking client (Ollama, OpenAI-compatible) |
| `app/rag_store.py` | Postgres (metadata, full-text search, job queue) + Qdrant (vectors) |
| `app/rag_retrieval.py` | Hybrid search: Qdrant ANN + Postgres FTS, fused (RRF), reranked |
| `app/rag_worker.py` | Background ingest worker (polls the Postgres job queue) |
| `app/rag_routes.py` | `/rag` web panel — upload, document list, search |

## Why the auth gate is two checks, not one

`main.py`'s `/mcp` handler checks that the bearer token (1) has a valid JWT
signature, **and** (2) still exists in the `oauth_tokens` table. Both matter:
a token can have a perfectly valid signature and still not be a real,
currently-issued session — e.g. after you call `revoke_tokens_for_user()`,
the JWT itself doesn't change, but it's deleted from the DB, so it correctly
stops working. Skip the second check and revocation silently does nothing.

## Token lifetimes and audience binding

The access token you send as `Authorization: Bearer` is short-lived (60 min
default, `auth.access_token_expire_minutes`) — that's the one that ends up
in logs/proxies, so it's the one worth keeping short-lived. A long-lived,
opaque `refresh_token` (90 days default, `auth.token_expire_days`, DB-backed
in `refresh_tokens`, nothing to decode) exchanges for a new pair via
`grant_type=refresh_token` without the user logging in again. Refresh tokens
rotate on every use (OAuth 2.1 §4.3.1) — each one is single-use, and using
one issues a fresh replacement while invalidating the old one, so a copied
refresh token is only useful until its legitimate owner's next refresh.

The access token also carries an `aud` claim set to this server's canonical
URI, and `/oauth/authorize`/`/oauth/token` validate an optional `resource`
parameter (RFC 8707) against it — so a token minted here can't be replayed
against a different resource server even if it somehow shared your
`SECRET_KEY`.

## Client registration: CIMD, then DCR

Per the [2026-07-28 MCP authorization spec](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/client-registration),
Dynamic Client Registration (RFC 7591) is deprecated in favor of **Client ID
Metadata Documents** (CIMD, `draft-ietf-oauth-client-id-metadata-document`):
a client identifies itself with an `https://` URL instead of registering
up front, and this server fetches a small JSON document (capped at 8 KiB,
per the draft's ~5 KiB recommendation) from that URL on first use
(`client_id`, `client_name`, `redirect_uris` — see `_fetch_cimd_metadata`
in `app/oauth.py`), caching it per `Cache-Control`. The `client_id` URL
itself is validated against the draft's format rules (`https`, a path, no
fragment/userinfo/`.`/`..` segments), and a document claiming a
shared-secret `token_endpoint_auth_method` (`client_secret_post` etc.) is
rejected outright — a "secret" published in a document anyone can fetch
isn't one. CIMD clients are otherwise public (no `client_secret` — PKCE is
what proves possession), and there's a basic SSRF guard on the fetch
(rejects loopback/private/link-local targets; doesn't defend against DNS
rebinding, see [SECURITY.md](SECURITY.md)). The consent page shows the
`client_id`'s hostname alongside the self-reported `client_name`, since
the hostname is harder to fake.

`/oauth/clients/register` (DCR) is still there, for clients that don't
speak CIMD yet — it now also accepts `application_type` (`"web"` or
`"native"`, SEP-837) and echoes it back. This server isn't an OIDC
provider, so it doesn't enforce anything from it (a `"web"` client can
still register a `localhost` redirect_uri) — it's stored and returned
purely so clients that send it, as the spec now requires, get a clean
registration instead of the field being silently dropped.

Authorization responses also now carry an `iss`
parameter (RFC 9207), so a client talking to more than one authorization
server can tell them apart.

## Quick start

```bash
make dev          # creates app/.venv, installs deps, runs directly (no build)
```

On first run you'll get a "Config not found" error — run the setup wizard
first (creates `config.json`, a `SECRET_KEY`, and an admin user):

```bash
cd app && python3 main.py --setup
```

Claude.ai requires a public HTTPS URL for custom connectors — it won't
accept `http://localhost:8000/mcp` directly. For local testing, expose
your server with a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/)
(`cloudflared tunnel --url http://localhost:8000`) and use the `https://`
URL it gives you.

Then add this server as a connector in Claude.ai (Settings → Connectors →
Add custom connector) with that URL plus `/mcp`. Claude.ai will discover
the OAuth endpoints automatically, register itself as a client, and
prompt you to log in with the admin account you just created.

## Adding your own tools

Edit `app/server.py`: add one `Tool(...)` entry to `list_tools()` and a
matching `if name == "...":` branch in `call_tool()`. `current_user.get()`
is always populated by the time `call_tool()` runs — the auth gate rejects
the request before it reaches here otherwise. Call `log_tool_call(user["username"], name)`
in your branch too, so "who ran this tool, and when" stays a query against
`tool_call_log` (see `app/users.py`) instead of a grep through log files
after the fact.

```python
if name == "my_tool":
    log_tool_call(user["username"], name)
    return _ok({"result": do_something(arguments)})
```

## RAG — document upload, search, and MCP tools

Upload PDFs, DOCX, TXT, or Markdown through the `/rag` web panel; the server
parses, chunks, and embeds them in the background, and both the panel and
Claude (via `rag_search`) can search the result. There's no generation model
in this server — retrieval only. The MCP tools hand chunks + citations back
to whatever client is connected, and *that* client's own model writes the
answer. That split keeps this repo's job to what it's actually good at
(auth, transport, retrieval) instead of half-building a second LLM
integration next to whatever the client already has.

### Architecture

```
Upload (panel) → Postgres (documents row, status=pending) → ingest_jobs queue
                                                                    │
                                            rag_worker.py polls (SKIP LOCKED)
                                                                    │
                     rag_ingest.py: parse → hierarchical chunk → rag_embed.py
                                                                    │
                             rag_store.py: chunk text → Postgres, vectors → Qdrant

Search (panel or rag_search tool)
  query → embed → [Qdrant ANN, Postgres full-text] → RRF fusion → rerank → top_k
```

- **Chunking** (`rag_ingest.py`) never crosses a section boundary — PDF
  headings are detected by font size relative to the document's body text,
  DOCX by paragraph style, Markdown by `#` headers. Within a section, text is
  split into ~350-word windows with a 40-word overlap. Every chunk is
  embedded with a `[filename > section]` prefix, so it's not embedded in
  isolation from what document/section it came from.
- **Storage** is Postgres (chunk text, full-text search via `tsvector`, the
  `documents`/`ingest_jobs` tables) + Qdrant (vectors only, with just enough
  payload — `owner`, `document_id` — to filter). Both are separate from
  `app.db` (the auth tables) — very different scale and access pattern, no
  reason to share a database. See `docker-compose.rag.yml`.
- **Retrieval** (`rag_retrieval.py`) fuses Qdrant's dense ANN search with
  Postgres full-text search by Reciprocal Rank Fusion (rank position, not raw
  score — the two live on incomparable scales), then optionally reranks the
  fused candidates with a cross-encoder (`bge-reranker-v2-m3`) for a final
  precision pass. Reranking is opportunistic: if that model isn't deployed on
  your Ollama yet, `rag_embed.rerank()` fails fast and search falls back to
  the fusion order — no config change needed once you do pull it.
- **Ingest is async** — a 150-page PDF's worth of chunking + embedding
  doesn't block the upload request. `rag_worker.py` polls `ingest_jobs` with
  `FOR UPDATE SKIP LOCKED`, so it's also safe to run more than one worker
  process against the same Postgres later without any code change.

### Quick start

```bash
make rag-up   # starts Postgres + Qdrant (docker-compose.rag.yml)
make dev      # same as before — the app now also serves /rag
```

Add to `config.json` (see `services/config.example.json` for the full
shape):

```json
"rag": {
  "postgres": { "dsn": "postgresql://rag:rag@localhost:5433/rag" },
  "qdrant":   { "url": "http://localhost:6333", "collection": "rag_chunks" },
  "embedding": { "base_url": "https://your-ollama-host/v1", "model": "bge-m3", "dimensions": 1024 },
  "reranker":  { "base_url": "https://your-ollama-host", "model": "bge-reranker-v2-m3", "enabled": true }
}
```

`embedding.base_url` is an OpenAI-compatible endpoint (`POST {base_url}/embeddings`);
`reranker.base_url` is native Ollama (`POST {base_url}/api/rerank` — there's
no OpenAI-spec equivalent). If your Ollama needs a bearer token, set
`RAG_OLLAMA_API_KEY` as an env var — never put it in `config.json`.

Sign in at `/rag` with the same admin account created by `--setup`. That
session is a separate cookie (`rag_session`, `SameSite=Strict`) from the
OAuth bearer tokens MCP clients use against `/mcp` — the panel is for a
human in a browser, not an MCP client.

### MCP tools

- **`rag_search(query, top_k?)`** — hybrid search + rerank, returns chunks
  with `filename`/`page`/`section` citations.
- **`rag_list_documents()`** — what's been uploaded and its ingest status,
  so the caller can check before searching.

### Known tradeoffs

- Uploaded files live on local disk (`paths.data_root/rag_uploads`), keyed by
  document ID. Fine for one process; if you ever run `rag_worker.py` across
  multiple pods, that needs shared storage (a volume, or S3/MinIO) — same
  category of tradeoff as the in-memory caches called out in
  [SECURITY.md](SECURITY.md).
- Upload size is capped by `rag.max_upload_mb` (default 200), checked against
  `Content-Length` before reading — a client that lies about that header
  isn't stopped by this check alone.
- No LLM-generated **Contextual Retrieval** (Anthropic's technique of having
  a generation model write a bespoke situating sentence per chunk) — the
  `[filename > section]` prefix is a free, deterministic stand-in that gets
  some of the same benefit. Adding real contextual retrieval means wiring a
  generation-model call into `rag_ingest.py`, which this repo doesn't do.

## Testing

```bash
make test
```

277 tests, ~76% line coverage (`pytest --cov=app`). `app/config.py` and the
interactive CLI wizard (`--setup`/`--adduser`) are the main gaps — they're
either constants or `input()`-driven, both low value to unit test.
`rag_store.py`'s Postgres/Qdrant queries are the other big one: this suite
mocks `rag_store` at the call boundary everywhere else (`rag_ingest`,
`rag_retrieval`, `rag_worker`, `rag_routes`), deliberately not the queries
themselves — asserting a mocked driver got called with the SQL string you
wrote mostly just re-asserts the SQL string you wrote. Exercise those against
`make rag-up` locally instead.

## Deployment

```bash
make build-binary   # PyInstaller single-file binary
sudo make install   # installs + registers a systemd service
sudo make start
```

See `services/` for the systemd unit and env file template. The RAG feature
needs Postgres + Qdrant reachable wherever the binary runs — `make rag-up`
(`docker-compose.rag.yml`) for a single host, or your own Postgres/Qdrant
deployment (e.g. CloudNativePG + the Qdrant Helm chart) if you're on
Kubernetes.

## What this deliberately leaves out

- **Multi-tenancy.** The JWT carries a `teams` claim and users have a
  `teams` column, but there's no tenant table or access-control gate built
  on top of it — most single-purpose MCP servers don't need one, and a
  half-built example is worse than none. If you need it, gate your tools on
  `current_user.get()["teams"]` yourself.
- **A skills/plugin system**, business logic, or any domain-specific
  tools — that's the whole point of `server.py` being ~60 lines.

## Security

Auth/transport code, so bugs here are security bugs. See
[SECURITY.md](SECURITY.md) for reporting a vulnerability, scope, and known
tradeoffs (in-memory token/rate-limit/auth-code caches don't scale across
processes, etc.).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Origin

This started as the auth/transport layer of a larger, private production
MCP server I maintain — extracted, genericized, and stripped of every bit
of that server's domain-specific logic. What's here is just the part that's
generically useful to anyone standing up their own MCP server: a working
OAuth 2.0 + JWT implementation you don't have to build from scratch.

## License

MIT — see [LICENSE](LICENSE).
