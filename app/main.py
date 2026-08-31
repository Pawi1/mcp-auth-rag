#!/usr/bin/env python3
"""
MCP Auth Starter — Streamable HTTP Transport + OAuth 2.0 (Dynamic Client
Registration) + JWT bearer tokens.

This is the whole point of the repo: a minimal, working example of the auth
and transport plumbing an MCP server needs to be added as a Claude.ai (or
any OAuth-aware MCP client) connector with a normal browser login — no
manual token pasting, no bypassing OAuth. Your actual tools live in
server.py; everything here is generic.
"""

import asyncio
import contextlib
import logging
import sys
import urllib.parse
from collections.abc import AsyncIterator

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from config import LOG_FILE, MCP_HOST, MCP_PORT, MCP_SCOPE, MCP_SERVER_NAME, SERVER_URL
from oauth import (
    _ensure_tokens_table, cleanup_expired_tokens, load_tokens_from_db,
    load_clients_from_db, oauth_authorize, oauth_login, oauth_login_post,
    oauth_metadata, oauth_protected_resource, oauth_clients_register,
    oauth_token, close_http_client, get_http_client, sweep_expired_state,
)
from server import mcp_server
from users import _ensure_db_schema, prune_login_alerts


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stderr),
    ],
)

logger = logging.getLogger("mcp-auth-starter")

session_manager = StreamableHTTPSessionManager(app=mcp_server, stateless=False)


class _NullResponse:
    """Starlette expects a callable response; session_manager already sent everything."""
    async def __call__(self, scope, receive, send):
        pass


# Matches the width of the column a downstream service is likely to store
# this in, and keeps a hostile value from being unbounded either way.
MAX_ACTOR_LEN = 150


def _requested_actor(request: Request) -> str:
    """The end user a service client says it is acting for, from X-MCP-Actor.

    Percent-decoded, because the header must survive a non-ASCII username and
    an HTTP header cannot carry one raw. Non-printable characters are dropped
    rather than the value rejected: this is informational, so a mangled name
    is worth recording as best it can be read, while a newline in it could
    split a log line or a downstream header.
    """
    raw = request.headers.get("X-MCP-Actor", "")
    if not raw:
        return ""
    value = urllib.parse.unquote(raw)
    return "".join(ch for ch in value if ch.isprintable())[:MAX_ACTOR_LEN].strip()


async def handle_mcp(request: Request):
    """POST/GET/DELETE /mcp — the actual MCP protocol endpoint.

    Every request must carry a bearer token (Authorization header) that is
    (a) a validly-signed JWT and (b) still present in the oauth_tokens table
    (so revocation actually revokes). Both checks matter: a token that only
    passes (a) but was never issued through OAuth is a forged/stale token,
    not a legitimate session.
    """
    from auth import verify_token
    from context import current_user
    from oauth import is_token_active

    scheme, _, raw_token = request.headers.get("Authorization", "").partition(" ")
    raw_token = raw_token.strip()

    if scheme.lower() != "bearer" or not raw_token:
        logger.info(f"MCP {request.method} 401 (no bearer token) from {request.client}")
        return Response(status_code=401, headers={
            "WWW-Authenticate": (
                f'Bearer realm="{SERVER_URL}/mcp",'
                f' resource_metadata="{SERVER_URL}/.well-known/oauth-protected-resource",'
                f' scope="{MCP_SCOPE}"'
            )
        })

    try:
        user = await verify_token(raw_token)
    except Exception:
        current_user.set(None)
        logger.warning(f"MCP {request.method} invalid token from {request.client}")
        return Response(status_code=401, headers={"WWW-Authenticate": 'Bearer error="invalid_token"'})

    if not is_token_active(raw_token):
        current_user.set(None)
        logger.warning(f"MCP {request.method} revoked token from {request.client}")
        return Response(status_code=401, headers={"WWW-Authenticate": 'Bearer error="invalid_token"'})

    # A service token (one minted by client_credentials, carrying `svc`) may
    # name the person it is acting for; a user token may not, and the claim's
    # absence is the whole check. Without this, anything reached through a
    # proxy would be recorded as the proxy's own machine account, and the
    # audit trail would answer "which service wrote this" instead of "who
    # asked for it" — the question it exists for. With it, a caller holding
    # an ordinary token cannot present itself as somebody else, because its
    # token cannot carry the claim that would let it.
    if user.get("svc"):
        on_behalf_of = _requested_actor(request)
        if on_behalf_of:
            user = {**user, "on_behalf_of": on_behalf_of}
            logger.info(
                f"MCP {request.method} svc={user['svc']} on behalf of {on_behalf_of} from {request.client}"
            )
    current_user.set(user)
    if not user.get("on_behalf_of"):
        logger.info(f"MCP {request.method} user={user['username']} from {request.client}")

    await session_manager.handle_request(request.scope, request.receive, request._send)
    return _NullResponse()


CLEANUP_INTERVAL = 60


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        try:
            sweep_expired_state()
            cleanup_expired_tokens()
            prune_login_alerts()
        except Exception:
            logger.exception("Cleanup pass failed")


@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    _ensure_db_schema()
    _ensure_tokens_table()
    load_tokens_from_db()
    load_clients_from_db()
    get_http_client()
    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        async with session_manager.run():
            logger.info("StreamableHTTP session manager running")
            yield
    finally:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task
        await close_http_client()


app = Starlette(
    lifespan=lifespan,
    routes=[
        # OAuth 2.0 discovery
        Route("/.well-known/oauth-protected-resource",     endpoint=oauth_protected_resource),
        Route("/.well-known/oauth-protected-resource/mcp", endpoint=oauth_protected_resource),
        Route("/.well-known/oauth-authorization-server",   endpoint=oauth_metadata),
        Route("/oauth/clients/register", endpoint=oauth_clients_register, methods=["POST"]),
        Route("/oauth/authorize",  endpoint=oauth_authorize),
        Route("/oauth/login",      endpoint=oauth_login,      methods=["GET"]),
        Route("/oauth/login",      endpoint=oauth_login_post, methods=["POST"]),
        Route("/oauth/token",      endpoint=oauth_token,      methods=["POST"]),
        # MCP
        Route("/mcp", endpoint=handle_mcp, methods=["GET", "POST", "DELETE"]),
        # Health
        Route("/health", endpoint=lambda r: JSONResponse({"status": "ok", "server": MCP_SERVER_NAME})),
    ],
)


# ============================================================================
# CLI — first-time setup and user management
# ============================================================================

def _getpass_stars(prompt="Password: ") -> str:
    import termios
    import tty
    print(prompt, end="", flush=True)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    pw = ""
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                break
            elif ch == "\x03":
                raise KeyboardInterrupt
            elif ch in ("\x7f", "\x08"):  # backspace
                if pw:
                    pw = pw[:-1]
                    print("\b \b", end="", flush=True)
            else:
                pw += ch
                print("*", end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print()
    return pw


def run_setup_wizard():
    """`python -m app.main --setup` — interactive first-time config bootstrap."""
    import json
    import secrets as _secrets
    from pathlib import Path as _Path

    print("\nMCP Auth Starter — first-time setup")
    print("=" * 40)

    def ask(prompt, default=""):
        suffix = f" [{default}]" if default else ""
        val = input(f"{prompt}{suffix}: ").strip()
        return val if val else default

    server_url = ask("Server URL",   "http://localhost:8000")
    data_dir   = ask("Data directory", "/srv/mcp-auth-starter")
    log_dir    = ask("Log directory",  "/var/log/mcp-auth-starter")

    cfg = {
        "paths":  {"data_root": data_dir, "log_dir": log_dir},
        "server": {"url": server_url, "name": "mcp-auth-starter", "host": "0.0.0.0", "port": 8000},
        "auth":   {"token_expire_days": 90},
    }

    cfg_dir  = _Path("/etc/mcp-auth-starter")
    cfg_path = cfg_dir / "config.json"
    env_path = cfg_dir / "mcp-auth-starter.env"

    print()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists() or input(f"{cfg_path} exists. Overwrite? [y/N]: ").strip().lower() == "y":
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"✓ {cfg_path}")

    for sub in ["", "tmp"]:
        (_Path(data_dir) / sub if sub else _Path(data_dir)).mkdir(parents=True, exist_ok=True)
    _Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ {data_dir}/, {log_dir}/")

    if not env_path.exists():
        env_path.write_text(f"SECRET_KEY={_secrets.token_hex(32)}\nMCP_CONFIG_PATH={cfg_path}\n")
        env_path.chmod(0o600)
        print(f"✓ {env_path} (SECRET_KEY pre-generated, chmod 600)")

    from users import _ensure_db_schema, prune_login_alerts as _schema, create_user as _cu, get_user as _gu
    _schema()
    print()
    if not _gu("admin"):
        print("Create admin user for OAuth login:")
        pw, pw2 = _getpass_stars("  Password: "), _getpass_stars("  Confirm:  ")
        if pw and pw == pw2:
            _cu("admin", pw)
            print("✓ admin user created")
        else:
            print("  Skipped — run: python -m app.main --adduser")

    print("\nSetup complete. Start with: python -m app.main\n")


def run_addserviceclient():
    """`python -m app.main --add-service-client` — provision a machine client
    for the client_credentials grant.

    Deliberately a CLI step and not an HTTP endpoint: see
    oauth.create_service_client for why granting this over the open
    registration endpoint would hand out /mcp access to anyone who asked.

    The secret is printed once and never again — only its value is stored,
    and there is nothing here that can show it a second time, same as any
    other credential this project mints.
    """
    from config import DB_PATH
    from oauth import _ensure_tokens_table, create_service_client

    _ensure_tokens_table()
    print(f"\nAdd a machine (client_credentials) client to {DB_PATH}")
    name = input("Client name (what is calling, e.g. 'mcp-proxy'): ").strip()
    username = input("Acts as which existing user: ").strip()
    if not name or not username:
        print("Aborted.")
        return
    try:
        client = create_service_client(name, username)
    except ValueError as e:
        print(f"{e}")
        return
    # codeql[py/clear-text-logging-sensitive-data] — one-time stdout display to
    # the operator running this command, not a log file/aggregator; only the
    # secret's value is stored, so showing it once here is the only moment it
    # can be captured at all. Same "shown once, never persisted in plaintext
    # again" pattern installer.py's generated-password display already uses —
    # see docs/security.md.
    print("\n✓ Created. Store these now — the secret is not recoverable:")
    print(f"  client_id:     {client['client_id']}")
    print(f"  client_secret: {client['client_secret']}")
    print(f"  acts as:       {client['service_username']}")


def run_adduser():
    """`python -m app.main --adduser` — create or reset a user without the full wizard."""
    from users import create_user, get_user, hash_password, _ensure_db_schema as _schema
    from config import DB_PATH

    _schema()
    print(f"\nAdd user to {DB_PATH}")
    username = input("Username: ").strip()
    if not username:
        print("Aborted.")
        return
    if get_user(username):
        if input(f"User '{username}' exists. Reset password? [y/N]: ").strip().lower() != "y":
            print("Aborted.")
            return
        import db
        pw = _getpass_stars("New password: ")
        conn = db.connect(DB_PATH)
        conn.execute("UPDATE users SET password_hash=? WHERE username=?", (hash_password(pw), username))
        conn.commit()
        print(f"✓ Password updated for '{username}'")
    else:
        pw, pw2 = _getpass_stars("Password: "), _getpass_stars("Confirm: ")
        if pw != pw2:
            print("Passwords do not match.")
            return
        print(f"✓ User '{username}' created" if create_user(username, pw) else "Failed to create user.")


def startup_checks():
    from config import CONFIG_FOUND, CONFIG_PATH, DB_PATH
    if not CONFIG_FOUND:
        logger.error(f"Config not found: {CONFIG_PATH}")
        logger.error("Run:  python -m app.main --setup")
        sys.exit(1)
    if not DB_PATH.parent.exists():
        logger.error(f"Data directory missing: {DB_PATH.parent}")
        logger.error("Run:  python -m app.main --setup")
        sys.exit(1)
    logger.info("✅ All checks passed")


if __name__ == "__main__":
    if "--setup" in sys.argv:
        run_setup_wizard()
        sys.exit(0)
    if "--adduser" in sys.argv:
        run_adduser()
        sys.exit(0)
    if "--add-service-client" in sys.argv:
        run_addserviceclient()
        sys.exit(0)
    startup_checks()
    logger.info(f"🚀 {MCP_SERVER_NAME} | http://{MCP_HOST}:{MCP_PORT}/mcp")
    logger.info(f"🔐 Login: {SERVER_URL}/oauth/login")
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT, log_level="warning")
