"""
MCP Auth Starter — RAG web panel: upload, document list, and search, all
gated by a browser session cookie (separate from the OAuth authorization-code
flow in oauth.py, which authenticates *MCP clients*, not humans in a browser).

CSRF: the session cookie is SameSite=Strict, which stops it being sent on any
cross-site request (including top-level navigations) — simpler than a second
token-based CSRF scheme, and sufficient since nothing here needs to work from
a cross-site link the way the OAuth login flow does.
"""

import logging
import time
from pathlib import Path

from jose import jwt as jose_jwt
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.templating import Jinja2Templates

from config import ALGORITHM, RAG_MAX_UPLOAD_MB, SECRET_KEY, SERVER_URL
from oauth import _page
from rag_ingest import content_hash, sniff_format
from users import verify_user

logger = logging.getLogger("mcp-auth-starter")

_SESSION_COOKIE = "rag_session"
_SESSION_AUD = f"{SERVER_URL}/rag"
_SESSION_TTL_SECONDS = 7 * 86400
_MAX_UPLOAD_BYTES = RAG_MAX_UPLOAD_MB * 1024 * 1024

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _issue_session(username: str) -> str:
    now = time.time()
    return jose_jwt.encode(
        {"sub": username, "aud": _SESSION_AUD, "exp": int(now + _SESSION_TTL_SECONDS)},
        SECRET_KEY, algorithm=ALGORITHM,
    )


def _current_username(request: Request) -> str | None:
    token = request.cookies.get(_SESSION_COOKIE, "")
    if not token:
        return None
    try:
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], audience=_SESSION_AUD)
        return payload.get("sub")
    except Exception:
        return None


def _set_session_cookie(response: Response, request: Request, username: str) -> None:
    response.set_cookie(
        _SESSION_COOKIE, _issue_session(username), max_age=_SESSION_TTL_SECONDS,
        httponly=True, samesite="strict", secure=request.url.scheme == "https",
    )


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------

async def rag_login(request: Request) -> Response:
    if _current_username(request):
        return RedirectResponse("/rag", status_code=303)

    error = request.query_params.get("error", "")
    error_html = f'<div class="err">{error}</div>' if error else ""
    body = f"""
<h2>RAG panel</h2>
<p class="sub">Sign in to upload and search documents</p>
{error_html}
<form method="post" action="/rag/login">
  <label>Username</label>
  <input name="username" type="text" autocomplete="username" required autofocus>
  <label>Password</label>
  <input name="password" type="password" autocomplete="current-password" required>
  <button class="btn" type="submit">Sign in</button>
</form>
"""
    return HTMLResponse(_page("RAG panel — sign in", body))


async def rag_login_post(request: Request) -> Response:
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))

    ok, _ = verify_user(username, password)
    if not ok:
        return RedirectResponse("/rag/login?error=Invalid+username+or+password", status_code=303)

    response = RedirectResponse("/rag", status_code=303)
    _set_session_cookie(response, request, username)
    return response


async def rag_logout(request: Request) -> Response:
    response = RedirectResponse("/rag/login", status_code=303)
    response.delete_cookie(_SESSION_COOKIE)
    return response


def _require_user(request: Request) -> str | Response:
    """Returns the username, or a redirect Response if not signed in —
    callers do `user = _require_user(request); if isinstance(user, Response): return user`."""
    username = _current_username(request)
    return username if username else RedirectResponse("/rag/login", status_code=303)


# ---------------------------------------------------------------------------
# Panel + documents
# ---------------------------------------------------------------------------

async def rag_panel(request: Request) -> Response:
    user = _require_user(request)
    if isinstance(user, Response):
        return user
    import rag_store
    documents = await rag_store.list_documents(user)
    return templates.TemplateResponse(request, "panel.html", {"username": user, "documents": documents})


async def rag_documents_fragment(request: Request) -> Response:
    user = _require_user(request)
    if isinstance(user, Response):
        return user
    import rag_store
    documents = await rag_store.list_documents(user)
    return templates.TemplateResponse(request, "_documents_fragment.html", {"documents": documents})


async def rag_upload(request: Request) -> Response:
    user = _require_user(request)
    if isinstance(user, Response):
        return user
    import rag_store

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_UPLOAD_BYTES:
        return HTMLResponse(f'<div class="err">File(s) too large — max {RAG_MAX_UPLOAD_MB}MB per upload.</div>', status_code=413)

    form = await request.form()
    files = form.getlist("files")
    for upload in files:
        filename = upload.filename or ""
        fmt = sniff_format(filename)
        if fmt is None:
            logger.warning(f"RAG upload rejected (unsupported format): {filename!r} by {user}")
            continue

        data = await upload.read()
        if len(data) > _MAX_UPLOAD_BYTES:
            logger.warning(f"RAG upload rejected (too large): {filename!r} by {user}")
            continue

        doc = await rag_store.create_document(user, filename, fmt, content_hash(data))
        if doc is None:
            logger.info(f"RAG upload skipped (duplicate content): {filename!r} by {user}")
            continue

        rag_store.upload_path(doc["id"], filename).write_bytes(data)
        await rag_store.enqueue_ingest_job(doc["id"])
        logger.info(f"RAG upload queued: {filename!r} by {user} (document_id={doc['id']})")

    documents = await rag_store.list_documents(user)
    return templates.TemplateResponse(request, "_documents_fragment.html", {"documents": documents})


async def rag_delete_document(request: Request) -> Response:
    user = _require_user(request)
    if isinstance(user, Response):
        return user
    import rag_store
    document_id = request.path_params["document_id"]
    await rag_store.delete_document(document_id, user)
    documents = await rag_store.list_documents(user)
    return templates.TemplateResponse(request, "_documents_fragment.html", {"documents": documents})


async def rag_search(request: Request) -> Response:
    user = _require_user(request)
    if isinstance(user, Response):
        return user
    import rag_retrieval

    form = await request.form()
    query = str(form.get("query", "")).strip()
    results = await rag_retrieval.search(query, user) if query else []
    return templates.TemplateResponse(request, "_search_results.html", {"query": query, "results": results})
