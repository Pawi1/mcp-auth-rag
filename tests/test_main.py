"""Tests for main.py - the /mcp auth gate and /health."""

import sqlite3
import time
from unittest.mock import patch

import pytest
import jwt as pyjwt
from starlette.testclient import TestClient

import config
import main
import oauth

SECRET_KEY = config.SECRET_KEY
ALGORITHM = config.ALGORITHM


def _make_token(username="alice", teams=None, exp_delta=86400, svc=""):
    claims = {
        "sub": username,
        "teams": teams if teams is not None else ["admins"],
        "aud": config.MCP_RESOURCE_URI,
        "exp": int(time.time()) + exp_delta,
    }
    if svc:
        claims["svc"] = svc
    return pyjwt.encode(claims, SECRET_KEY, algorithm=ALGORITHM)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(oauth, "DB_PATH", db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE oauth_tokens (
        token TEXT PRIMARY KEY, username TEXT, issued_at REAL, expires_at REAL
    )""")
    conn.commit()
    conn.close()
    return db_path


def _register_token(db_path, token, username="alice"):
    now = time.time()
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO oauth_tokens VALUES (?,?,?,?)", (token, username, now, now + 86400))
    conn.commit()
    conn.close()


@pytest.fixture
def client():
    return TestClient(main.app)


async def _fake_handle_request(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


class TestHandleMcpAuth:
    def test_no_token_401(self, client):
        resp = client.post("/mcp")
        assert resp.status_code == 401
        assert "Bearer" in resp.headers["www-authenticate"]

    def test_challenge_advertises_scope(self, client):
        """MCP 2026-07-28: the challenge SHOULD carry the scope the client needs,
        so it can request the right one before starting the flow."""
        resp = client.post("/mcp")
        assert 'scope="mcp"' in resp.headers["www-authenticate"]

    def test_non_bearer_scheme_401(self, client):
        """Slicing off a fixed 7 chars turned "Basic dXNlcjpwYXNz" into a
        mangled token and reported it as invalid rather than unauthenticated."""
        resp = client.post("/mcp", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert resp.status_code == 401
        assert "resource_metadata" in resp.headers.get("WWW-Authenticate", "")

    def test_bearer_without_token_401(self, client):
        resp = client.post("/mcp", headers={"Authorization": "Bearer"})
        assert resp.status_code == 401

    def test_invalid_signature_401(self, client):
        bad = pyjwt.encode(
            {"sub": "alice", "teams": [], "exp": int(time.time()) + 3600},
            "wrong-secret", algorithm=ALGORITHM,
        )
        resp = client.post("/mcp", headers={"Authorization": f"Bearer {bad}"})
        assert resp.status_code == 401

    def test_valid_signature_but_not_registered_401(self, client):
        """A structurally-valid JWT never inserted into oauth_tokens must still 401 -
        this is what makes token revocation actually work."""
        token = _make_token()
        resp = client.post("/mcp", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_expired_registered_token_401(self, client, tmp_db):
        token = _make_token(exp_delta=86400)
        now = time.time()
        conn = sqlite3.connect(str(tmp_db))
        conn.execute("INSERT INTO oauth_tokens VALUES (?,?,?,?)", (token, "alice", now - 100, now - 1))
        conn.commit()
        conn.close()
        resp = client.post("/mcp", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_valid_registered_token_reaches_dispatch(self, client, tmp_db):
        token = _make_token(username="alice", teams=["admins"])
        _register_token(tmp_db, token)

        with patch.object(main.session_manager, "handle_request", side_effect=_fake_handle_request):
            resp = client.post("/mcp", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.content == b"ok"

    def test_token_via_query_param_rejected(self, client, tmp_db):
        # MCP spec requires the Authorization header; a token in the query
        # string must not be accepted (it'd end up in access/proxy logs).
        token = _make_token(username="bob", teams=["admins"])
        _register_token(tmp_db, token, username="bob")

        resp = client.post("/mcp", params={"token": token})
        assert resp.status_code == 401


class TestActingOnBehalfOf:
    """X-MCP-Actor lets a service client say which person a call is really
    for, so an audit trail downstream records them and not the machine
    account the proxy authenticates as. The gate is the `svc` claim, which
    only the client_credentials grant puts on a token - every test here is
    about that gate holding.
    """

    def _seen_user(self, client, tmp_db, token, headers=None):
        from context import current_user

        seen = {}

        async def _capture(scope, receive, send):
            seen["user"] = current_user.get()
            await _fake_handle_request(scope, receive, send)

        _register_token(tmp_db, token)
        with patch.object(main.session_manager, "handle_request", side_effect=_capture):
            resp = client.post("/mcp", headers={"Authorization": f"Bearer {token}", **(headers or {})})
        assert resp.status_code == 200
        return seen["user"]

    def test_a_service_token_may_name_who_it_acts_for(self, client, tmp_db):
        user = self._seen_user(
            client, tmp_db, _make_token(username="svc-proxy", svc="mcp-proxy"),
            {"X-MCP-Actor": "pawel"},
        )
        assert user["on_behalf_of"] == "pawel"
        assert user["username"] == "svc-proxy"  # authorization is still the service account's

    def test_an_ordinary_token_may_not(self, client, tmp_db):
        # The impersonation case. A user token carries no `svc` claim, so the
        # header is ignored entirely rather than trusted.
        user = self._seen_user(
            client, tmp_db, _make_token(username="alice"), {"X-MCP-Actor": "someone-else"},
        )
        assert not user.get("on_behalf_of")
        assert user["username"] == "alice"

    def test_a_service_token_without_the_header_acts_as_itself(self, client, tmp_db):
        user = self._seen_user(client, tmp_db, _make_token(username="svc-proxy", svc="mcp-proxy"))
        assert not user.get("on_behalf_of")

    def test_a_percent_encoded_name_survives(self, client, tmp_db):
        # An HTTP header cannot carry a non-ASCII name raw, so the proxy
        # encodes it and this end has to decode it back.
        user = self._seen_user(
            client, tmp_db, _make_token(username="svc-proxy", svc="mcp-proxy"),
            {"X-MCP-Actor": "pawe%C5%82"},
        )
        assert user["on_behalf_of"] == "pawe\u0142"

    def test_control_characters_are_stripped(self, client, tmp_db):
        # A newline here could split a log line or a downstream header.
        user = self._seen_user(
            client, tmp_db, _make_token(username="svc-proxy", svc="mcp-proxy"),
            {"X-MCP-Actor": "pawel%0AX-Injected%3A%20yes"},
        )
        assert "\n" not in user["on_behalf_of"]
        assert user["on_behalf_of"].startswith("pawel")

    def test_an_overlong_name_is_capped(self, client, tmp_db):
        user = self._seen_user(
            client, tmp_db, _make_token(username="svc-proxy", svc="mcp-proxy"),
            {"X-MCP-Actor": "a" * 500},
        )
        assert len(user["on_behalf_of"]) == main.MAX_ACTOR_LEN


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "server": config.MCP_SERVER_NAME}
