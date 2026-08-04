"""Tests for rag_routes.py: the panel's own session-cookie auth (separate
from oauth.py's bearer tokens for MCP clients), and upload/search/delete
with rag_store/rag_retrieval mocked so nothing touches Postgres/Qdrant."""

from unittest.mock import AsyncMock

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import rag_retrieval
import rag_routes
import rag_store
import users as _users


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("users.DB_PATH", db_path)
    _users._ensure_db_schema()
    _users.create_user("alice", "correct-password")
    return db_path


@pytest.fixture()
def test_client():
    app = Starlette(routes=[
        Route("/rag",                     endpoint=rag_routes.rag_panel,              methods=["GET"]),
        Route("/rag/login",               endpoint=rag_routes.rag_login,              methods=["GET"]),
        Route("/rag/login",               endpoint=rag_routes.rag_login_post,         methods=["POST"]),
        Route("/rag/logout",              endpoint=rag_routes.rag_logout,             methods=["GET"]),
        Route("/rag/documents",           endpoint=rag_routes.rag_upload,             methods=["POST"]),
        Route("/rag/documents/fragment",  endpoint=rag_routes.rag_documents_fragment, methods=["GET"]),
        Route("/rag/documents/{document_id}", endpoint=rag_routes.rag_delete_document, methods=["DELETE"]),
        Route("/rag/search",              endpoint=rag_routes.rag_search,             methods=["POST"]),
    ])
    return TestClient(app, raise_server_exceptions=True, follow_redirects=False)


@pytest.fixture()
def logged_in_client(test_client, monkeypatch):
    monkeypatch.setattr(rag_store, "list_documents", AsyncMock(return_value=[]))
    resp = test_client.post("/rag/login", data={"username": "alice", "password": "correct-password"})
    assert resp.status_code == 303
    return test_client


class TestSessionRoundTrip:
    class _FakeRequest:
        def __init__(self, cookies):
            self.cookies = cookies

    def test_issue_then_verify_recovers_the_username(self):
        token = rag_routes._issue_session("alice")
        assert rag_routes._current_username(self._FakeRequest({"rag_session": token})) == "alice"

    def test_garbage_token_is_rejected(self):
        assert rag_routes._current_username(self._FakeRequest({"rag_session": "not-a-jwt"})) is None

    def test_missing_cookie_is_rejected(self):
        assert rag_routes._current_username(self._FakeRequest({})) is None


class TestLogin:
    def test_wrong_password_redirects_with_error(self, test_client):
        resp = test_client.post("/rag/login", data={"username": "alice", "password": "wrong"})
        assert resp.status_code == 303
        assert "error=" in resp.headers["location"]
        assert "rag_session" not in resp.cookies

    def test_correct_password_redirects_to_panel_with_session_cookie(self, test_client, monkeypatch):
        monkeypatch.setattr(rag_store, "list_documents", AsyncMock(return_value=[]))
        resp = test_client.post("/rag/login", data={"username": "alice", "password": "correct-password"})
        assert resp.status_code == 303
        assert resp.headers["location"] == "/rag"
        assert "rag_session" in resp.cookies

    def test_session_cookie_is_httponly_and_samesite_strict(self, test_client):
        resp = test_client.post("/rag/login", data={"username": "alice", "password": "correct-password"})
        set_cookie = resp.headers.get("set-cookie", "").lower()
        assert "httponly" in set_cookie
        assert "samesite=strict" in set_cookie

    def test_already_signed_in_visiting_login_redirects_to_panel(self, logged_in_client):
        resp = logged_in_client.get("/rag/login")
        assert resp.status_code == 303
        assert resp.headers["location"] == "/rag"


class TestRequiresAuth:
    def test_panel_redirects_when_signed_out(self, test_client):
        resp = test_client.get("/rag")
        assert resp.status_code == 303
        assert resp.headers["location"] == "/rag/login"

    def test_upload_redirects_when_signed_out(self, test_client):
        resp = test_client.post("/rag/documents", files=[("files", ("a.pdf", b"x", "application/pdf"))])
        assert resp.status_code == 303

    def test_search_redirects_when_signed_out(self, test_client):
        resp = test_client.post("/rag/search", data={"query": "x"})
        assert resp.status_code == 303

    def test_delete_redirects_when_signed_out(self, test_client):
        resp = test_client.delete("/rag/documents/11111111-1111-1111-1111-111111111111")
        assert resp.status_code == 303


class TestPanel:
    def test_renders_with_document_list(self, logged_in_client, monkeypatch):
        monkeypatch.setattr(rag_store, "list_documents", AsyncMock(return_value=[]))
        resp = logged_in_client.get("/rag")
        assert resp.status_code == 200
        assert "alice" in resp.text

    def test_renders_the_upload_dropzone(self, logged_in_client, monkeypatch):
        monkeypatch.setattr(rag_store, "list_documents", AsyncMock(return_value=[]))
        resp = logged_in_client.get("/rag")
        assert 'id="dropzone"' in resp.text
        assert 'id="file-input"' in resp.text
        assert 'id="progress-track"' in resp.text


class TestUpload:
    def test_unsupported_format_is_not_queued(self, logged_in_client, monkeypatch):
        create_calls = []
        monkeypatch.setattr(rag_store, "create_document", AsyncMock(side_effect=lambda *a: create_calls.append(a)))
        resp = logged_in_client.post("/rag/documents", files=[("files", ("virus.exe", b"data", "application/octet-stream"))])
        assert resp.status_code == 200
        assert create_calls == []

    def test_valid_pdf_is_saved_and_queued(self, logged_in_client, monkeypatch, tmp_path):
        doc_row = {"id": "11111111-1111-1111-1111-111111111111", "filename": "paper.pdf"}
        monkeypatch.setattr(rag_store, "create_document", AsyncMock(return_value=doc_row))
        saved_path = tmp_path / "upload.pdf"
        monkeypatch.setattr(rag_store, "upload_path", lambda doc_id, filename: saved_path)
        enqueue_calls = []
        monkeypatch.setattr(rag_store, "enqueue_ingest_job", AsyncMock(side_effect=lambda doc_id: enqueue_calls.append(doc_id)))

        resp = logged_in_client.post("/rag/documents", files=[("files", ("paper.pdf", b"%PDF-1.4 fake", "application/pdf"))])
        assert resp.status_code == 200
        assert enqueue_calls == [doc_row["id"]]
        assert saved_path.read_bytes() == b"%PDF-1.4 fake"

    def test_duplicate_content_is_skipped(self, logged_in_client, monkeypatch):
        monkeypatch.setattr(rag_store, "create_document", AsyncMock(return_value=None))
        enqueue_calls = []
        monkeypatch.setattr(rag_store, "enqueue_ingest_job", AsyncMock(side_effect=lambda doc_id: enqueue_calls.append(doc_id)))
        resp = logged_in_client.post("/rag/documents", files=[("files", ("paper.pdf", b"content", "application/pdf"))])
        assert resp.status_code == 200
        assert enqueue_calls == []

    def test_oversized_upload_rejected_before_reading(self, logged_in_client, monkeypatch):
        monkeypatch.setattr(rag_routes, "_MAX_UPLOAD_BYTES", 10)
        create_calls = []
        monkeypatch.setattr(rag_store, "create_document", AsyncMock(side_effect=lambda *a: create_calls.append(a)))
        resp = logged_in_client.post("/rag/documents", files=[("files", ("paper.pdf", b"x" * 1000, "application/pdf"))])
        assert resp.status_code == 413
        assert create_calls == []


class TestDelete:
    def test_scopes_deletion_to_the_signed_in_owner(self, logged_in_client, monkeypatch):
        delete_calls = []

        async def fake_delete(document_id, owner):
            delete_calls.append((document_id, owner))
            return True

        monkeypatch.setattr(rag_store, "delete_document", fake_delete)
        resp = logged_in_client.delete("/rag/documents/11111111-1111-1111-1111-111111111111")
        assert resp.status_code == 200
        assert delete_calls == [("11111111-1111-1111-1111-111111111111", "alice")]


class TestSearch:
    def test_blank_query_does_not_call_retrieval(self, logged_in_client, monkeypatch):
        search_mock = AsyncMock(return_value=[])
        monkeypatch.setattr(rag_retrieval, "search", search_mock)
        resp = logged_in_client.post("/rag/search", data={"query": ""})
        assert resp.status_code == 200
        search_mock.assert_not_called()

    def test_valid_query_renders_results_with_citation(self, logged_in_client, monkeypatch):
        monkeypatch.setattr(rag_retrieval, "search", AsyncMock(return_value=[
            {"text": "found this passage", "filename": "paper.pdf", "page": 3, "section": "Wyniki", "document_id": "x"},
        ]))
        resp = logged_in_client.post("/rag/search", data={"query": "test"})
        assert resp.status_code == 200
        assert "found this passage" in resp.text
        assert "paper.pdf" in resp.text


class TestLogout:
    def test_clears_cookie_and_redirects_to_login(self, logged_in_client):
        resp = logged_in_client.get("/rag/logout")
        assert resp.status_code == 303
        assert resp.headers["location"] == "/rag/login"
