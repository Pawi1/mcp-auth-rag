"""Tests for server.py — the demo `whoami` tool, the RAG tools, and the auth
gate in call_tool()."""

import json
import sqlite3
from unittest.mock import AsyncMock

import pytest

import server
from context import current_user


@pytest.fixture(autouse=True)
def reset_current_user():
    token = current_user.set(None)
    yield
    current_user.reset(token)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("users.DB_PATH", db_path)
    import users as _users
    _users._ensure_db_schema()
    return db_path


def _result_json(result):
    assert len(result) == 1
    return json.loads(result[0].text)


class TestToolConsistency:
    async def test_every_advertised_tool_has_a_dispatch_branch(self):
        """A cheap regression guard against tool-name drift between list_tools()
        and call_tool() as you add your own tools."""
        import inspect
        import re

        tools = await server.list_tools()
        advertised = {t.name for t in tools}
        source = inspect.getsource(server.call_tool)
        handled = set(re.findall(r'name == "([^"]+)"', source))
        assert advertised <= handled


class TestAuthGate:
    async def test_unauthenticated_call_is_rejected(self):
        current_user.set(None)
        result = await server.call_tool("whoami", {})
        data = _result_json(result)
        assert "error" in data
        assert "not authenticated" in data["error"].lower()

    async def test_unknown_tool_name_is_rejected(self):
        current_user.set({"username": "alice", "teams": ["admins"]})
        result = await server.call_tool("this_tool_does_not_exist", {})
        data = _result_json(result)
        assert "error" in data


class TestWhoami:
    async def test_returns_authenticated_identity(self):
        current_user.set({"username": "alice", "teams": ["admins", "beta"]})
        result = await server.call_tool("whoami", {})
        data = _result_json(result)
        assert data == {"username": "alice", "teams": ["admins", "beta"]}


class TestRagSearch:
    async def test_unauthenticated_call_is_rejected(self):
        current_user.set(None)
        result = await server.call_tool("rag_search", {"query": "test"})
        data = _result_json(result)
        assert "error" in data

    async def test_calls_retrieval_search_with_the_caller_as_owner(self, monkeypatch):
        import rag_retrieval
        search_mock = AsyncMock(return_value=[
            {"text": "found it", "filename": "paper.pdf", "page": 2, "section": "Wyniki", "document_id": "x"}
        ])
        monkeypatch.setattr(rag_retrieval, "search", search_mock)
        current_user.set({"username": "alice", "teams": []})

        result = await server.call_tool("rag_search", {"query": "wyniki badania", "top_k": 3})
        data = _result_json(result)

        assert data["results"][0]["text"] == "found it"
        search_mock.assert_awaited_once_with("wyniki badania", "alice", top_k=3)

    async def test_defaults_top_k_when_not_provided(self, monkeypatch):
        import rag_retrieval
        from config import RAG_TOP_K
        search_mock = AsyncMock(return_value=[])
        monkeypatch.setattr(rag_retrieval, "search", search_mock)
        current_user.set({"username": "alice", "teams": []})

        await server.call_tool("rag_search", {"query": "x"})
        search_mock.assert_awaited_once_with("x", "alice", top_k=RAG_TOP_K)


class TestRagListDocuments:
    async def test_unauthenticated_call_is_rejected(self):
        current_user.set(None)
        result = await server.call_tool("rag_list_documents", {})
        data = _result_json(result)
        assert "error" in data

    async def test_serializes_documents_for_the_caller(self, monkeypatch):
        import datetime
        import uuid
        import rag_store

        doc = {
            "id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
            "filename": "paper.pdf", "format": "pdf", "status": "done", "error": None,
            "page_count": 12, "chunk_count": 40,
            "uploaded_at": datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            "processed_at": None,
        }
        list_mock = AsyncMock(return_value=[doc])
        monkeypatch.setattr(rag_store, "list_documents", list_mock)
        current_user.set({"username": "alice", "teams": []})

        result = await server.call_tool("rag_list_documents", {})
        data = _result_json(result)

        list_mock.assert_awaited_once_with("alice")
        assert data["documents"][0]["id"] == "11111111-1111-1111-1111-111111111111"
        assert data["documents"][0]["uploaded_at"] == "2026-01-01T00:00:00+00:00"
        assert data["documents"][0]["processed_at"] is None


class TestRagGetContext:
    async def test_unauthenticated_call_is_rejected(self):
        current_user.set(None)
        result = await server.call_tool("rag_get_context", {"document_id": "doc-1", "ordinal": 3})
        data = _result_json(result)
        assert "error" in data

    async def test_passes_arguments_through_to_retrieval(self, monkeypatch):
        import rag_retrieval

        get_context_mock = AsyncMock(return_value={
            "filename": "paper.pdf", "text": "more context", "start_ordinal": 1, "end_ordinal": 5, "pages": [1],
        })
        monkeypatch.setattr(rag_retrieval, "get_context", get_context_mock)
        current_user.set({"username": "alice", "teams": []})

        result = await server.call_tool(
            "rag_get_context", {"document_id": "doc-1", "ordinal": 3, "chunks_before": 1, "chunks_after": 4}
        )
        data = _result_json(result)

        assert data["text"] == "more context"
        get_context_mock.assert_awaited_once_with("doc-1", 3, "alice", chunks_before=1, chunks_after=4)

    async def test_defaults_chunks_before_after_when_not_provided(self, monkeypatch):
        import rag_retrieval

        get_context_mock = AsyncMock(return_value={
            "filename": "paper.pdf", "text": "x", "start_ordinal": 1, "end_ordinal": 1, "pages": [],
        })
        monkeypatch.setattr(rag_retrieval, "get_context", get_context_mock)
        current_user.set({"username": "alice", "teams": []})

        await server.call_tool("rag_get_context", {"document_id": "doc-1", "ordinal": 3})

        get_context_mock.assert_awaited_once_with("doc-1", 3, "alice", chunks_before=2, chunks_after=2)

    async def test_not_found_returns_error_payload(self, monkeypatch):
        import rag_retrieval

        monkeypatch.setattr(rag_retrieval, "get_context", AsyncMock(return_value=None))
        current_user.set({"username": "alice", "teams": []})

        result = await server.call_tool("rag_get_context", {"document_id": "doc-1", "ordinal": 3})
        data = _result_json(result)
        assert "error" in data


class TestToolCallAudit:
    async def test_successful_call_is_audited(self, tmp_db):
        current_user.set({"username": "alice", "teams": ["admins"]})
        await server.call_tool("whoami", {})
        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute(
            "SELECT success FROM tool_call_log WHERE username='alice' AND tool_name='whoami'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 1

    async def test_unknown_tool_is_audited_as_failure(self, tmp_db):
        current_user.set({"username": "alice", "teams": ["admins"]})
        await server.call_tool("delete_everything", {})
        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute(
            "SELECT success, reason FROM tool_call_log WHERE username='alice' AND tool_name='delete_everything'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 0
        assert row[1] == "unknown_tool"

    async def test_unauthenticated_call_is_not_audited(self, tmp_db):
        current_user.set(None)
        await server.call_tool("whoami", {})
        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute("SELECT COUNT(*) FROM tool_call_log").fetchone()[0]
        conn.close()
        assert count == 0
