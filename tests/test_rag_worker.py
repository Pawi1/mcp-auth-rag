"""Tests for rag_worker.py: one poll cycle claims a queued job, runs ingest,
and marks it done. Runs the (intentionally infinite) poll loop as a task and
cancels it once the mocked work has happened."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import rag_store
import rag_worker


class _FakePath:
    def __init__(self, data: bytes):
        self._data = data

    def read_bytes(self) -> bytes:
        return self._data


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(rag_worker, "_POLL_INTERVAL_SECONDS", 0.01)


async def _run_briefly_until(condition, timeout=1.0):
    task = asyncio.create_task(rag_worker._run_forever())
    elapsed = 0.0
    while not condition() and elapsed < timeout:
        await asyncio.sleep(0.02)
        elapsed += 0.02
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return condition()


class TestRunForever:
    async def test_claims_ingests_and_finishes_a_job(self, monkeypatch):
        job = {"id": 1, "document_id": "doc-1"}
        document = {"id": "doc-1", "owner": "alice", "filename": "paper.pdf", "format": "pdf"}
        finish_calls = []

        state = {"claimed": False}

        async def fake_claim():
            if state["claimed"]:
                return None
            state["claimed"] = True
            return job

        async def fake_finish(job_id, ok, error=""):
            finish_calls.append((job_id, ok))

        monkeypatch.setattr(rag_store, "claim_next_job", fake_claim)
        monkeypatch.setattr(rag_store, "get_document", AsyncMock(return_value=document))
        monkeypatch.setattr(rag_store, "upload_path", lambda doc_id, filename: _FakePath(b"pdf-bytes"))
        monkeypatch.setattr(rag_store, "set_document_status", AsyncMock())
        monkeypatch.setattr(rag_store, "finish_job", fake_finish)

        ingest_mock = AsyncMock()
        monkeypatch.setattr("rag_ingest.ingest_document", ingest_mock)

        assert await _run_briefly_until(lambda: bool(finish_calls))
        assert finish_calls == [(1, True)]
        ingest_mock.assert_awaited_once_with("doc-1", "alice", "paper.pdf", "pdf", b"pdf-bytes")

    async def test_ingest_failure_finishes_the_job_as_failed(self, monkeypatch):
        job = {"id": 2, "document_id": "doc-2"}
        document = {"id": "doc-2", "owner": "alice", "filename": "paper.pdf", "format": "pdf"}
        finish_calls = []
        state = {"claimed": False}

        async def fake_claim():
            if state["claimed"]:
                return None
            state["claimed"] = True
            return job

        async def fake_finish(job_id, ok, error=""):
            finish_calls.append((job_id, ok, error))

        monkeypatch.setattr(rag_store, "claim_next_job", fake_claim)
        monkeypatch.setattr(rag_store, "get_document", AsyncMock(return_value=document))
        monkeypatch.setattr(rag_store, "upload_path", lambda doc_id, filename: _FakePath(b"pdf-bytes"))
        monkeypatch.setattr(rag_store, "set_document_status", AsyncMock())
        monkeypatch.setattr(rag_store, "finish_job", fake_finish)
        monkeypatch.setattr("rag_ingest.ingest_document", AsyncMock(side_effect=RuntimeError("boom")))

        assert await _run_briefly_until(lambda: bool(finish_calls))
        assert finish_calls == [(2, False, "boom")]

    async def test_missing_document_finishes_the_job_without_ingesting(self, monkeypatch):
        job = {"id": 3, "document_id": "doc-missing"}
        finish_calls = []
        state = {"claimed": False}

        async def fake_claim():
            if state["claimed"]:
                return None
            state["claimed"] = True
            return job

        async def fake_finish(job_id, ok, error=""):
            finish_calls.append((job_id, ok))

        monkeypatch.setattr(rag_store, "claim_next_job", fake_claim)
        monkeypatch.setattr(rag_store, "get_document", AsyncMock(return_value=None))
        monkeypatch.setattr(rag_store, "finish_job", fake_finish)
        ingest_mock = AsyncMock()
        monkeypatch.setattr("rag_ingest.ingest_document", ingest_mock)

        assert await _run_briefly_until(lambda: bool(finish_calls))
        assert finish_calls == [(3, False)]
        ingest_mock.assert_not_awaited()


class TestStartStop:
    async def test_start_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(rag_store, "claim_next_job", AsyncMock(return_value=None))
        rag_worker.start()
        first_task = rag_worker._task
        rag_worker.start()
        assert rag_worker._task is first_task
        rag_worker.stop()

    async def test_stop_clears_the_task(self, monkeypatch):
        monkeypatch.setattr(rag_store, "claim_next_job", AsyncMock(return_value=None))
        rag_worker.start()
        rag_worker.stop()
        assert rag_worker._task is None
