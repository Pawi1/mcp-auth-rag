"""Tests for rag_store.py.

Only upload_path() is a pure function — everything else in this module is a
thin wrapper around live Postgres/Qdrant queries (schema creation, SKIP
LOCKED job claiming, hybrid search), which this suite deliberately doesn't
mock: a mocked asyncpg/qdrant-client test would mostly just assert "the SQL
string I wrote matches the SQL string I wrote," not catch real bugs. Exercise
those against docker-compose.rag.yml locally (see README) — CI stays
dependency-free, matching the rest of this repo's test suite.
"""

import uuid

import rag_store


class TestUploadPath:
    def test_uses_document_id_and_lowercased_extension(self, monkeypatch):
        monkeypatch.setattr(rag_store, "RAG_UPLOAD_DIR", rag_store.Path("/srv/rag_uploads"))
        doc_id = uuid.uuid4()
        path = rag_store.upload_path(doc_id, "Research Paper.PDF")
        assert path == rag_store.Path("/srv/rag_uploads") / f"{doc_id}.pdf"

    def test_different_filenames_same_id_collide_by_design(self, monkeypatch):
        # upload_path is keyed by document_id, not filename — this is
        # intentional (one file per document row), not a bug to fix here.
        monkeypatch.setattr(rag_store, "RAG_UPLOAD_DIR", rag_store.Path("/srv/rag_uploads"))
        doc_id = uuid.uuid4()
        assert rag_store.upload_path(doc_id, "a.pdf") == rag_store.upload_path(doc_id, "b.PDF")
