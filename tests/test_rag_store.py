"""Tests for rag_store.py.

upload_path() and _no_nul() are pure functions — everything else in this
module is a thin wrapper around live Postgres/Qdrant queries (schema
creation, SKIP LOCKED job claiming, hybrid search), which this suite
deliberately doesn't mock: a mocked asyncpg/qdrant-client test would mostly
just assert "the SQL string I wrote matches the SQL string I wrote," not
catch real bugs. Exercise those against docker-compose.rag.yml locally (see
README) — CI stays dependency-free, matching the rest of this repo's test
suite.
"""

import uuid

import pytest

import rag_store


class TestNoNul:
    def test_strips_nul_bytes(self):
        assert rag_store._no_nul("hello\x00world") == "helloworld"

    def test_leaves_clean_text_unchanged(self):
        assert rag_store._no_nul("hello world") == "hello world"

    def test_passes_through_none(self):
        assert rag_store._no_nul(None) is None


class TestFtsLanguageValidation:
    """rag_store interpolates RAG_FTS_LANGUAGE directly into the chunks
    table's GENERATED column DDL (Postgres requires a literal there, not a
    bind parameter) — _FTS_LANGUAGE_RE is the guard against that ever being
    anything but a plain regconfig-shaped name."""

    @pytest.mark.parametrize("name", ["simple", "english", "german", "pl_stem"])
    def test_accepts_plausible_regconfig_names(self, name):
        assert rag_store._FTS_LANGUAGE_RE.fullmatch(name)

    @pytest.mark.parametrize("name", [
        "english; DROP TABLE chunks;--",
        "english'); DROP TABLE chunks;--",
        "",
        "English",
        "with space",
        "with-dash",
    ])
    def test_rejects_anything_else(self, name):
        assert not rag_store._FTS_LANGUAGE_RE.fullmatch(name)


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
