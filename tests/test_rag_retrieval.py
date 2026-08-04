"""Tests for rag_retrieval.py: reciprocal_rank_fusion (pure) and search()'s
orchestration of vector search + full-text search + rerank, with rag_store
and rag_embed mocked so nothing touches Postgres/Qdrant/Ollama."""

from unittest.mock import AsyncMock

import rag_retrieval
import rag_store
import rag_embed
from rag_retrieval import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_item_ranked_first_in_both_lists_wins(self):
        result = reciprocal_rank_fusion([["a", "b", "c"], ["a", "d", "e"]])
        assert result[0] == "a"

    def test_item_present_in_both_lists_outranks_single_list_item(self):
        result = reciprocal_rank_fusion([["x", "y", "z"], ["y", "x", "w"]])
        assert set(result[:2]) == {"x", "y"}

    def test_empty_lists_yield_empty_result(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list_preserves_its_order(self):
        assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]

    def test_disjoint_lists_interleave_by_rank(self):
        # both single-list items at rank 0 score equally; the fused order is
        # stable-sorted, so first-list items keep first-list precedence
        result = reciprocal_rank_fusion([["a"], ["b"]])
        assert set(result) == {"a", "b"}


def _chunk(cid: str, text: str = "some text", filename: str = "doc.pdf") -> dict:
    return {"id": cid, "document_id": "11111111-1111-1111-1111-111111111111",
            "page": 1, "section": "Intro", "text": text, "filename": filename}


class TestSearch:
    async def test_empty_query_returns_no_results(self, monkeypatch):
        assert await rag_retrieval.search("   ", "alice") == []

    async def test_fuses_vector_and_fts_results(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "embed_query", AsyncMock(return_value=[0.1, 0.2]))
        monkeypatch.setattr(rag_store, "vector_search", AsyncMock(return_value=["c1", "c2"]))
        monkeypatch.setattr(rag_store, "fts_search", AsyncMock(return_value=["c2", "c3"]))
        monkeypatch.setattr(rag_store, "fetch_chunks_by_id", AsyncMock(
            return_value={cid: _chunk(cid) for cid in ["c1", "c2", "c3"]}
        ))
        monkeypatch.setattr(rag_embed, "rerank", AsyncMock(return_value=None))

        results = await rag_retrieval.search("query", "alice", top_k=8, candidate_k=8)
        assert {r["filename"] for r in results} == {"doc.pdf"}
        assert len(results) == 3  # c1, c2, c3 all surfaced by the union of both lists

    async def test_reranker_reorders_when_available(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "embed_query", AsyncMock(return_value=[0.1]))
        monkeypatch.setattr(rag_store, "vector_search", AsyncMock(return_value=["c1", "c2"]))
        monkeypatch.setattr(rag_store, "fts_search", AsyncMock(return_value=[]))
        monkeypatch.setattr(rag_store, "fetch_chunks_by_id", AsyncMock(return_value={
            "c1": _chunk("c1", text="irrelevant"),
            "c2": _chunk("c2", text="relevant"),
        }))
        # reranker flips the fusion order: c2 (2nd by fusion) scores higher than c1
        monkeypatch.setattr(rag_embed, "rerank", AsyncMock(return_value=[0.1, 0.9]))

        results = await rag_retrieval.search("query", "alice", top_k=8, candidate_k=8)
        assert [r["text"] for r in results] == ["relevant", "irrelevant"]

    async def test_falls_back_to_fusion_order_when_reranker_unavailable(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "embed_query", AsyncMock(return_value=[0.1]))
        monkeypatch.setattr(rag_store, "vector_search", AsyncMock(return_value=["c1", "c2"]))
        monkeypatch.setattr(rag_store, "fts_search", AsyncMock(return_value=[]))
        monkeypatch.setattr(rag_store, "fetch_chunks_by_id", AsyncMock(return_value={
            "c1": _chunk("c1", text="first"),
            "c2": _chunk("c2", text="second"),
        }))
        monkeypatch.setattr(rag_embed, "rerank", AsyncMock(return_value=None))

        results = await rag_retrieval.search("query", "alice", top_k=8, candidate_k=8)
        assert [r["text"] for r in results] == ["first", "second"]

    async def test_respects_top_k(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "embed_query", AsyncMock(return_value=[0.1]))
        monkeypatch.setattr(rag_store, "vector_search", AsyncMock(return_value=["c1", "c2", "c3"]))
        monkeypatch.setattr(rag_store, "fts_search", AsyncMock(return_value=[]))
        monkeypatch.setattr(rag_store, "fetch_chunks_by_id", AsyncMock(
            return_value={cid: _chunk(cid) for cid in ["c1", "c2", "c3"]}
        ))
        monkeypatch.setattr(rag_embed, "rerank", AsyncMock(return_value=None))

        results = await rag_retrieval.search("query", "alice", top_k=1, candidate_k=8)
        assert len(results) == 1

    async def test_vector_search_failure_does_not_break_search(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "embed_query", AsyncMock(return_value=[0.1]))
        monkeypatch.setattr(rag_store, "vector_search", AsyncMock(side_effect=RuntimeError("qdrant down")))
        monkeypatch.setattr(rag_store, "fts_search", AsyncMock(return_value=["c1"]))
        monkeypatch.setattr(rag_store, "fetch_chunks_by_id", AsyncMock(return_value={"c1": _chunk("c1")}))
        monkeypatch.setattr(rag_embed, "rerank", AsyncMock(return_value=None))

        results = await rag_retrieval.search("query", "alice", top_k=8, candidate_k=8)
        assert len(results) == 1

    async def test_both_backends_failing_returns_empty(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "embed_query", AsyncMock(return_value=[0.1]))
        monkeypatch.setattr(rag_store, "vector_search", AsyncMock(side_effect=RuntimeError("down")))
        monkeypatch.setattr(rag_store, "fts_search", AsyncMock(side_effect=RuntimeError("down")))

        results = await rag_retrieval.search("query", "alice", top_k=8, candidate_k=8)
        assert results == []
