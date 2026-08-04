"""Tests for rag_embed.py — batching/order/concurrency for embeddings, and
the reranker's graceful-fallback behavior, all against a fake httpx client
(no network)."""

import pytest

import rag_embed


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient so these tests never touch the network."""
    def __init__(self, post_handler, **_kwargs):
        self._post_handler = post_handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def post(self, url, headers=None, json=None):
        return self._post_handler(url, headers, json)


def _patch_client(monkeypatch, post_handler):
    monkeypatch.setattr(rag_embed.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(post_handler, **kw))


@pytest.fixture(autouse=True)
def reset_reranker_warned():
    rag_embed._reranker_warned = False
    yield
    rag_embed._reranker_warned = False


class TestEmbedTexts:
    async def test_empty_input_makes_no_request(self, monkeypatch):
        calls = []
        _patch_client(monkeypatch, lambda url, headers, json: calls.append(1) or _FakeResponse({"data": []}))
        result = await rag_embed.embed_texts([])
        assert result == []
        assert calls == []

    async def test_preserves_order_across_batches(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "_EMBED_BATCH_SIZE", 2)

        def handler(url, headers, json):
            batch = json["input"]
            return _FakeResponse({"data": [{"embedding": [float(len(t))]} for t in batch]})

        _patch_client(monkeypatch, handler)
        texts = ["a", "bb", "ccc", "dddd", "e"]
        result = await rag_embed.embed_texts(texts)
        assert result == [[1.0], [2.0], [3.0], [4.0], [1.0]]

    async def test_sends_configured_model(self, monkeypatch):
        seen = {}

        def handler(url, headers, json):
            seen["model"] = json["model"]
            seen["url"] = url
            return _FakeResponse({"data": [{"embedding": [0.1]}]})

        _patch_client(monkeypatch, handler)
        await rag_embed.embed_texts(["x"])
        assert seen["model"] == rag_embed.RAG_EMBEDDING_MODEL
        assert seen["url"].endswith("/embeddings")

    async def test_omits_auth_header_when_no_api_key(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "RAG_OLLAMA_API_KEY", "")
        seen = {}

        def handler(url, headers, json):
            seen["headers"] = headers
            return _FakeResponse({"data": [{"embedding": [0.1]}]})

        _patch_client(monkeypatch, handler)
        await rag_embed.embed_texts(["x"])
        assert "Authorization" not in seen["headers"]

    async def test_sends_bearer_header_when_api_key_set(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "RAG_OLLAMA_API_KEY", "secret-key")
        seen = {}

        def handler(url, headers, json):
            seen["headers"] = headers
            return _FakeResponse({"data": [{"embedding": [0.1]}]})

        _patch_client(monkeypatch, handler)
        await rag_embed.embed_texts(["x"])
        assert seen["headers"]["Authorization"] == "Bearer secret-key"


class TestEmbedQuery:
    async def test_returns_single_vector(self, monkeypatch):
        _patch_client(monkeypatch, lambda url, headers, json: _FakeResponse({"data": [{"embedding": [1.0, 2.0]}]}))
        result = await rag_embed.embed_query("hello")
        assert result == [1.0, 2.0]


class TestRerank:
    async def test_returns_scores_matching_input_order(self, monkeypatch):
        def handler(url, headers, json):
            assert url.endswith("/api/rerank")
            return _FakeResponse({"results": [
                {"document": "doc B", "relevance_score": 0.9},
                {"document": "doc A", "relevance_score": 0.1},
            ]})

        _patch_client(monkeypatch, handler)
        scores = await rag_embed.rerank("query", ["doc A", "doc B"])
        assert scores == [0.1, 0.9]

    async def test_returns_none_on_http_error_without_raising(self, monkeypatch):
        _patch_client(monkeypatch, lambda url, headers, json: _FakeResponse({}, status_code=404))
        scores = await rag_embed.rerank("query", ["doc A"])
        assert scores is None

    async def test_returns_none_on_malformed_response(self, monkeypatch):
        _patch_client(monkeypatch, lambda url, headers, json: _FakeResponse({"unexpected": "shape"}))
        scores = await rag_embed.rerank("query", ["doc A"])
        assert scores is None

    async def test_empty_documents_returns_none_without_request(self, monkeypatch):
        calls = []
        _patch_client(monkeypatch, lambda url, headers, json: calls.append(1) or _FakeResponse({"results": []}))
        assert await rag_embed.rerank("query", []) is None
        assert calls == []

    async def test_disabled_reranker_returns_none_without_request(self, monkeypatch):
        monkeypatch.setattr(rag_embed, "RAG_RERANKER_ENABLED", False)
        calls = []
        _patch_client(monkeypatch, lambda url, headers, json: calls.append(1) or _FakeResponse({"results": []}))
        assert await rag_embed.rerank("query", ["doc A"]) is None
        assert calls == []

    async def test_warns_only_once_across_repeated_failures(self, monkeypatch, caplog):
        _patch_client(monkeypatch, lambda url, headers, json: _FakeResponse({}, status_code=404))
        await rag_embed.rerank("q", ["a"])
        await rag_embed.rerank("q", ["a"])
        warnings = [r for r in caplog.records if r.levelname == "WARNING" and "Reranker unavailable" in r.message]
        assert len(warnings) == 1

    async def test_recovers_and_warns_again_after_a_later_failure(self, monkeypatch, caplog):
        _patch_client(monkeypatch, lambda url, headers, json: _FakeResponse({}, status_code=404))
        await rag_embed.rerank("q", ["a"])  # fails, warns

        _patch_client(monkeypatch, lambda url, headers, json: _FakeResponse(
            {"results": [{"document": "a", "relevance_score": 1.0}]}
        ))
        await rag_embed.rerank("q", ["a"])  # succeeds, resets the warned flag

        _patch_client(monkeypatch, lambda url, headers, json: _FakeResponse({}, status_code=404))
        await rag_embed.rerank("q", ["a"])  # fails again, should warn again

        warnings = [r for r in caplog.records if r.levelname == "WARNING" and "Reranker unavailable" in r.message]
        assert len(warnings) == 2
