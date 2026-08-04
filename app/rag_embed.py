"""
MCP Auth Starter — RAG embedding + reranking client.

Embeddings go through the OpenAI-compatible /v1/embeddings shape (confirmed
against the user's own Ollama gateway). Reranking has no OpenAI-spec
equivalent, so it talks to Ollama's native /api/rerank instead, and is
opportunistic: if the reranker model isn't deployed yet, calls fail fast and
callers fall back to fusion-only ranking — see rerank()'s docstring.
"""

import asyncio
import logging

import httpx

from config import (
    RAG_EMBEDDING_MODEL,
    RAG_OLLAMA_API_KEY,
    RAG_OLLAMA_BASE_URL,
    RAG_RERANKER_BASE_URL,
    RAG_RERANKER_ENABLED,
    RAG_RERANKER_MODEL,
)

logger = logging.getLogger("mcp-auth-starter")

_EMBED_BATCH_SIZE = 32
_EMBED_CONCURRENCY = 8
_EMBED_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_RERANK_TIMEOUT = httpx.Timeout(20.0, connect=5.0)

_reranker_warned = False  # only log the "unavailable" warning once per outage


def _headers() -> dict:
    return {"Authorization": f"Bearer {RAG_OLLAMA_API_KEY}"} if RAG_OLLAMA_API_KEY else {}


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batches and bounds concurrency so ingesting a 150-page PDF's worth of
    chunks doesn't fire hundreds of simultaneous requests. Preserves input
    order regardless of batch/request completion order."""
    if not texts:
        return []
    batches = [texts[i:i + _EMBED_BATCH_SIZE] for i in range(0, len(texts), _EMBED_BATCH_SIZE)]
    semaphore = asyncio.Semaphore(_EMBED_CONCURRENCY)

    async def _one(batch: list[str]) -> list[list[float]]:
        async with semaphore, httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
            resp = await client.post(
                f"{RAG_OLLAMA_BASE_URL}/embeddings",
                headers=_headers(),
                json={"model": RAG_EMBEDDING_MODEL, "input": batch},
            )
            resp.raise_for_status()
            return [item["embedding"] for item in resp.json()["data"]]

    results = await asyncio.gather(*(_one(b) for b in batches))
    return [vec for batch_result in results for vec in batch_result]


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]


async def rerank(query: str, documents: list[str]) -> list[float] | None:
    """Cross-encoder relevance score per document, same order as input.

    Returns None (not raises) on any failure — unreachable host, model not
    pulled yet, unexpected response shape — so a reranker outage degrades
    search quality rather than breaking it. Always retries on the next call
    rather than giving up permanently, so it self-heals the moment the model
    is deployed; only the warning log is rate-limited to once per outage.
    """
    global _reranker_warned
    if not RAG_RERANKER_ENABLED or not documents:
        return None

    try:
        async with httpx.AsyncClient(timeout=_RERANK_TIMEOUT) as client:
            resp = await client.post(
                f"{RAG_RERANKER_BASE_URL}/api/rerank",
                headers=_headers(),
                json={"model": RAG_RERANKER_MODEL, "query": query, "documents": documents},
            )
            resp.raise_for_status()
            results = resp.json()["results"]
            score_by_text = {r["document"]: r["relevance_score"] for r in results}
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
        if not _reranker_warned:
            logger.warning(f"Reranker unavailable ({e}) — falling back to fusion-only ranking; will keep retrying")
            _reranker_warned = True
        return None

    _reranker_warned = False
    return [score_by_text.get(doc, float("-inf")) for doc in documents]
