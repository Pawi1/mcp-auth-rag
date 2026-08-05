"""
MCP Auth Starter — RAG retrieval: hybrid search (Qdrant dense ANN + Postgres
full-text) fused by Reciprocal Rank Fusion, then optionally reranked by a
cross-encoder for a final precision pass.

    query -> embed -> [Qdrant ANN, Postgres FTS] -> RRF -> rerank -> top_k

RRF combines by rank position, not raw score, which is why it doesn't matter
that ts_rank and cosine similarity live on completely different scales.
"""

import logging

from config import RAG_CANDIDATE_K, RAG_TOP_K

logger = logging.getLogger("mcp-auth-starter")

_RRF_K = 60  # standard constant (Cormack et al.) — dampens the influence of any single ranked list
_CONTEXT_WINDOW_MAX = 10  # clamp chunks_before/chunks_after so get_context() can't be asked for a whole document


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = _RRF_K) -> list[str]:
    """Each input is a list of ids, best-first. Returns fused ids, best-first.

    Pure function — no I/O — so this is the one piece of retrieval.py that's
    fully unit-testable without mocking Postgres/Qdrant.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


async def search(query: str, owner: str, top_k: int = RAG_TOP_K, candidate_k: int = RAG_CANDIDATE_K) -> list[dict]:
    """Returns up to top_k results: [{text, filename, page, section, document_id, ordinal}, ...].
    ordinal is a chunk's position within its document — pass it (with
    document_id) to get_context() to pull more of that document around a
    result that looks worth reading further."""
    import rag_embed
    import rag_store

    if not query.strip():
        return []

    query_vector = await rag_embed.embed_query(query)
    vector_ids, fts_ids = [], []
    try:
        vector_ids = await rag_store.vector_search(query_vector, owner, candidate_k)
    except Exception:
        logger.exception("Vector search failed")
    try:
        fts_ids = await rag_store.fts_search(query, owner, candidate_k)
    except Exception:
        logger.exception("Full-text search failed")

    fused_ids = reciprocal_rank_fusion([vector_ids, fts_ids])[:candidate_k]
    if not fused_ids:
        return []

    chunks_by_id = await rag_store.fetch_chunks_by_id(fused_ids)
    candidates = [chunks_by_id[cid] for cid in fused_ids if cid in chunks_by_id]

    ordered = await _rerank_or_keep_order(query, candidates)
    return [
        {
            "text": c["text"],
            "filename": c["filename"],
            "page": c["page"],
            "section": c["section"],
            "document_id": str(c["document_id"]),
            "ordinal": c["ordinal"],
        }
        for c in ordered[:top_k]
    ]


async def get_context(
    document_id: str, ordinal: int, owner: str, chunks_before: int = 2, chunks_after: int = 2
) -> dict | None:
    """Expands a single search hit (document_id + ordinal, both from a
    search() result) into a larger contiguous passage — the chunks
    immediately before/after it, concatenated in order. Returns None if the
    document doesn't exist, isn't owned by `owner`, or that ordinal is out
    of range."""
    import rag_store

    chunks_before = max(0, min(chunks_before, _CONTEXT_WINDOW_MAX))
    chunks_after = max(0, min(chunks_after, _CONTEXT_WINDOW_MAX))

    chunks = await rag_store.get_context_chunks(document_id, owner, ordinal, chunks_before, chunks_after)
    if not chunks:
        return None

    return {
        "filename": chunks[0]["filename"],
        "text": "\n\n".join(c["text"] for c in chunks),
        "start_ordinal": chunks[0]["ordinal"],
        "end_ordinal": chunks[-1]["ordinal"],
        "pages": sorted({c["page"] for c in chunks if c["page"] is not None}),
    }


async def _rerank_or_keep_order(query: str, candidates: list[dict]) -> list[dict]:
    import rag_embed

    if not candidates:
        return candidates
    scores = await rag_embed.rerank(query, [c["text"] for c in candidates])
    if scores is None:
        return candidates  # reranker unavailable — RRF fusion order stands
    return [c for _, c in sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)]
