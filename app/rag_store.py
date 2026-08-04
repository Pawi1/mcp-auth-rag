"""
MCP Auth Starter — RAG storage. Postgres holds document/chunk metadata, chunk
text, full-text search, and the ingest job queue; Qdrant holds only the
vectors + enough payload to filter by owner/document. Both live outside
app.db (see docker-compose.rag.yml) — very different scale and access
pattern than the auth tables, so sharing one SQLite file made no sense.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from config import RAG_EMBEDDING_DIM, RAG_POSTGRES_DSN, RAG_QDRANT_COLLECTION, RAG_QDRANT_URL, RAG_UPLOAD_DIR

logger = logging.getLogger("mcp-auth-starter")

_pg_pool: Optional[asyncpg.Pool] = None
_qdrant: Optional[AsyncQdrantClient] = None


async def init_stores() -> None:
    """Called once from main.py's lifespan. Idempotent — safe to call again."""
    global _pg_pool, _qdrant
    _pg_pool = await asyncpg.create_pool(RAG_POSTGRES_DSN, min_size=2, max_size=10)
    _qdrant = AsyncQdrantClient(url=RAG_QDRANT_URL)
    await _ensure_schema()
    await _ensure_collection()
    logger.info("RAG stores ready (Postgres + Qdrant)")


async def close_stores() -> None:
    global _pg_pool, _qdrant
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None
    if _qdrant is not None:
        await _qdrant.close()
        _qdrant = None


def pg_pool() -> asyncpg.Pool:
    if _pg_pool is None:
        raise RuntimeError("RAG Postgres pool not initialized — call init_stores() first")
    return _pg_pool


def qdrant() -> AsyncQdrantClient:
    if _qdrant is None:
        raise RuntimeError("RAG Qdrant client not initialized — call init_stores() first")
    return _qdrant


async def _ensure_schema() -> None:
    async with _pg_pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")  # gen_random_uuid()
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                owner TEXT NOT NULL,
                filename TEXT NOT NULL,
                format TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT,
                page_count INT,
                chunk_count INT NOT NULL DEFAULT 0,
                uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                processed_at TIMESTAMPTZ,
                UNIQUE (owner, content_hash)
            )
        """)
        # to_tsvector('simple', text) in a GENERATED column is the documented
        # supported pattern (PG treats a literal regconfig argument as
        # immutable enough). 'simple' (tokenize + lowercase, no stemming) —
        # not 'polish', which doesn't exist: Postgres's built-in configs are
        # Snowball-based, and Snowball has no Polish stemmer. Getting real
        # Polish stemming means installing an ispell dictionary into the
        # Postgres image and registering a TEXT SEARCH CONFIGURATION for it
        # — a real upgrade, just not one this schema forces on day one.
        # RRF fusion (rag_retrieval.py) means Qdrant's semantic search still
        # covers for the stemming this gives up.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INT NOT NULL,
                page INT,
                section TEXT,
                text TEXT NOT NULL,
                word_count INT NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
                UNIQUE (document_id, ordinal)
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv)")
        await conn.execute("CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id)")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ingest_jobs (
                id BIGSERIAL PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'queued',
                attempts INT NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                claimed_at TIMESTAMPTZ
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS ingest_jobs_status_idx ON ingest_jobs (status, id)")


async def _ensure_collection() -> None:
    existing = {c.name for c in (await _qdrant.get_collections()).collections}
    if RAG_QDRANT_COLLECTION in existing:
        return
    await _qdrant.create_collection(
        collection_name=RAG_QDRANT_COLLECTION,
        vectors_config=VectorParams(size=RAG_EMBEDDING_DIM, distance=Distance.COSINE),
    )
    # payload indexes created before any points land, so the HNSW graph is
    # built with filtering already in mind (Qdrant's own tuning guidance)
    await _qdrant.create_payload_index(RAG_QDRANT_COLLECTION, field_name="owner", field_schema="keyword")
    await _qdrant.create_payload_index(RAG_QDRANT_COLLECTION, field_name="document_id", field_schema="keyword")


def upload_path(document_id, filename: str) -> Path:
    return RAG_UPLOAD_DIR / f"{document_id}{Path(filename).suffix.lower()}"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

async def create_document(owner: str, filename: str, fmt: str, content_hash: str) -> Optional[dict]:
    """Returns the new/reset row, or None if this owner already has this
    exact content ingested or in progress (content_hash dedup — no point
    re-ingesting the same PDF). A prior *failed* attempt (status='error')
    is reset to pending and retried instead of silently skipped — "fix the
    bug, re-upload the same file" is the normal recovery path, and without
    this it'd be deduped away forever. Any chunks left over from that failed
    attempt are purged first, so the retry can't collide on (document_id,
    ordinal) — a no-op for a genuinely new document, which has none."""
    async with pg_pool().acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """INSERT INTO documents (owner, filename, format, content_hash)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (owner, content_hash) DO UPDATE SET
                       filename = EXCLUDED.filename,
                       status = 'pending', error = NULL, chunk_count = 0,
                       uploaded_at = now(), processed_at = NULL
                   WHERE documents.status = 'error'
                   RETURNING *""",
                owner, filename, fmt, content_hash,
            )
            if row is None:
                return None
            stale_chunk_ids = [
                r["id"] for r in await conn.fetch("SELECT id FROM chunks WHERE document_id=$1", row["id"])
            ]
            if stale_chunk_ids:
                await conn.execute("DELETE FROM chunks WHERE document_id=$1", row["id"])

    if stale_chunk_ids:
        await qdrant().delete(collection_name=RAG_QDRANT_COLLECTION, points_selector=[str(c) for c in stale_chunk_ids])

    return dict(row)


async def get_document(document_id, owner: Optional[str] = None) -> Optional[dict]:
    async with pg_pool().acquire() as conn:
        if owner is not None:
            row = await conn.fetchrow("SELECT * FROM documents WHERE id=$1 AND owner=$2", document_id, owner)
        else:
            row = await conn.fetchrow("SELECT * FROM documents WHERE id=$1", document_id)
        return dict(row) if row else None


async def list_documents(owner: str) -> list[dict]:
    async with pg_pool().acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, filename, format, status, error, page_count, chunk_count, uploaded_at, processed_at
               FROM documents WHERE owner=$1 ORDER BY uploaded_at DESC""",
            owner,
        )
        return [dict(r) for r in rows]


async def delete_document(document_id, owner: str) -> bool:
    async with pg_pool().acquire() as conn:
        chunk_ids = [r["id"] for r in await conn.fetch("SELECT id FROM chunks WHERE document_id=$1", document_id)]
        doc = await conn.fetchrow("SELECT filename FROM documents WHERE id=$1 AND owner=$2", document_id, owner)
        if doc is None:
            return False
        await conn.execute("DELETE FROM documents WHERE id=$1 AND owner=$2", document_id, owner)

    if chunk_ids:
        await qdrant().delete(collection_name=RAG_QDRANT_COLLECTION, points_selector=[str(c) for c in chunk_ids])

    path = upload_path(document_id, doc["filename"])
    path.unlink(missing_ok=True)
    return True


async def set_document_status(document_id, status: str, error: str = None, page_count: int = None) -> None:
    async with pg_pool().acquire() as conn:
        await conn.execute(
            """UPDATE documents SET status=$2, error=$3,
                   page_count=COALESCE($4, page_count),
                   processed_at = CASE WHEN $2 IN ('done', 'error') THEN now() ELSE processed_at END
               WHERE id=$1""",
            document_id, status, error, page_count,
        )


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

async def insert_chunks(document_id, owner: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """chunks: [{ordinal, page, section, text, word_count}, ...], same order as embeddings."""
    if not chunks:
        return
    assert len(chunks) == len(embeddings)
    ids = [uuid.uuid4() for _ in chunks]

    async with pg_pool().acquire() as conn:
        # explicit transaction: without it, executemany() commits each row
        # as it goes (no implicit all-or-nothing), so one bad row partway
        # through a batch would leave the rest permanently half-inserted.
        async with conn.transaction():
            await conn.executemany(
                """INSERT INTO chunks (id, document_id, ordinal, page, section, text, word_count)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                [
                    (ids[i], document_id, c["ordinal"], c["page"], c["section"], c["text"], c["word_count"])
                    for i, c in enumerate(chunks)
                ],
            )
            await conn.execute("UPDATE documents SET chunk_count = chunk_count + $2 WHERE id=$1", document_id, len(chunks))

    points = [
        PointStruct(
            id=str(ids[i]),
            vector=embeddings[i],
            payload={"document_id": str(document_id), "owner": owner, "page": c["page"], "section": c["section"]},
        )
        for i, c in enumerate(chunks)
    ]
    await qdrant().upsert(collection_name=RAG_QDRANT_COLLECTION, points=points)


async def fetch_chunks_by_id(chunk_ids: list[str]) -> dict[str, dict]:
    """Batch-hydrates Qdrant's id-only hits with actual text/metadata, keyed by str(id)."""
    if not chunk_ids:
        return {}
    async with pg_pool().acquire() as conn:
        rows = await conn.fetch(
            """SELECT c.id, c.document_id, c.page, c.section, c.text, d.filename
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.id = ANY($1::uuid[])""",
            [uuid.UUID(cid) for cid in chunk_ids],
        )
        return {str(r["id"]): dict(r) for r in rows}


async def fts_search(query: str, owner: str, limit: int) -> list[str]:
    """Full-text search over chunk content (keyword match, no stemming — see
    the 'simple' vs. 'polish' note on the chunks table above). Returns chunk
    id strings ranked best-first (rank position is all retrieval.py needs —
    see reciprocal_rank_fusion)."""
    async with pg_pool().acquire() as conn:
        rows = await conn.fetch(
            """SELECT c.id
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE d.owner = $1 AND c.tsv @@ websearch_to_tsquery('simple', $2)
               ORDER BY ts_rank(c.tsv, websearch_to_tsquery('simple', $2)) DESC
               LIMIT $3""",
            owner, query, limit,
        )
        return [str(r["id"]) for r in rows]


async def vector_search(vector: list[float], owner: str, limit: int) -> list[str]:
    """Qdrant ANN search, ranked best-first. Same contract as fts_search."""
    hits = await qdrant().query_points(
        collection_name=RAG_QDRANT_COLLECTION,
        query=vector,
        query_filter=Filter(must=[FieldCondition(key="owner", match=MatchValue(value=owner))]),
        limit=limit,
    )
    return [str(p.id) for p in hits.points]


# ---------------------------------------------------------------------------
# Ingest job queue — Postgres as the broker (FOR UPDATE SKIP LOCKED), so
# multiple worker processes/pods can poll the same table without stepping on
# each other's claims. No Redis/Celery needed at this scale.
# ---------------------------------------------------------------------------

async def enqueue_ingest_job(document_id) -> None:
    async with pg_pool().acquire() as conn:
        await conn.execute("INSERT INTO ingest_jobs (document_id) VALUES ($1)", document_id)


async def claim_next_job() -> Optional[dict]:
    async with pg_pool().acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """SELECT * FROM ingest_jobs
                   WHERE status = 'queued'
                   ORDER BY id
                   FOR UPDATE SKIP LOCKED
                   LIMIT 1"""
            )
            if row is None:
                return None
            await conn.execute(
                "UPDATE ingest_jobs SET status='running', claimed_at=now(), attempts=attempts+1 WHERE id=$1",
                row["id"],
            )
            return dict(row)


async def finish_job(job_id: int, ok: bool, error: str = "") -> None:
    async with pg_pool().acquire() as conn:
        await conn.execute(
            "UPDATE ingest_jobs SET status=$2, last_error=$3 WHERE id=$1",
            job_id, "done" if ok else "error", error,
        )
