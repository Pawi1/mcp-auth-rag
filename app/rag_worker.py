"""
MCP Auth Starter — RAG ingest worker: polls ingest_jobs (Postgres, SKIP
LOCKED) and runs ingest_document() for each. In-process asyncio task, not a
separate service — the SKIP LOCKED claim means it's also safe to run this
as more than one process/pod against the same Postgres later without any
code change, if ingest volume ever outgrows one process.
"""

import asyncio
import logging

logger = logging.getLogger("mcp-auth-starter")

_POLL_INTERVAL_SECONDS = 2.0
_task: asyncio.Task | None = None


async def _run_forever() -> None:
    import rag_store
    from rag_ingest import ingest_document

    while True:
        try:
            job = await rag_store.claim_next_job()
        except Exception:
            logger.exception("RAG worker: failed to poll job queue")
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            continue

        if job is None:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            continue

        document = await rag_store.get_document(job["document_id"])
        if document is None:
            await rag_store.finish_job(job["id"], ok=False, error="document deleted before ingest ran")
            continue

        try:
            path = rag_store.upload_path(document["id"], document["filename"])
            data = path.read_bytes()
            await rag_store.set_document_status(document["id"], "processing")
            await ingest_document(document["id"], document["owner"], document["filename"], document["format"], data)
            await rag_store.finish_job(job["id"], ok=True)
        except Exception as e:
            await rag_store.finish_job(job["id"], ok=False, error=str(e))


def start() -> None:
    global _task
    if _task is None:
        _task = asyncio.create_task(_run_forever())
        logger.info("RAG ingest worker started")


def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
