"""
MCP Auth Starter — configuration. Loads from config.json, secrets from env vars.
"""

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent

if os.getenv("MCP_CONFIG_PATH"):
    _cfg_path = Path(os.getenv("MCP_CONFIG_PATH"))
elif getattr(sys, "frozen", False):
    # When frozen by PyInstaller, config.json lives next to the binary, not in _MEIPASS
    _cfg_path = Path(sys.executable).parent / "config.json"
else:
    _cfg_path = _HERE.parent / "config.json"

_cfg: dict = json.loads(_cfg_path.read_text(encoding="utf-8")) if _cfg_path.exists() else {}


def _p(keys: str, default=None):
    """Dot-path lookup in _cfg, e.g. 'paths.data_root'."""
    node = _cfg
    for k in keys.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node if node is not None else default


# Security — secrets always from env vars
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"

# Paths
DATA_ROOT = Path(_p("paths.data_root", "/srv/mcp-auth-starter"))
LOG_DIR   = Path(os.getenv("LOG_DIR", _p("paths.log_dir", str(DATA_ROOT / "tmp" / "logs"))))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE  = LOG_DIR / "mcp_server.log"

DB_PATH = DATA_ROOT / "app.db"
DB_TIMEOUT = 30

# Server
SERVER_URL         = os.getenv("SERVER_URL", _p("server.url", "http://localhost:8000"))
MCP_SERVER_NAME    = _p("server.name", "mcp-auth-starter")
MCP_SERVER_VERSION = "0.1.0"
MCP_HOST           = os.getenv("MCP_HOST", _p("server.host", "0.0.0.0"))
MCP_PORT           = int(os.getenv("MCP_PORT", str(_p("server.port", 8000))))

# Auth
# token_expire_days now governs the long-lived refresh_token; the bearer access
# token sent on every MCP request is short-lived (access_token_expire_minutes)
# so a leaked one has a small blast radius — the MCP client refreshes silently.
REFRESH_TOKEN_EXPIRE_DAYS   = int(_p("auth.token_expire_days", 90))
ACCESS_TOKEN_EXPIRE_MINUTES = int(_p("auth.access_token_expire_minutes", 60))

# Canonical resource URI for this MCP server — RFC 8707 / RFC 9728 audience
# binding, so a token issued here can't be replayed against a different
# resource server even if it shared the same signing key.
MCP_RESOURCE_URI = f"{SERVER_URL.rstrip('/')}/mcp"

# RAG — document storage lives outside app.db (own Postgres + Qdrant, see
# docker-compose.rag.yml) since the corpus/vector-search workload has very
# different scale and access patterns than the auth tables above.
RAG_POSTGRES_DSN = os.getenv("RAG_POSTGRES_DSN", _p("rag.postgres.dsn", "postgresql://rag:rag@localhost:5433/rag"))
RAG_QDRANT_URL        = os.getenv("RAG_QDRANT_URL", _p("rag.qdrant.url", "http://localhost:6333"))
RAG_QDRANT_COLLECTION = _p("rag.qdrant.collection", "rag_chunks")

# Embedding/reranking provider — an OpenAI-compatible endpoint in front of
# Ollama. RAG_OLLAMA_API_KEY is optional (omit the header entirely if unset,
# rather than send an empty bearer token).
RAG_OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL", _p("rag.embedding.base_url", "http://localhost:11434/v1"))
RAG_OLLAMA_API_KEY  = os.getenv("RAG_OLLAMA_API_KEY", "")
RAG_EMBEDDING_MODEL = _p("rag.embedding.model", "bge-m3")
RAG_EMBEDDING_DIM   = int(_p("rag.embedding.dimensions", 1024))

# Reranking is a native-Ollama endpoint (no OpenAI equivalent exists), so it
# talks to the same host with the "/v1" suffix stripped. It's attempted
# opportunistically — see app/rag/embed.py — and skipped if the model isn't
# deployed yet, so this doesn't have to be true on day one.
RAG_RERANKER_BASE_URL = _p("rag.reranker.base_url", RAG_OLLAMA_BASE_URL.removesuffix("/v1").removesuffix("/"))
RAG_RERANKER_MODEL    = _p("rag.reranker.model", "bge-reranker-v2-m3")
RAG_RERANKER_ENABLED  = bool(_p("rag.reranker.enabled", True))

# Chunking — word counts, not tokens: bge-m3 isn't a tiktoken-family model, so
# an exact token count would need pulling in its tokenizer as a dependency
# just to size chunks. Word count is a fine proxy for "roughly consistent
# chunk size" and needs nothing extra.
RAG_CHUNK_TARGET_WORDS  = int(_p("rag.chunking.target_words", 350))
RAG_CHUNK_OVERLAP_WORDS = int(_p("rag.chunking.overlap_words", 40))

# Full-text search language — a Postgres regconfig name (see rag_store.py's
# schema comment for why this defaults to 'simple' rather than assuming a
# language). Set to 'english' etc. if your corpus is actually one language;
# wrong-language stemming (e.g. English rules on Polish text) can do more
# harm than 'simple''s plain no-stemming tokenization.
RAG_FTS_LANGUAGE = _p("rag.fts_language", "simple")

# OCR fallback for scanned PDF pages (no text layer) — via PyMuPDF's built-in
# Tesseract integration, so no extra Python dependency, only the system
# tesseract-ocr binary + language packs (see README). Best-effort: a page
# just stays textless if OCR isn't available or fails.
RAG_OCR_LANGUAGES = _p("rag.ocr.languages", "pol+eng")
RAG_OCR_DPI       = int(_p("rag.ocr.dpi", 200))

# Retrieval
RAG_CANDIDATE_K = int(_p("rag.retrieval.candidate_k", 25))  # fused before rerank
RAG_TOP_K       = int(_p("rag.retrieval.top_k", 8))          # returned to the caller

# Uploads
RAG_UPLOAD_DIR = DATA_ROOT / "rag_uploads"
RAG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RAG_MAX_UPLOAD_MB = int(_p("rag.max_upload_mb", 200))

# Setup state — used by startup checks to detect missing config
CONFIG_PATH  = _cfg_path
CONFIG_FOUND = _cfg_path.exists()
