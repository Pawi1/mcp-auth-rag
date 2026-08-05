import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

# Stub config so tests don't need a real config.json
_cfg = types.ModuleType("config")
_cfg.SECRET_KEY = "test-secret-key-32-chars-padding!"
_cfg.ALGORITHM = "HS256"
_cfg.SERVER_URL = "http://localhost:8000"
_cfg.DB_PATH = Path("/tmp/_test_mcp_auth_starter.db")
_cfg.REFRESH_TOKEN_EXPIRE_DAYS = 30
_cfg.ACCESS_TOKEN_EXPIRE_MINUTES = 60
_cfg.MCP_RESOURCE_URI = "http://localhost:8000/mcp"
_cfg.LOG_FILE = Path("/tmp/_test_mcp_auth_starter.log")
_cfg.MCP_HOST = "0.0.0.0"
_cfg.MCP_PORT = 8000
_cfg.MCP_SERVER_NAME = "mcp-auth-starter-test"
_cfg.DB_TIMEOUT = 5

# RAG
_cfg.RAG_POSTGRES_DSN = "postgresql://rag:rag@localhost:5433/rag_test"
_cfg.RAG_QDRANT_URL = "http://localhost:6333"
_cfg.RAG_QDRANT_COLLECTION = "rag_chunks_test"
_cfg.RAG_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_cfg.RAG_OLLAMA_API_KEY = ""
_cfg.RAG_EMBEDDING_MODEL = "bge-m3"
_cfg.RAG_EMBEDDING_DIM = 1024
_cfg.RAG_RERANKER_BASE_URL = "http://localhost:11434"
_cfg.RAG_RERANKER_MODEL = "bge-reranker-v2-m3"
_cfg.RAG_RERANKER_ENABLED = True
_cfg.RAG_CHUNK_TARGET_WORDS = 350
_cfg.RAG_CHUNK_OVERLAP_WORDS = 40
_cfg.RAG_CANDIDATE_K = 25
_cfg.RAG_TOP_K = 8
_cfg.RAG_UPLOAD_DIR = Path("/tmp/_test_mcp_auth_rag_uploads")
_cfg.RAG_MAX_UPLOAD_MB = 200
_cfg.RAG_OCR_LANGUAGES = "pol+eng"
_cfg.RAG_OCR_DPI = 200
_cfg.RAG_FTS_LANGUAGE = "simple"

sys.modules.setdefault("config", _cfg)
