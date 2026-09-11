from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MARKDOWN_DIR = DATA_DIR / "markdown"


# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gpt-oss:20b-cloud"


# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "organization_documents"


# RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


# Embedding dimensions for nomic-embed-text
EMBEDDING_DIMENSIONS = 768