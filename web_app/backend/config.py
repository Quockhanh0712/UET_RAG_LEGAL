"""
Backend Configuration Settings
"""
import os
from pathlib import Path
from typing import Optional

from src.config.embedding_contract import DEFAULT_LEGAL_EMBEDDING_MODEL


class Settings:
    """Application settings with environment variable support."""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    
    # Qdrant Settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_LEGAL_COLLECTION: str = "legal_rag_hybrid"
    QDRANT_USER_COLLECTION: str = "user_docs_private"
    
    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", DEFAULT_LEGAL_EMBEDDING_MODEL)
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cuda")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"
    
    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # Reranker Settings
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "false").lower() == "true"
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    
    # Search Settings
    TOP_K: int = int(os.getenv("TOP_K", "10"))
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"
    
    def __init__(self):
        """Initialize settings and create directories."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
