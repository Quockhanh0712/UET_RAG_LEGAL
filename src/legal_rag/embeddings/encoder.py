"""Embedding model wrapper for Legal RAG retrieval."""

from __future__ import annotations

from typing import List, Optional

from src.embeddings import DEFAULT_MODEL, embed_query, embed_texts, get_embedding_dimension


class LegalEmbedding:
    """Wrapper around src.embeddings to unify ingestion and retrieval spaces."""

    def __init__(self, model_name: str = DEFAULT_MODEL, normalize_embeddings: Optional[bool] = None):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = 0

    def load(self):
        self.embedding_dim = int(get_embedding_dimension(self.model_name))
        return

    def encode(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        return embed_texts(
            texts,
            batch_size=batch_size,
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings,
            text_type="passage",
            add_prefix=True,
        )

    def encode_query(self, query: str) -> List[float]:
        return embed_query(
            query,
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings,
        )

