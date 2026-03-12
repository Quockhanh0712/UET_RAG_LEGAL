"""
Qdrant Vector Store for Legal RAG

Features:
- Dense vectors (DEk21 embedding)
- Sparse vectors (BM25-style)
- Hybrid search with RRF (Reciprocal Rank Fusion)
- Cross-Encoder Reranking
- Payload indexing for legal metadata
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector, NamedVector,
    Filter, FieldCondition, MatchValue,
    SearchParams, ScoredPoint
)

logger = logging.getLogger(__name__)

# Configuration
QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/qdrant_db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_USE_SERVER = os.getenv("QDRANT_USE_SERVER", "true").lower() == "true"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legal_documents")
DENSE_DIM = 768  # DEk21_hcmute_embedding dimension

# Reranker settings
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "20"))  # Fetch more for reranking

# Hybrid search weights
DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_SPARSE_WEIGHT = 0.4


@dataclass
class SearchResult:
    """Search result with metadata"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict
    
    @property
    def van_ban_id(self) -> str:
        return self.metadata.get("van_ban_id", "")
    
    @property
    def dieu_so(self) -> str:
        return self.metadata.get("dieu_so", "")
    
    @property
    def ten_van_ban(self) -> str:
        return self.metadata.get("ten_van_ban", "")


class BM25Encoder:
    """Simple BM25-style sparse encoder for Vietnamese legal text"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = {}
        self.avg_doc_len = 0
        self.doc_count = 0
        
        # Try to load pyvi for Vietnamese tokenization
        try:
            from pyvi import ViTokenizer
            self.tokenizer = ViTokenizer.tokenize
            logger.info("[BM25] Using pyvi Vietnamese tokenizer")
        except ImportError:
            self.tokenizer = lambda x: x
            logger.warning("[BM25] pyvi not found, using simple tokenizer")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text"""
        text = self.tokenizer(text.lower())
        # Split by spaces and underscores (pyvi joins words with _)
        tokens = []
        for word in text.split():
            # Keep compound words from pyvi
            tokens.append(word.replace("_", " "))
        return tokens
    
    def fit(self, documents: List[str]):
        """Build vocabulary and IDF from documents"""
        import math
        
        self.doc_count = len(documents)
        doc_freq = {}
        total_len = 0
        
        for doc in documents:
            tokens = self.tokenize(doc)
            total_len += len(tokens)
            
            # Count unique tokens per doc for DF
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
                
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 1
        
        # Calculate IDF
        for token, df in doc_freq.items():
            self.idf[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
        
        logger.info(f"[BM25] Vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """Encode text to sparse vector (indices, values)"""
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        
        # Count term frequency
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        indices = []
        values = []
        
        for token, freq in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf = self.idf.get(token, 0)
                
                # BM25 score
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score = idf * (numerator / denominator)
                
                indices.append(idx)
                values.append(float(score))
        
        return indices, values
    
    def save(self, path: str):
        """Save encoder to file"""
        import json
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "avg_doc_len": self.avg_doc_len,
            "doc_count": self.doc_count,
            "k1": self.k1,
            "b": self.b
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"[BM25] Saved encoder to {path}")
    
    def load(self, path: str):
        """Load encoder from file"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.avg_doc_len = data["avg_doc_len"]
        self.doc_count = data["doc_count"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        logger.info(f"[BM25] Loaded encoder with {len(self.vocab)} vocab")


class QdrantLegalStore:
    """
    Qdrant Vector Store for Vietnamese Legal Documents
    
    Features:
    - Hybrid search (dense + sparse)
    - Metadata filtering
    - Payload indexing
    """
    
    def __init__(
        self,
        path: str = QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
        force_unlock: bool = True,
        use_server: bool = QDRANT_USE_SERVER,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT
    ):
        self.path = path
        self.collection_name = collection_name
        self.use_server = use_server
        
        # Initialize Qdrant client
        if use_server:
            # Server mode - connect to Qdrant server (Docker)
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"[QDRANT] Connected to server at {host}:{port}")
        else:
            # Local mode - SQLite storage
            os.makedirs(path, exist_ok=True)
            if force_unlock:
                self._remove_lock_file()
            self.client = QdrantClient(path=path)
            logger.info(f"[QDRANT] Initialized local storage at {path}")
        
        # BM25 encoder for sparse vectors
        self.bm25_encoder = BM25Encoder()
        self.bm25_path = os.path.join(path, "bm25_encoder.json")
        
        # Load BM25 if exists
        if os.path.exists(self.bm25_path):
            self.bm25_encoder.load(self.bm25_path)
        
        logger.info(f"[QDRANT] Ready (server={use_server})")
    
    def _remove_lock_file(self):
        """Remove Qdrant lock file to allow reconnection"""
        import glob
        lock_patterns = [
            os.path.join(self.path, ".lock"),
            os.path.join(self.path, "*.lock"),
            os.path.join(self.path, "storage.lock"),
        ]
        
        for pattern in lock_patterns:
            for lock_file in glob.glob(pattern):
                try:
                    os.remove(lock_file)
                    logger.info(f"[QDRANT] Removed lock file: {lock_file}")
                except Exception as e:
                    logger.warning(f"[QDRANT] Could not remove lock: {e}")
    
    def create_collection(self, force_recreate: bool = False):
        """Create collection with dense + sparse vectors"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists:
            if force_recreate:
                self.client.delete_collection(self.collection_name)
                logger.info(f"[QDRANT] Deleted existing collection: {self.collection_name}")
            else:
                logger.info(f"[QDRANT] Collection exists: {self.collection_name}")
                return
        
        # Create collection with hybrid vectors
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )
        
        # Create payload indexes for filtering
        self._create_payload_indexes()
        
        logger.info(f"[QDRANT] Created collection: {self.collection_name}")
    
    def _create_payload_indexes(self):
        """Create indexes for efficient filtering"""
        indexed_fields = [
            ("van_ban_id", "keyword"),
            ("loai_van_ban", "keyword"),
            ("co_quan", "keyword"),
            ("dieu_so", "keyword"),
            ("chuong", "keyword")
        ]
        
        for field_name, field_type in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"[QDRANT] Created index for: {field_name}")
            except Exception as e:
                logger.debug(f"[QDRANT] Index may exist: {field_name}")
    
    def add_documents(
        self,
        chunks: List,  # List[LegalChunk]
        embeddings: List[List[float]],
        batch_size: int = 100
    ):
        """Add documents with dense and sparse vectors"""
        
        # First, fit BM25 encoder on all documents
        contents = [chunk.content for chunk in chunks]
        self.bm25_encoder.fit(contents)
        self.bm25_encoder.save(self.bm25_path)
        
        # Add in batches
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            points = []
            for j, (chunk, dense_vec) in enumerate(zip(batch_chunks, batch_embeddings)):
                # Generate sparse vector
                sparse_indices, sparse_values = self.bm25_encoder.encode(chunk.content)
                
                # Create point ID from chunk_id hash
                point_id = int(hashlib.md5(chunk.chunk_id.encode()).hexdigest()[:16], 16)
                
                # Prepare payload (metadata)
                payload = chunk.to_dict()
                
                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        )
                    },
                    payload=payload
                )
                points.append(point)
            
            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"[QDRANT] Indexed {min(i + batch_size, total)}/{total} documents")
    
    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        filter_conditions: Optional[Dict] = None,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense and sparse vectors
        
        Args:
            query: Query text (for sparse search)
            query_embedding: Dense embedding of query
            top_k: Number of results
            dense_weight: Weight for dense vector (α)
            sparse_weight: Weight for sparse vector (β)
            filter_conditions: Optional metadata filters
            collection_name: Optional custom collection name (default: self.collection_name)
        """
        # Build sparse query vector
        sparse_indices, sparse_values = self.bm25_encoder.encode(query)
        
        # Build filter if provided
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for field, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            qdrant_filter = Filter(must=conditions)
        
        # Use custom collection name if provided
        target_collection = collection_name if collection_name else self.collection_name
        
        # Perform hybrid search using Qdrant's prefetch + fusion
        results = self.client.query_points(
            collection_name=target_collection,
            prefetch=[
                models.Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=top_k * 2  # Fetch more for fusion
                ),
                models.Prefetch(
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    ),
                    using="sparse",
                    limit=top_k * 2
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
            ),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True
        )
        
        # Convert to SearchResult
        search_results = []
        for point in results.points:
            search_results.append(SearchResult(
                chunk_id=point.payload.get("chunk_id", ""),
                content=point.payload.get("content", ""),
                score=point.score,
                metadata=point.payload
            ))
        
        return search_results
    
    def search_by_metadata(
        self,
        van_ban_id: Optional[str] = None,
        loai_van_ban: Optional[str] = None,
        dieu_so: Optional[str] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """Search by metadata filters only"""
        conditions = []
        
        if van_ban_id:
            conditions.append(FieldCondition(key="van_ban_id", match=MatchValue(value=van_ban_id)))
        if loai_van_ban:
            conditions.append(FieldCondition(key="loai_van_ban", match=MatchValue(value=loai_van_ban)))
        if dieu_so:
            conditions.append(FieldCondition(key="dieu_so", match=MatchValue(value=dieu_so)))
        
        if not conditions:
            return []
        
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True
        )
        
        search_results = []
        for point in results[0]:
            search_results.append(SearchResult(
                chunk_id=point.payload.get("chunk_id", ""),
                content=point.payload.get("content", ""),
                score=1.0,  # Exact match
                metadata=point.payload
            ))
        
        return search_results
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', info.points_count),
                "status": info.status.value
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_by_van_ban(self, van_ban_id: str):
        """Delete all chunks from a specific van_ban"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="van_ban_id",
                            match=MatchValue(value=van_ban_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"[QDRANT] Deleted documents for: {van_ban_id}")
    
    def hybrid_search_with_rerank(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        use_reranker: bool = USE_RERANKER,
        rerank_weight: float = 0.6,
        filter_conditions: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Advanced hybrid search with RRF + Cross-Encoder Reranking
        
        Pipeline:
        1. Dense search (semantic) with Qdrant
        2. Sparse search (BM25) with Qdrant
        3. RRF fusion (built-in Qdrant)
        4. Cross-Encoder rerank (optional, improves quality)
        
        Args:
            query: Query text
            query_embedding: Dense embedding of query
            top_k: Number of final results
            use_reranker: Whether to use cross-encoder reranking
            rerank_weight: Weight for reranker scores (0=original, 1=rerank)
            filter_conditions: Optional metadata filters
            
        Returns:
            List of SearchResult sorted by final score
        """
        import time
        start = time.time()
        
        # Fetch more candidates for reranking
        fetch_k = top_k * 3 if use_reranker else top_k
        
        # Get initial results using Qdrant's built-in RRF
        initial_results = self.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=fetch_k,
            filter_conditions=filter_conditions
        )
        
        if not initial_results:
            return []
        
        # Apply cross-encoder reranking if enabled
        if use_reranker and len(initial_results) > 0:
            try:
                from src.reranker import LegalReranker
                
                # Lazy load reranker
                if not hasattr(self, '_reranker') or self._reranker is None:
                    self._reranker = LegalReranker()
                
                # Prepare documents for reranking
                docs_for_rerank = [
                    {
                        "content": r.content,
                        "score": r.score,
                        "chunk_id": r.chunk_id,
                        "metadata": r.metadata
                    }
                    for r in initial_results
                ]
                
                # Rerank
                reranked = self._reranker.rerank(
                    query=query,
                    documents=docs_for_rerank,
                    top_k=top_k,
                    rrf_weight=rerank_weight
                )
                
                # Convert back to SearchResult
                reranked_results = []
                for r in reranked:
                    reranked_results.append(SearchResult(
                        chunk_id=r.metadata.get("chunk_id", ""),
                        content=r.content,
                        score=r.final_score,
                        metadata=r.metadata.get("metadata", r.metadata)
                    ))
                
                elapsed = time.time() - start
                logger.info(f"[QDRANT] Hybrid search + rerank: {len(reranked_results)} results in {elapsed:.2f}s")
                
                return reranked_results
                
            except ImportError as e:
                logger.warning(f"[QDRANT] Reranker not available: {e}")
            except Exception as e:
                logger.warning(f"[QDRANT] Reranking failed, using RRF results: {e}")
        
        # Return RRF results without reranking
        elapsed = time.time() - start
        logger.info(f"[QDRANT] Hybrid search (RRF only): {len(initial_results[:top_k])} results in {elapsed:.2f}s")
        
        return initial_results[:top_k]


# Singleton instance with thread safety
_qdrant_store: Optional[QdrantLegalStore] = None
_qdrant_lock = None

def _get_lock():
    """Get or create threading lock"""
    global _qdrant_lock
    if _qdrant_lock is None:
        import threading
        _qdrant_lock = threading.Lock()
    return _qdrant_lock

def get_qdrant_store(max_retries: int = 3) -> QdrantLegalStore:
    """Get singleton Qdrant store instance with retry logic"""
    global _qdrant_store
    
    # Return existing instance if available
    if _qdrant_store is not None:
        return _qdrant_store
    
    lock = _get_lock()
    with lock:
        # Double-check after acquiring lock
        if _qdrant_store is not None:
            return _qdrant_store
        
        import time
        last_error = None
        
        for attempt in range(max_retries):
            try:
                _qdrant_store = QdrantLegalStore()
                return _qdrant_store
            except RuntimeError as e:
                if "already accessed" in str(e):
                    last_error = e
                    logger.warning(f"[QDRANT] Retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1)  # Wait before retry
                else:
                    raise
        
        raise RuntimeError(f"Failed to initialize Qdrant after {max_retries} retries: {last_error}")


def reset_qdrant_store():
    """Reset singleton (useful for testing or reload)"""
    global _qdrant_store
    if _qdrant_store is not None:
        try:
            _qdrant_store.client.close()
        except:
            pass
        _qdrant_store = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    store = QdrantLegalStore()
    store.create_collection(force_recreate=True)
    
    print("Stats:", store.get_stats())
