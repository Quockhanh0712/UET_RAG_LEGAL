"""
User Document Store - Private Collection per User/Session

Kiến trúc Dual-Store:
1. legal_rag_hybrid_full - Kho Luật chính (Read-Only, Global)
2. user_docs_private - Kho Cá nhân (Read/Write/Delete, Isolated by user_id/session_id)

Features:
- Session-based document isolation
- Automatic cleanup on session end
- Filter by user_id mandatory
- Smart chunking for uploaded files

Author: Legal RAG System
Date: 2024-12
"""

import os
import uuid
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserDocument:
    """User uploaded document metadata"""
    doc_id: str
    user_id: str
    session_id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    upload_time: str
    status: str = "active"  # active, processing, deleted
    
    def to_dict(self) -> Dict:
        return asdict(self)


class UserDocumentStore:
    """
    Private document store for user uploads
    
    Collection: user_docs_private
    - Isolated by user_id/session_id
    - Supports add/delete operations
    - Auto-cleanup after session timeout
    """
    
    COLLECTION_NAME = "user_docs_private"
    SESSION_TIMEOUT_HOURS = 24
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_model_name: str = "huyydangg/DEk21_hcmute_embedding",
        qdrant_cloud_url: str = "",
        qdrant_api_key: str = "",
        use_cloud: bool = False
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.embedding_model_name = embedding_model_name
        self.qdrant_cloud_url = qdrant_cloud_url
        self.qdrant_api_key = qdrant_api_key
        self.use_cloud = use_cloud
        
        self._client = None
        self._embedding_model = None
        self._bm25_encoder = None
        
        # In-memory document registry
        self._documents: Dict[str, UserDocument] = {}
        
    def _get_client(self):
        """Get Qdrant client (lazy loading)"""
        if self._client is None:
            from qdrant_client import QdrantClient
            
            if self.use_cloud and self.qdrant_cloud_url:
                self._client = QdrantClient(
                    url=self.qdrant_cloud_url,
                    api_key=self.qdrant_api_key
                )
                logger.info(f"[USER_STORE] Connected to Qdrant Cloud")
            else:
                self._client = QdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port
                )
                logger.info(f"[USER_STORE] Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
        return self._client
    
    def _get_embedding_model(self):
        """Get embedding model (lazy loading)"""
        if self._embedding_model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=device
            )
            
            if device == "cuda":
                self._embedding_model.half()
            
            logger.info(f"[USER_STORE] Loaded embedding model on {device}")
        
        return self._embedding_model
    
    def _get_bm25_encoder(self):
        """Get BM25 encoder (lazy loading)"""
        if self._bm25_encoder is None:
            from smart_chunker import SmartRecursiveChunker
            from pathlib import Path
            import json
            
            # Try to load from cache
            cache_path = Path(__file__).parent.parent.parent / "data" / "qdrant_db" / "bm25_encoder.json"
            
            if cache_path.exists():
                # Use existing BM25 vocab from legal collection
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                class SimpleBM25:
                    def __init__(self, data):
                        self.vocab = data["vocab"]
                        self.idf = data["idf"]
                        self.avg_doc_len = data["avg_doc_len"]
                        self.k1 = data.get("k1", 1.5)
                        self.b = data.get("b", 0.75)
                        self._tokenizer = None
                    
                    def _load_tokenizer(self):
                        if self._tokenizer is None:
                            try:
                                from pyvi import ViTokenizer
                                self._tokenizer = ViTokenizer.tokenize
                            except:
                                self._tokenizer = lambda x: x
                    
                    def tokenize(self, text):
                        self._load_tokenizer()
                        text = self._tokenizer(text.lower())
                        tokens = []
                        for word in text.split():
                            clean = ''.join(c for c in word if c.isalnum() or c == '_')
                            if clean:
                                tokens.append(clean)
                        return tokens
                    
                    def encode(self, text):
                        tokens = self.tokenize(text)
                        doc_len = len(tokens)
                        
                        if doc_len == 0:
                            return [], []
                        
                        tf = {}
                        for token in tokens:
                            tf[token] = tf.get(token, 0) + 1
                        
                        indices = []
                        values = []
                        
                        for token, freq in tf.items():
                            if token in self.vocab:
                                idx = self.vocab[token]
                                idf = self.idf.get(token, 0)
                                numerator = freq * (self.k1 + 1)
                                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                                score = idf * (numerator / denominator)
                                if score > 0:
                                    indices.append(idx)
                                    values.append(float(score))
                        
                        return indices, values
                
                self._bm25_encoder = SimpleBM25(data)
                logger.info(f"[USER_STORE] Loaded BM25 with {len(data['vocab'])} vocab")
            else:
                logger.warning("[USER_STORE] BM25 cache not found, sparse search disabled")
                self._bm25_encoder = None
        
        return self._bm25_encoder
    
    def ensure_collection(self):
        """Ensure user_docs_private collection exists"""
        from qdrant_client.http import models
        
        client = self._get_client()
        
        needs_create = False
        try:
            info = client.get_collection(self.COLLECTION_NAME)
            logger.info(f"[USER_STORE] Collection exists with {info.points_count} points")
            
            # Check if sparse vector config is present
            has_sparse = (
                info.config.params.sparse_vectors is not None
                and "sparse" in info.config.params.sparse_vectors
            )
            if not has_sparse:
                logger.warning(
                    "[USER_STORE] Collection missing 'sparse' vector — recreating collection. "
                    "Existing documents will need to be re-uploaded."
                )
                client.delete_collection(self.COLLECTION_NAME)
                needs_create = True
        except Exception:
            needs_create = True
        
        if needs_create:
            # Create collection with dense + sparse vectors
            client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )
            logger.info(f"[USER_STORE] Created collection: {self.COLLECTION_NAME}")
        
        # Always ensure payload indexes exist (idempotent - safe to run on existing collections)
        for field in ("user_id", "session_id", "filename", "doc_id"):
            try:
                client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info(f"[USER_STORE] Ensured payload index: {field}")
            except Exception as idx_err:
                # Index may already exist — not an error
                logger.debug(f"[USER_STORE] Payload index '{field}' already exists or skipped: {idx_err}")
        
        return True
    
    async def upload_file(
        self,
        file_path: str,
        user_id: str,
        session_id: str,
        filename: Optional[str] = None
    ) -> Tuple[bool, str, int]:
        """
        Upload and process user file
        
        Args:
            file_path: Path to uploaded file
            user_id: User identifier
            session_id: Session identifier
            filename: Original filename
            
        Returns:
            Tuple of (success, doc_id, chunk_count)
        """
        from smart_chunker import SmartRecursiveChunker
        
        path = Path(file_path)
        if filename is None:
            filename = path.name
        
        file_size = path.stat().st_size
        file_type = path.suffix.lower().replace('.', '')
        
        # Generate document ID
        doc_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}"
        
        # Create document metadata
        doc = UserDocument(
            doc_id=doc_id,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            chunk_count=0,
            upload_time=datetime.now().isoformat()
        )
        
        try:
            # Chunk file
            chunker = SmartRecursiveChunker(
                chunk_size=512,
                overlap_ratio=0.12,
                inject_context=True
            )
            
            chunks = chunker.chunk_file(file_path)
            
            if not chunks:
                return False, doc_id, 0
            
            doc.chunk_count = len(chunks)
            
            # Upload chunks to Qdrant
            await self._upload_chunks(
                chunks=chunks,
                doc_id=doc_id,
                user_id=user_id,
                session_id=session_id,
                filename=filename
            )
            
            # Register document
            self._documents[doc_id] = doc
            
            logger.info(f"[USER_STORE] Uploaded {filename}: {len(chunks)} chunks")
            return True, doc_id, len(chunks)
            
        except Exception as e:
            logger.error(f"[USER_STORE] Upload failed: {e}")
            return False, doc_id, 0
    
    async def _upload_chunks(
        self,
        chunks: List,
        doc_id: str,
        user_id: str,
        session_id: str,
        filename: str
    ):
        """Upload chunks to Qdrant"""
        from qdrant_client.http import models
        
        client = self._get_client()
        embed_model = self._get_embedding_model()
        bm25 = self._get_bm25_encoder()
        
        points = []
        
        for i, chunk in enumerate(chunks):
            point_id = uuid.uuid4().hex
            
            # Generate embeddings
            content = chunk.content_with_header if hasattr(chunk, 'content_with_header') else chunk.content
            dense_vector = embed_model.encode(content).tolist()
            
            # Generate sparse vector
            sparse_indices, sparse_values = [], []
            if bm25:
                sparse_indices, sparse_values = bm25.encode(content)
            
            # Create payload
            payload = {
                "content": content,
                "doc_id": doc_id,
                "user_id": user_id,
                "session_id": session_id,
                "filename": filename,
                "chunk_index": chunk.chunk_index if hasattr(chunk, 'chunk_index') else i,
                "source_type": "user_upload",
                "upload_time": datetime.now().isoformat()
            }
            
            if hasattr(chunk, 'page_number') and chunk.page_number:
                payload["page"] = chunk.page_number
            
            # Create point
            vectors = {"dense": dense_vector}
            if sparse_indices:
                vectors["sparse"] = models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            
            point = models.PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 32
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=batch
            )
        
        logger.info(f"[USER_STORE] Uploaded {len(points)} chunks for {doc_id}")
    
    def search(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Search user documents (with mandatory user_id filter)
        
        Args:
            query: Search query
            user_id: User identifier (MANDATORY)
            session_id: Optional session filter
            top_k: Number of results
            use_hybrid: Use hybrid search (dense + sparse)
            
        Returns:
            List of matching documents
        """
        from qdrant_client.http import models
        
        client = self._get_client()
        embed_model = self._get_embedding_model()
        bm25 = self._get_bm25_encoder()
        
        # Generate query vectors
        query_vector = embed_model.encode(query).tolist()
        
        # Build filter (MANDATORY user_id)
        filter_conditions = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id)
            )
        ]
        
        if session_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id)
                )
            )
        
        query_filter = models.Filter(must=filter_conditions)
        
        # Hybrid search
        if use_hybrid and bm25:
            sparse_indices, sparse_values = bm25.encode(query)
            try:
                results = client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    prefetch=[
                        models.Prefetch(
                            query=query_vector,
                            using="dense",
                            limit=top_k * 2,
                            filter=query_filter
                        ),
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            ),
                            using="sparse",
                            limit=top_k * 2,
                            filter=query_filter
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True
                )
            except Exception as hybrid_err:
                logger.warning(
                    f"[USER_STORE] Hybrid search failed ({hybrid_err}), falling back to dense-only"
                )
                results = client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=query_vector,
                    using="dense",
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                )
        else:
            # Dense-only search
            results = client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector,
                using="dense",
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
        
        # Handle different result types
        points = results.points if hasattr(results, 'points') else results
        
        # Log search results for debugging
        logger.info(f"[USER_STORE] Search: user={user_id}, session={session_id}, query_len={len(query)}, found={len(points)} results")
        if len(points) > 0:
            logger.debug(f"[USER_STORE] Top result: score={points[0].score:.4f}, filename={points[0].payload.get('filename', 'N/A')}")
        else:
            logger.warning(f"[USER_STORE] No results found for query: '{query[:50]}...'")
        
        return [
            {
                "content": point.payload.get("content", ""),
                "score": point.score,
                "filename": point.payload.get("filename", ""),
                "doc_id": point.payload.get("doc_id", ""),
                "page": point.payload.get("page"),
                "source_type": "user_upload",
                "payload": point.payload
            }
            for point in points
        ]
    
    def get_user_documents(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> List[UserDocument]:
        """Get list of user's uploaded documents"""
        docs = []
        
        for doc in self._documents.values():
            if doc.user_id == user_id:
                if session_id is None or doc.session_id == session_id:
                    if doc.status == "active":
                        docs.append(doc)
        
        return docs
    
    def delete_document(
        self,
        doc_id: str,
        user_id: str
    ) -> bool:
        """Delete user document and its chunks"""
        from qdrant_client.http import models
        
        # Verify ownership
        if doc_id in self._documents:
            doc = self._documents[doc_id]
            if doc.user_id != user_id:
                logger.warning(f"[USER_STORE] Unauthorized delete attempt: {doc_id}")
                return False
        
        client = self._get_client()
        
        # Delete from Qdrant
        client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id)
                        ),
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                )
            )
        )
        
        # Remove from registry
        if doc_id in self._documents:
            self._documents[doc_id].status = "deleted"
            del self._documents[doc_id]
        
        logger.info(f"[USER_STORE] Deleted document: {doc_id}")
        return True
    
    def cleanup_session(self, session_id: str) -> int:
        """Cleanup all documents from a session"""
        from qdrant_client.http import models
        
        client = self._get_client()
        
        # Delete from Qdrant
        client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id)
                        )
                    ]
                )
            )
        )
        
        # Remove from registry
        deleted_count = 0
        docs_to_delete = [
            doc_id for doc_id, doc in self._documents.items()
            if doc.session_id == session_id
        ]
        
        for doc_id in docs_to_delete:
            del self._documents[doc_id]
            deleted_count += 1
        
        logger.info(f"[USER_STORE] Cleaned up session {session_id}: {deleted_count} documents")
        return deleted_count


# Singleton instance
_user_store = None

def get_user_store() -> UserDocumentStore:
    """Get singleton UserDocumentStore instance"""
    global _user_store
    
    if _user_store is None:
        use_cloud = os.getenv("QDRANT_USE_CLOUD", "false").lower() == "true"
        
        _user_store = UserDocumentStore(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_cloud_url=os.getenv("QDRANT_CLOUD_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            use_cloud=use_cloud
        )
        _user_store.ensure_collection()
    
    return _user_store


if __name__ == "__main__":
    # Test
    store = get_user_store()
    print(f"User Store initialized: {store.COLLECTION_NAME}")
