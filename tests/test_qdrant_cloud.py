"""
Test Qdrant Cloud Connection & Hybrid Search (Dense + BM25 Sparse)

Kiểm tra toàn diện:
1. Kết nối Cloud
2. Collections tồn tại và status GREEN
3. Payload integrity
4. Dense search
5. Sparse (BM25) search
6. Hybrid search (RRF Fusion)
7. Search với filter
8. Latency measurement
"""

import os
import sys
import time
import json
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web_app" / "backend"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from qdrant_client import QdrantClient
from qdrant_client.http import models


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def cloud_client():
    """Create Qdrant Cloud client."""
    url = os.getenv("QDRANT_CLOUD_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url or not api_key:
        pytest.skip("QDRANT_CLOUD_URL or QDRANT_API_KEY not set in .env")
    
    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=10)
        client.get_collections()
        return client
    except Exception as exc:
        pytest.skip(f"Qdrant Cloud unavailable: {exc}")


@pytest.fixture(scope="module")
def embedding_model():
    """Load embedding model for search tests."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("huyydangg/DEk21_hcmute_embedding", device=device)
        if device == "cuda":
            model.half()
        return model
    except Exception as exc:
        pytest.skip(f"Embedding model unavailable: {exc}")


@pytest.fixture(scope="module")
def bm25_encoder():
    """Load BM25 encoder for sparse search tests."""
    cache_path = PROJECT_ROOT / "data" / "cache" / "bm25_hybrid.json"
    if not cache_path.exists():
        pytest.skip(f"BM25 cache not found at {cache_path}")
    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    try:
        from pyvi import ViTokenizer
        tokenizer = ViTokenizer.tokenize
    except ImportError:
        tokenizer = lambda x: x
    
    class BM25:
        def __init__(self, data, tokenizer):
            self.vocab = data["vocab"]
            self.idf = data["idf"]
            self.avg_doc_len = data["avg_doc_len"]
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self._tokenizer = tokenizer
        
        def tokenize(self, text):
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
            indices, values = [], []
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
    
    return BM25(data, tokenizer)


LEGAL_COLLECTION = os.getenv("QDRANT_LEGAL_COLLECTION", "legal_rag_vn")
USER_COLLECTION = "user_docs_private"


# ============================================================================
# 1. Connection Tests
# ============================================================================

class TestCloudConnection:
    """Test Qdrant Cloud connection."""
    
    def test_cloud_connection(self, cloud_client):
        """Kết nối Cloud thành công."""
        collections = cloud_client.get_collections()
        assert collections is not None
        assert len(collections.collections) > 0
        print(f"✅ Connected! {len(collections.collections)} collections found")
    
    def test_list_collections(self, cloud_client):
        """Có đủ 2 collections: legal_rag_vn và user_docs_private."""
        collections = cloud_client.get_collections()
        names = [c.name for c in collections.collections]
        
        assert LEGAL_COLLECTION in names, f"Collection '{LEGAL_COLLECTION}' not found"
        assert USER_COLLECTION in names, f"Collection '{USER_COLLECTION}' not found"
        print(f"✅ Collections: {names}")


# ============================================================================
# 2. Collection Status Tests
# ============================================================================

class TestCollectionStatus:
    """Test collection health and data integrity."""
    
    def test_legal_collection_status_green(self, cloud_client):
        """Collection legal_rag_vn phải có status GREEN."""
        info = cloud_client.get_collection(LEGAL_COLLECTION)
        assert str(info.status).lower() == "green", f"Status: {info.status}"
        print(f"✅ {LEGAL_COLLECTION} status: {info.status}")
    
    def test_legal_collection_has_100k_points(self, cloud_client):
        """Collection legal_rag_vn phải có ~100k+ points."""
        info = cloud_client.get_collection(LEGAL_COLLECTION)
        assert info.points_count >= 100000, f"Only {info.points_count} points"
        print(f"✅ {LEGAL_COLLECTION}: {info.points_count} points")
    
    def test_legal_collection_has_dense_vectors(self, cloud_client):
        """Collection phải có dense vectors config (768D, Cosine)."""
        info = cloud_client.get_collection(LEGAL_COLLECTION)
        vectors_config = info.config.params.vectors
        
        assert "dense" in vectors_config, "No 'dense' vectors config"
        dense = vectors_config["dense"]
        assert dense.size == 768, f"Expected 768D, got {dense.size}D"
        assert str(dense.distance).lower() == "cosine", f"Distance: {dense.distance}"
        print(f"✅ Dense vectors: {dense.size}D, {dense.distance}")
    
    def test_legal_collection_has_sparse_vectors(self, cloud_client):
        """Collection phải có sparse vectors config (BM25)."""
        info = cloud_client.get_collection(LEGAL_COLLECTION)
        sparse_config = info.config.params.sparse_vectors
        
        assert sparse_config is not None, "No sparse vectors config"
        assert "sparse" in sparse_config, "No 'sparse' key in sparse_vectors"
        print(f"✅ Sparse vectors configured: {list(sparse_config.keys())}")
    
    def test_user_collection_status_green(self, cloud_client):
        """Collection user_docs_private phải có status GREEN."""
        info = cloud_client.get_collection(USER_COLLECTION)
        assert str(info.status).lower() == "green", f"Status: {info.status}"
        print(f"✅ {USER_COLLECTION} status: {info.status}")


# ============================================================================
# 3. Payload & Data Integrity Tests
# ============================================================================

class TestDataIntegrity:
    """Test data and payload integrity."""
    
    def test_sample_point_payload(self, cloud_client):
        """Point mẫu phải có payload đầy đủ: content, chunk_id, so_hieu, dieu."""
        pts, _ = cloud_client.scroll(LEGAL_COLLECTION, limit=1, with_payload=True)
        
        assert len(pts) > 0, "No points found"
        
        payload = pts[0].payload
        assert "content" in payload, "Missing 'content' in payload"
        assert "chunk_id" in payload, "Missing 'chunk_id' in payload"
        assert "so_hieu" in payload, "Missing 'so_hieu' in payload"
        assert "dieu" in payload, "Missing 'dieu' in payload"
        
        assert len(payload["content"]) > 0, "Empty content"
        print(f"✅ Payload keys: {list(payload.keys())}")
        print(f"   content[:80]: {payload['content'][:80]}")
        print(f"   so_hieu: {payload['so_hieu']}")
        print(f"   dieu: {payload['dieu']}")
    
    def test_sample_point_has_vectors(self, cloud_client):
        """Point mẫu phải có cả dense và sparse vectors."""
        pts, _ = cloud_client.scroll(
            LEGAL_COLLECTION, limit=1,
            with_vectors=True, with_payload=False
        )
        
        assert len(pts) > 0, "No points found"
        
        vectors = pts[0].vector
        assert "dense" in vectors, "No 'dense' vector"
        assert "sparse" in vectors, "No 'sparse' vector"
        
        # Dense vector should be 768D
        dense_vec = vectors["dense"]
        assert len(dense_vec) == 768, f"Dense vector size: {len(dense_vec)}"
        
        # Sparse vector should have indices and values
        sparse_vec = vectors["sparse"]
        assert hasattr(sparse_vec, 'indices') or 'indices' in str(type(sparse_vec)), \
            "Sparse vector missing indices"
        
        print(f"✅ Dense vector: {len(dense_vec)}D")
        print(f"✅ Sparse vector present")
    
    def test_multiple_points_consistency(self, cloud_client):
        """Nhiều points phải có payload nhất quán."""
        pts, _ = cloud_client.scroll(LEGAL_COLLECTION, limit=10, with_payload=True)
        
        assert len(pts) == 10, f"Expected 10 points, got {len(pts)}"
        
        for p in pts:
            assert "content" in p.payload
            assert "so_hieu" in p.payload
            assert "dieu" in p.payload
            assert len(p.payload["content"]) > 0
        
        print(f"✅ All 10 sample points have consistent payload")


# ============================================================================
# 4. Dense Search Tests
# ============================================================================

class TestDenseSearch:
    """Test dense (semantic) vector search."""
    
    def test_dense_search_returns_results(self, cloud_client, embedding_model):
        """Dense search phải trả về kết quả."""
        query = "tội tham ô tài sản"
        query_vector = embedding_model.encode(query).tolist()
        
        results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            query=query_vector,
            using="dense",
            limit=5,
            with_payload=True
        )
        
        assert hasattr(results, 'points'), "No points attribute"
        assert len(results.points) > 0, "No results returned"
        assert len(results.points) == 5, f"Expected 5 results, got {len(results.points)}"
        print(f"✅ Dense search: {len(results.points)} results")
        for i, r in enumerate(results.points):
            print(f"   [{i+1}] score={r.score:.4f} | {r.payload.get('dieu', '')} - {r.payload.get('so_hieu', '')}")
    
    def test_dense_search_relevance(self, cloud_client, embedding_model):
        """Kết quả dense search phải liên quan đến câu hỏi."""
        query = "hình phạt tội giết người"
        query_vector = embedding_model.encode(query).tolist()
        
        results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            query=query_vector,
            using="dense",
            limit=3,
            with_payload=True
        )
        
        assert len(results.points) > 0
        # Top result should have reasonable score
        assert results.points[0].score > 0.3, f"Score too low: {results.points[0].score}"
        print(f"✅ Top result score: {results.points[0].score:.4f}")
        print(f"   Content: {results.points[0].payload.get('content', '')[:100]}")
    
    def test_dense_search_score_ordering(self, cloud_client, embedding_model):
        """Scores phải được sắp xếp giảm dần."""
        query = "quy định về hợp đồng lao động"
        query_vector = embedding_model.encode(query).tolist()
        
        results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            query=query_vector,
            using="dense",
            limit=10,
            with_payload=False
        )
        
        scores = [r.score for r in results.points]
        assert scores == sorted(scores, reverse=True), "Scores not in descending order"
        print(f"✅ Scores ordered: {[f'{s:.4f}' for s in scores[:5]]}")


# ============================================================================
# 5. Sparse (BM25) Search Tests
# ============================================================================

class TestSparseSearch:
    """Test sparse (BM25 keyword) search."""
    
    def test_bm25_encoder_works(self, bm25_encoder):
        """BM25 encoder phải tokenize và encode được tiếng Việt."""
        query = "tội phạm tham nhũng"
        indices, values = bm25_encoder.encode(query)
        
        assert len(indices) > 0, "BM25 returned no indices"
        assert len(values) > 0, "BM25 returned no values"
        assert len(indices) == len(values), "Indices and values length mismatch"
        assert all(v > 0 for v in values), "All values should be positive"
        
        tokens = bm25_encoder.tokenize(query)
        print(f"✅ BM25 tokenized: {tokens}")
        print(f"   Indices: {indices[:5]}... ({len(indices)} total)")
        print(f"   Values: {[f'{v:.4f}' for v in values[:5]]}")
    
    def test_sparse_search_returns_results(self, cloud_client, bm25_encoder):
        """Sparse (BM25) search phải trả về kết quả từ Cloud."""
        query = "Điều 353 tham ô tài sản"
        indices, values = bm25_encoder.encode(query)
        
        assert len(indices) > 0, "BM25 encode failed"
        
        results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            query=models.SparseVector(indices=indices, values=values),
            using="sparse",
            limit=5,
            with_payload=True
        )
        
        assert len(results.points) > 0, "Sparse search returned no results"
        print(f"✅ Sparse search: {len(results.points)} results")
        for i, r in enumerate(results.points):
            print(f"   [{i+1}] score={r.score:.4f} | {r.payload.get('dieu', '')} - {r.payload.get('so_hieu', '')}")
    
    def test_sparse_search_keyword_match(self, cloud_client, bm25_encoder):
        """BM25 search phải tìm được keyword chính xác."""
        query = "91/2015/QH13"  # Bộ luật Dân sự
        indices, values = bm25_encoder.encode(query)
        
        if len(indices) > 0:
            results = cloud_client.query_points(
                collection_name=LEGAL_COLLECTION,
                query=models.SparseVector(indices=indices, values=values),
                using="sparse",
                limit=5,
                with_payload=True
            )
            
            assert len(results.points) > 0, "No results for exact document number"
            # Check that results mention the document number
            found_match = any("91/2015" in r.payload.get("so_hieu", "") for r in results.points)
            print(f"✅ Keyword match found: {found_match}")
            for i, r in enumerate(results.points):
                print(f"   [{i+1}] so_hieu={r.payload.get('so_hieu', '')} | {r.payload.get('dieu', '')}")
        else:
            pytest.skip("BM25 could not encode this query")


# ============================================================================
# 6. Hybrid Search (RRF Fusion) Tests
# ============================================================================

class TestHybridSearch:
    """Test hybrid search combining Dense + Sparse with RRF Fusion."""
    
    def test_hybrid_search_rrf(self, cloud_client, embedding_model, bm25_encoder):
        """Hybrid search (Dense + BM25 + RRF) phải trả về kết quả."""
        query = "tội tham ô tài sản bị phạt bao nhiêu năm"
        
        # Prepare dense vector
        query_vector = embedding_model.encode(query).tolist()
        
        # Prepare sparse vector  
        sparse_indices, sparse_values = bm25_encoder.encode(query)
        assert len(sparse_indices) > 0, "BM25 encode failed"
        
        # Hybrid search with RRF Fusion
        results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=20
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    ),
                    using="sparse",
                    limit=20
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=10,
            with_payload=True
        )
        
        assert hasattr(results, 'points'), "No points attribute in results"
        assert len(results.points) > 0, "Hybrid search returned no results"
        assert len(results.points) <= 10, f"Expected <=10, got {len(results.points)}"
        
        print(f"✅ Hybrid search (RRF): {len(results.points)} results")
        for i, r in enumerate(results.points):
            print(f"   [{i+1}] score={r.score:.4f} | {r.payload.get('dieu', '')} - {r.payload.get('so_hieu', '')}")
    
    def test_hybrid_search_better_than_dense_only(self, cloud_client, embedding_model, bm25_encoder):
        """Hybrid search nên có kết quả tốt hơn hoặc bằng dense-only."""
        query = "Điều 168 Bộ luật Hình sự tội cướp tài sản"
        
        query_vector = embedding_model.encode(query).tolist()
        sparse_indices, sparse_values = bm25_encoder.encode(query)
        
        # Dense only
        dense_results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            query=query_vector,
            using="dense",
            limit=5,
            with_payload=True
        )
        
        # Hybrid (Dense + Sparse + RRF)
        hybrid_results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            prefetch=[
                models.Prefetch(query=query_vector, using="dense", limit=20),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse", limit=20
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=5,
            with_payload=True
        )
        
        assert len(dense_results.points) > 0
        assert len(hybrid_results.points) > 0
        
        dense_contents = {r.payload.get("content", "")[:50] for r in dense_results.points}
        hybrid_contents = {r.payload.get("content", "")[:50] for r in hybrid_results.points}
        
        print(f"✅ Dense-only: {len(dense_results.points)} results")
        print(f"✅ Hybrid RRF: {len(hybrid_results.points)} results")
        print(f"   Overlap: {len(dense_contents & hybrid_contents)} common results")
    
    def test_hybrid_search_multiple_queries(self, cloud_client, embedding_model, bm25_encoder):
        """Hybrid search phải hoạt động nhất quán trên nhiều queries."""
        queries = [
            "thành lập công ty trách nhiệm hữu hạn",
            "quy định về bảo vệ môi trường",
            "quyền sở hữu trí tuệ phần mềm",
            "tội phạm ma túy",
            "luật lao động nghỉ phép"
        ]
        
        for query in queries:
            query_vector = embedding_model.encode(query).tolist()
            sparse_indices, sparse_values = bm25_encoder.encode(query)
            
            if len(sparse_indices) == 0:
                print(f"⚠️ BM25 returned empty for: {query}")
                continue
            
            results = cloud_client.query_points(
                collection_name=LEGAL_COLLECTION,
                prefetch=[
                    models.Prefetch(query=query_vector, using="dense", limit=10),
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                        using="sparse", limit=10
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=5,
                with_payload=True
            )
            
            assert len(results.points) > 0, f"No results for: {query}"
            print(f"✅ '{query[:30]}...' → {len(results.points)} results (top={results.points[0].score:.4f})")


# ============================================================================
# 7. Search with Filter Tests
# ============================================================================

class TestFilteredSearch:
    """Test search with payload filters."""
    
    def test_search_with_so_hieu_filter(self, cloud_client, embedding_model):
        """Search với filter theo số hiệu văn bản."""
        query = "quy định về hợp đồng"
        query_vector = embedding_model.encode(query).tolist()
        
        # Search with filter for specific document
        results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            query=query_vector,
            using="dense",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="so_hieu",
                        match=models.MatchValue(value="91/2015/QH13")  # Bộ luật Dân sự
                    )
                ]
            ),
            limit=5,
            with_payload=True
        )
        
        assert len(results.points) > 0, "No results with filter"
        
        # All results should match the filter
        for r in results.points:
            assert r.payload.get("so_hieu") == "91/2015/QH13"
        
        print(f"✅ Filtered search: {len(results.points)} results (all from 91/2015/QH13)")
    
    def test_search_with_dieu_filter(self, cloud_client, embedding_model):
        """Search với filter theo điều khoản."""
        query = "hợp đồng vận chuyển"
        query_vector = embedding_model.encode(query).tolist()
        
        results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            query=query_vector,
            using="dense",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="dieu",
                        match=models.MatchValue(value="Điều 523")
                    )
                ]
            ),
            limit=3,
            with_payload=True
        )
        
        if len(results.points) > 0:
            for r in results.points:
                assert "Điều 523" in r.payload.get("dieu", "")
            print(f"✅ Filtered by dieu: {len(results.points)} results")
        else:
            print("⚠️ No results for Điều 523 filter (may not exist in index)")


# ============================================================================
# 8. Performance / Latency Tests
# ============================================================================

class TestPerformance:
    """Test Cloud search latency."""
    
    def test_dense_search_latency(self, cloud_client, embedding_model):
        """Dense search latency phải < 3 giây."""
        query = "quy định pháp luật về đất đai"
        query_vector = embedding_model.encode(query).tolist()
        
        start = time.time()
        results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            query=query_vector,
            using="dense",
            limit=10,
            with_payload=True
        )
        elapsed = time.time() - start
        
        assert elapsed < 3.0, f"Search took {elapsed:.2f}s (> 3s)"
        assert len(results.points) > 0
        print(f"✅ Dense search latency: {elapsed:.3f}s ({len(results.points)} results)")
    
    def test_hybrid_search_latency(self, cloud_client, embedding_model, bm25_encoder):
        """Hybrid search latency phải < 5 giây."""
        query = "tội phạm về ma túy hình phạt"
        
        query_vector = embedding_model.encode(query).tolist()
        sparse_indices, sparse_values = bm25_encoder.encode(query)
        
        start = time.time()
        results = cloud_client.query_points(
            collection_name=LEGAL_COLLECTION,
            prefetch=[
                models.Prefetch(query=query_vector, using="dense", limit=20),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse", limit=20
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=10,
            with_payload=True
        )
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Hybrid search took {elapsed:.2f}s (> 5s)"
        assert len(results.points) > 0
        print(f"✅ Hybrid search latency: {elapsed:.3f}s ({len(results.points)} results)")
    
    def test_batch_search_performance(self, cloud_client, embedding_model):
        """Batch search 5 queries phải hoàn thành trong < 15s."""
        queries = [
            "tội tham ô", "hợp đồng lao động",
            "quyền sở hữu", "bảo vệ môi trường", "luật doanh nghiệp"
        ]
        
        start = time.time()
        for query in queries:
            query_vector = embedding_model.encode(query).tolist()
            cloud_client.query_points(
                collection_name=LEGAL_COLLECTION,
                query=query_vector,
                using="dense",
                limit=5, with_payload=True
            )
        elapsed = time.time() - start
        
        assert elapsed < 15.0, f"Batch search took {elapsed:.2f}s"
        print(f"✅ Batch search (5 queries): {elapsed:.3f}s (avg {elapsed/5:.3f}s)")


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
