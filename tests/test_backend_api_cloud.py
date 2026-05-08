"""
Integration Tests for Backend API with Qdrant Cloud

Kiểm tra:
1. API status endpoint với Cloud
2. Search endpoints (legal, hybrid)
3. Chat endpoint
4. Embedding model loads correctly
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


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def backend_config():
    """Load and verify backend Config."""
    from main import Config
    return Config


@pytest.fixture(scope="module")
def qdrant_client():
    """Get Qdrant client from backend."""
    # Reset singleton
    import main
    main._qdrant_client = None
    return main.get_qdrant_client()


@pytest.fixture(scope="module")
def embed_model():
    """Get embedding model from shared embeddings module."""
    from src.embeddings import get_model

    get_model.cache_clear()
    return get_model()


@pytest.fixture(scope="module")
def bm25():
    """Get BM25 encoder from backend."""
    import main
    main._bm25_encoder = None
    return main.get_bm25_encoder()


# ============================================================================
# 1. Config Tests
# ============================================================================

class TestBackendConfig:
    """Test backend configuration for Qdrant Cloud."""
    
    def test_config_uses_cloud(self, backend_config):
        """Config QDRANT_USE_CLOUD phải là True."""
        assert backend_config.QDRANT_USE_CLOUD is True, \
            f"QDRANT_USE_CLOUD={backend_config.QDRANT_USE_CLOUD}"
        print(f"✅ QDRANT_USE_CLOUD = True")
    
    def test_config_cloud_url_set(self, backend_config):
        """Config QDRANT_CLOUD_URL phải không rỗng."""
        assert backend_config.QDRANT_CLOUD_URL, "QDRANT_CLOUD_URL is empty"
        assert "qdrant.io" in backend_config.QDRANT_CLOUD_URL
        print(f"✅ Cloud URL: {backend_config.QDRANT_CLOUD_URL[:50]}...")
    
    def test_config_api_key_set(self, backend_config):
        """Config QDRANT_API_KEY phải không rỗng."""
        assert backend_config.QDRANT_API_KEY, "QDRANT_API_KEY is empty"
        assert len(backend_config.QDRANT_API_KEY) > 10
        print(f"✅ API Key set (len={len(backend_config.QDRANT_API_KEY)})")
    
    def test_config_collection_name(self, backend_config):
        """Collection name phải là legal_rag_vn (Cloud)."""
        assert backend_config.LEGAL_COLLECTION == "legal_rag_vn", \
            f"Expected 'legal_rag_vn', got '{backend_config.LEGAL_COLLECTION}'"
        print(f"✅ Legal collection: {backend_config.LEGAL_COLLECTION}")


# ============================================================================
# 2. Qdrant Client Tests
# ============================================================================

class TestQdrantClient:
    """Test Qdrant client connects to Cloud."""
    
    def test_client_connects(self, qdrant_client):
        """Client phải kết nối được với Cloud."""
        assert qdrant_client is not None
        collections = qdrant_client.get_collections()
        assert len(collections.collections) > 0
        print(f"✅ Connected to Cloud ({len(collections.collections)} collections)")
    
    def test_legal_collection_accessible(self, qdrant_client, backend_config):
        """Phải truy cập được collection legal_rag_vn."""
        info = qdrant_client.get_collection(backend_config.LEGAL_COLLECTION)
        assert info.points_count >= 100000
        assert str(info.status).lower() == "green"
        print(f"✅ {backend_config.LEGAL_COLLECTION}: {info.points_count} points, status={info.status}")
    
    def test_user_collection_accessible(self, qdrant_client, backend_config):
        """Phải truy cập được collection user_docs_private."""
        info = qdrant_client.get_collection(backend_config.USER_COLLECTION)
        assert str(info.status).lower() == "green"
        print(f"✅ {backend_config.USER_COLLECTION}: {info.points_count} points, status={info.status}")


# ============================================================================
# 3. Embedding Model Tests
# ============================================================================

class TestEmbeddingModel:
    """Test embedding model for Cloud search."""
    
    def test_model_loads(self, embed_model):
        """Embedding model phải load thành công."""
        assert embed_model is not None
        print(f"✅ Embedding model loaded: {embed_model}")
    
    def test_embed_single_query(self, embed_model):
        """Phải embed được 1 query thành vector 768D."""
        query = "tội tham ô tài sản"
        vector = embed_model.encode(query)
        
        assert vector is not None
        assert len(vector) == 768
        assert not all(v == 0 for v in vector)
        print(f"✅ Embedded query to {len(vector)}D vector")
    
    def test_embed_batch(self, embed_model):
        """Phải embed được batch texts."""
        texts = [
            "Điều 353 quy định về tội tham ô tài sản",
            "Hình phạt tù từ 2 đến 7 năm",
            "Bộ luật Hình sự 2015"
        ]
        vectors = embed_model.encode(texts)
        
        assert len(vectors) == 3
        assert all(len(v) == 768 for v in vectors)
        print(f"✅ Embedded {len(vectors)} texts")
    
    def test_embed_similarity(self, embed_model):
        """Vectors phải phản ánh similarity ngữ nghĩa."""
        similar_texts = [
            "quy định về quyền sở hữu nhà ở theo luật dân sự",
            "luật dân sự quy định quyền sở hữu nhà ở"
        ]
        different_text = "công nghệ trí tuệ nhân tạo trong y học hiện đại"
        
        vecs = embed_model.encode(similar_texts + [different_text])
        
        # Cosine similarity
        from numpy.linalg import norm
        sim_similar = np.dot(vecs[0], vecs[1]) / (norm(vecs[0]) * norm(vecs[1]))
        sim_different = np.dot(vecs[0], vecs[2]) / (norm(vecs[0]) * norm(vecs[2]))
        
        assert sim_similar > sim_different, \
            f"Similar ({sim_similar:.4f}) should > Different ({sim_different:.4f})"
        print(f"[PASS] Similar: {sim_similar:.4f}, Different: {sim_different:.4f}")


# ============================================================================
# 4. BM25 Encoder Tests
# ============================================================================

class TestBM25Encoder:
    """Test BM25 sparse encoder."""
    
    def test_bm25_loaded(self, bm25):
        """BM25 encoder phải load từ cache."""
        assert bm25 is not None
        assert hasattr(bm25, 'encode')
        assert hasattr(bm25, 'vocab')
        assert len(bm25.vocab) > 0
        print(f"✅ BM25 loaded: {len(bm25.vocab)} vocab")
    
    def test_bm25_encode_vietnamese(self, bm25):
        """BM25 phải encode được tiếng Việt."""
        query = "Tội phạm tham nhũng hình phạt"
        indices, values = bm25.encode(query)
        
        assert len(indices) > 0, "No indices generated"
        assert len(indices) == len(values)
        print(f"✅ BM25: {len(indices)} terms, values={[f'{v:.3f}' for v in values[:3]]}")
    
    def test_bm25_tokenizer_works(self, bm25):
        """BM25 tokenizer phải xử lý tiếng Việt đúng."""
        text = "Bộ luật hình sự năm 2015"
        tokens = bm25.tokenize(text)
        
        assert len(tokens) > 0
        print(f"✅ Tokens: {tokens}")


# ============================================================================
# 5. End-to-End Search Tests
# ============================================================================

class TestEndToEndSearch:
    """Test complete search pipeline through backend functions."""
    
    def test_legal_search_function(self):
        """search_legal_collection() phải hoạt động với Cloud."""
        import main
        # Reset singletons  
        main._qdrant_client = None
        main._bm25_encoder = None
        
        results = main.search_legal_collection(
            query="tội tham ô tài sản",
            top_k=5
        )
        
        assert len(results) > 0, "No results from search_legal_collection"
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)
        assert all(r["source_type"] == "legal" for r in results)
        
        print(f"✅ search_legal_collection: {len(results)} results")
        for i, r in enumerate(results):
            print(f"   [{i+1}] score={r['score']:.4f} | {r.get('dieu', '')} - {r.get('so_hieu', '')}")
    
    def test_hybrid_search_function(self):
        """hybrid_search() phải hoạt động với Cloud."""
        import main
        
        results, rerank_time, reranker_used = main.hybrid_search(
            query="quy định về hợp đồng lao động",
            user_id="test_user",
            search_mode="legal",
            top_k=5
        )
        
        assert len(results) > 0, "No results from hybrid_search"
        print(f"✅ hybrid_search: {len(results)} results, rerank={reranker_used}")
    
    def test_search_returns_all_metadata(self):
        """Search phải trả về đủ metadata: content, score, so_hieu, dieu, source_type."""
        import main
        
        results = main.search_legal_collection(
            query="bảo vệ môi trường",
            top_k=3
        )
        
        required_fields = ["content", "score", "so_hieu", "dieu", "source_type"]
        for r in results:
            for field in required_fields:
                assert field in r, f"Missing '{field}' in result"
        
        print(f"✅ All {len(required_fields)} fields present in {len(results)} results")
    
    def test_search_with_different_queries(self):
        """Search phải hoạt động với nhiều loại câu hỏi."""
        import main
        
        queries = [
            ("tội giết người", "hình sự"),
            ("hợp đồng thuê nhà", "dân sự"),
            ("thành lập doanh nghiệp", "thương mại"),
        ]
        
        for query, category in queries:
            results = main.search_legal_collection(query=query, top_k=3)
            assert len(results) > 0, f"No results for '{query}'"
            print(f"✅ '{query}' ({category}): {len(results)} results")


# ============================================================================
# 6. Prompt & LLM Integration Tests
# ============================================================================

class TestPromptBuilding:
    """Test prompt building for LLM."""
    
    def test_build_legal_prompt(self):
        """build_legal_prompt() phải tạo prompt đúng format."""
        import main
        
        contexts = [
            {
                "content": "Điều 353. Tội tham ô tài sản...",
                "so_hieu": "100/2015/QH13",
                "dieu": "Điều 353",
                "source_type": "legal"
            }
        ]
        
        prompt = main.build_legal_prompt(
            query="Tội tham ô bị phạt bao nhiêu năm?",
            contexts=contexts
        )
        
        assert "LUẬT" in prompt
        assert "Điều 353" in prompt
        assert "Tội tham ô" in prompt
        assert len(prompt) > 100
        print(f"✅ Prompt built ({len(prompt)} chars)")


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])


