"""
Unit Tests for Core Components: Embedding, Reranker, Qdrant.
Tests the fundamental logic without external dependencies where possible.
Skips if backend.core module is not available (different architecture).
"""
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "web_app"))

# Check if backend.core module exists
try:
    import backend.core
    HAS_BACKEND_CORE = True
except (ImportError, ModuleNotFoundError):
    HAS_BACKEND_CORE = False

skip_no_core = pytest.mark.skipif(
    not HAS_BACKEND_CORE,
    reason="backend.core module not available (project uses flat backend structure)"
)


# ==================== Config Tests ====================

@pytest.mark.unit
class TestConfig:
    """Test configuration loading."""
    
    def test_config_loads(self):
        """Config should load without errors."""
        from backend.config import settings
        
        assert settings is not None
        assert settings.API_PORT == 8080
        assert settings.QDRANT_HOST == "localhost"
        assert settings.QDRANT_PORT == 6333
    
    def test_config_collections(self):
        """Config should have correct collection names."""
        from backend.config import settings
        
        assert settings.QDRANT_USER_COLLECTION == "user_docs_private"


# ==================== Embedding Tests ====================

@skip_no_core
@pytest.mark.unit
class TestEmbedding:
    """Test embedding model functionality."""
    
    def test_embedding_singleton(self):
        """Embedding model should be singleton."""
        from backend.core.embeddings import get_embedding_model
        
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        
        assert model1 is model2
    
    def test_embedding_info(self, embedding_model):
        """Embedding model should return correct info."""
        info = embedding_model.get_info()
        
        assert info["dimension"] == 768
        assert info["model_name"] == "huyydangg/DEk21_hcmute_embedding"
    
    @pytest.mark.slow
    def test_embed_single_query(self, embedding_model, sample_query):
        """Single query embedding should return 768-dim vector."""
        vector = embedding_model.embed_query(sample_query)
        
        assert vector is not None
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (768,)
        assert not np.isnan(vector).any()
    
    @pytest.mark.slow
    def test_embed_batch(self, embedding_model, sample_texts):
        """Batch embedding should return correct shape."""
        vectors = embedding_model.embed(sample_texts)
        
        assert vectors is not None
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (len(sample_texts), 768)
    
    @pytest.mark.slow
    def test_embed_empty_list(self, embedding_model):
        """Empty list should return empty array."""
        vectors = embedding_model.embed([])
        
        assert vectors is not None
        assert len(vectors) == 0
    
    @pytest.mark.slow
    def test_embed_similarity(self, embedding_model):
        """Similar texts should have high cosine similarity."""
        texts = [
            "Điều 353 về tội tham ô tài sản",
            "Quy định về tội tham ô trong bộ luật hình sự",
            "Thời tiết hôm nay rất đẹp"  # Unrelated text
        ]
        
        vectors = embedding_model.embed(texts)
        
        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_related = cosine_sim(vectors[0], vectors[1])
        sim_unrelated = cosine_sim(vectors[0], vectors[2])
        
        # Related texts should be more similar
        assert sim_related > sim_unrelated


# ==================== Qdrant Tests ====================

@skip_no_core
class TestQdrantConnector:
    """Test Qdrant connector functionality."""
    
    @pytest.mark.unit
    def test_qdrant_singleton(self):
        """Qdrant connector should be singleton."""
        from backend.core.qdrant_store import get_qdrant_connector
        
        conn1 = get_qdrant_connector()
        conn2 = get_qdrant_connector()
        
        assert conn1 is conn2
    
    @pytest.mark.unit
    def test_qdrant_config(self, qdrant_connector):
        """Qdrant connector should have correct config."""
        assert qdrant_connector.host == "localhost"
        assert qdrant_connector.port == 6333
        assert qdrant_connector.legal_collection == "legal_rag_hybrid"
        assert qdrant_connector.user_collection == "user_docs_private"
    
    @pytest.mark.integration
    def test_qdrant_connection(self, qdrant_connector):
        """Should connect to Qdrant successfully."""
        connected = qdrant_connector.check_connection()
        
        assert connected is True
    
    @pytest.mark.integration
    def test_qdrant_list_collections(self, qdrant_connector):
        """Should list available collections."""
        collections = qdrant_connector.list_collections()
        
        assert isinstance(collections, list)
        assert len(collections) > 0
        assert "legal_rag_hybrid" in collections
    
    @pytest.mark.integration
    def test_qdrant_collection_info(self, qdrant_connector):
        """Should get collection info."""
        info = qdrant_connector.get_collection_info("legal_rag_hybrid")
        
        assert info is not None
        assert "name" in info
        assert "vectors_count" in info or "points_count" in info


# ==================== Reranker Tests ====================

@skip_no_core
@pytest.mark.unit
class TestReranker:
    """Test reranker functionality."""
    
    def test_reranker_singleton(self):
        """Reranker should be singleton."""
        from backend.core.reranker import get_reranker
        
        r1 = get_reranker()
        r2 = get_reranker()
        
        assert r1 is r2
    
    def test_reranker_info(self, reranker_model):
        """Reranker should return correct info."""
        info = reranker_model.get_info()
        
        assert info["model_name"] == "AITeamVN/Vietnamese_Reranker"
    
    @pytest.mark.slow
    def test_rerank_documents(self, reranker_model, sample_query):
        """Reranker should reorder documents by relevance."""
        documents = [
            {"text": "Thời tiết hôm nay đẹp", "score": 0.5},
            {"text": "Điều 353 quy định về tội tham ô tài sản có thể bị phạt tù", "score": 0.6},
            {"text": "Hình phạt cho tội tham nhũng theo bộ luật hình sự", "score": 0.4}
        ]
        
        reranked = reranker_model.rerank(sample_query, documents)
        
        assert len(reranked) == 3
        assert all("rerank_score" in doc for doc in reranked)
        
        # First result should be most relevant (about corruption)
        assert "tham" in reranked[0]["text"].lower() or "phạt" in reranked[0]["text"].lower()
    
    @pytest.mark.slow
    def test_rerank_empty_list(self, reranker_model, sample_query):
        """Reranker should handle empty list."""
        reranked = reranker_model.rerank(sample_query, [])
        
        assert reranked == []
    
    @pytest.mark.slow
    def test_rerank_scores_range(self, reranker_model, sample_query):
        """Rerank scores should be between 0 and 1."""
        documents = [
            {"text": "Điều 353 về tội tham ô"},
            {"text": "Hình phạt tù cho tội phạm"}
        ]
        
        reranked = reranker_model.rerank(sample_query, documents)
        
        for doc in reranked:
            assert 0 <= doc["rerank_score"] <= 1


# ==================== LLM Client Tests ====================

@skip_no_core
@pytest.mark.unit
class TestLLMClient:
    """Test LLM client functionality."""
    
    def test_llm_singleton(self):
        """LLM client should be singleton."""
        from backend.core.llm_client import get_llm_client
        
        c1 = get_llm_client()
        c2 = get_llm_client()
        
        assert c1 is c2
    
    def test_llm_info(self, llm_client):
        """LLM client should return correct info."""
        info = llm_client.get_info()
        
        assert info["model"] == "qwen2.5:3b"
        assert info["num_gpu"] == 999
    
    @pytest.mark.integration
    def test_llm_availability(self, llm_client):
        """Should check Ollama availability."""
        available = llm_client.check_available()
        
        # Note: This will fail if Ollama is not running
        # We just check it returns a boolean
        assert isinstance(available, bool)


# ==================== RRF Fusion Tests ====================

@skip_no_core
@pytest.mark.unit
class TestRRFFusion:
    """Test RRF fusion algorithm."""
    
    def test_rrf_fusion(self, qdrant_connector):
        """RRF fusion should combine results correctly."""
        # Create mock results
        class MockHit:
            def __init__(self, id, score, payload):
                self.id = id
                self.score = score
                self.payload = payload
        
        dense = [
            MockHit("1", 0.9, {"text": "Doc 1"}),
            MockHit("2", 0.8, {"text": "Doc 2"}),
            MockHit("3", 0.7, {"text": "Doc 3"}),
        ]
        
        sparse = [
            MockHit("2", 0.95, {"text": "Doc 2"}),
            MockHit("1", 0.85, {"text": "Doc 1"}),
            MockHit("4", 0.75, {"text": "Doc 4"}),
        ]
        
        fused = qdrant_connector._rrf_fusion(dense, sparse)
        
        # With dense_weight=0.7 and sparse_weight=0.3:
        # Doc 1: 0.7/(60+1) + 0.3/(60+2) = 0.0163 (rank 1 dense, rank 2 sparse)
        # Doc 2: 0.7/(60+2) + 0.3/(60+1) = 0.0162 (rank 2 dense, rank 1 sparse)
        # Doc 1 should be first since dense has higher weight
        assert len(fused) == 4  # All unique docs
        assert fused[0]["id"] == "1"  # Doc 1 ranks highest with dense weight=0.7
        assert "score" in fused[0]


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
