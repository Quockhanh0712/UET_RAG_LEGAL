"""
Integration Tests for API Endpoints.
Tests actual HTTP requests to the API with real services.
Updated to match actual backend API routes and response schemas.
"""
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "web_app"))


# ==================== Status API Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestStatusAPI:
    """Test status endpoints."""
    
    async def test_root_endpoint(self, async_client):
        """Root endpoint should return service info."""
        response = await async_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "Legal RAG API"
        assert data["version"] == "1.0.0"
    
    async def test_full_status(self, async_client):
        """Full status should check all components."""
        response = await async_client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ok"
        assert "legal_collection" in data
        assert "legal_documents" in data
        assert "user_collection" in data
        assert "embedding_model" in data
        assert "llm_model" in data
        # Verify Cloud collection
        assert data["legal_collection"] == "legal_rag_vn"
        assert data["legal_documents"] > 0


# ==================== Search API Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestSearchAPI:
    """Test /api/search endpoint."""
    
    async def test_search_basic(self, async_client, sample_query):
        """Basic search should return results."""
        response = await async_client.post("/api/search", json={
            "query": sample_query,
            "top_k": 5,
            "search_mode": "legal",
            "use_reranker": False
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "search_time" in data
        assert "mode" in data
        assert "reranker_used" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0
    
    async def test_search_with_reranker(self, async_client, sample_query):
        """Search with reranker should work."""
        response = await async_client.post("/api/search", json={
            "query": sample_query,
            "top_k": 3,
            "search_mode": "legal",
            "use_reranker": True
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "reranker_used" in data
        # Results should have scores
        if data["results"]:
            assert "score" in data["results"][0]
    
    async def test_search_result_structure(self, async_client, sample_query):
        """Search results should have correct metadata."""
        response = await async_client.post("/api/search", json={
            "query": sample_query,
            "top_k": 3,
            "search_mode": "legal"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        if data["results"]:
            result = data["results"][0]
            assert "content" in result
            assert "score" in result
            assert "source_type" in result


# ==================== Chat API Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestChatAPI:
    """Test /api/chat endpoint."""
    
    async def test_chat_basic(self, async_client, sample_query, sample_user_id):
        """Basic chat should return answer."""
        response = await async_client.post("/api/chat", json={
            "message": sample_query,
            "user_id": sample_user_id,
            "search_mode": "legal",
            "use_reranker": False
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "search_time" in data
        assert "generation_time" in data
        assert "total_time" in data
        assert len(data["answer"]) > 0
    
    async def test_chat_returns_sources(self, async_client, sample_query, sample_user_id):
        """Chat should return source citations."""
        response = await async_client.post("/api/chat", json={
            "message": sample_query,
            "user_id": sample_user_id,
            "search_mode": "legal",
            "use_reranker": False
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sources" in data
        assert isinstance(data["sources"], list)
        
        if data["sources"]:
            source = data["sources"][0]
            # Actual source fields from backend
            assert "type" in source or "source_type" in source
            assert "score" in source
    
    async def test_chat_quality_metrics(self, async_client, sample_query, sample_user_id):
        """Chat should return quality metrics."""
        response = await async_client.post("/api/chat", json={
            "message": sample_query,
            "user_id": sample_user_id,
            "search_mode": "legal"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Quality metrics may be present
        if "quality_grade" in data:
            assert data["quality_grade"] in ["A", "B", "C", "D", "F"]


# ==================== Document API Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestDocumentAPI:
    """Test /api/documents endpoint."""
    
    async def test_list_documents(self, async_client, sample_user_id):
        """Should list user documents."""
        response = await async_client.get(
            "/api/documents",
            params={"user_id": sample_user_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data


# ==================== Session API Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestSessionAPI:
    """Test /api/sessions endpoint."""
    
    async def test_list_sessions(self, async_client, sample_user_id):
        """Should list user sessions."""
        response = await async_client.get(
            "/api/sessions",
            params={"user_id": sample_user_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data


# ==================== Error Handling Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandling:
    """Test API error handling."""
    
    async def test_invalid_json(self, async_client):
        """Invalid JSON should return 422."""
        response = await async_client.post(
            "/api/search",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    async def test_404_not_found(self, async_client):
        """Invalid endpoint should return 404."""
        response = await async_client.get("/api/nonexistent")
        
        assert response.status_code == 404


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
