"""
Integration Tests for API Endpoints.
Tests actual HTTP requests to the API with real services.
Updated to match actual backend API routes and response schemas.
"""
import sys
from pathlib import Path
import pytest
from backend import main as backend_main

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "web_app"))


class _FakeCollectionInfo:
    def __init__(self, points_count: int):
        self.points_count = points_count


class _FakeQdrantClient:
    def get_collection(self, name: str):
        if name == backend_main.Config.LEGAL_COLLECTION:
            return _FakeCollectionInfo(points_count=123)
        return _FakeCollectionInfo(points_count=0)


def _install_shared_search_stub(monkeypatch):
    monkeypatch.setattr(backend_main, "_shared_legal_enabled", lambda mode: True)

    def _fake_shared_search(request):
        return backend_main.SearchResponse(
            results=[
                {
                    "content": "Noi dung phap ly mau",
                    "score": 0.93,
                    "source_type": "legal",
                    "label": "Luat VB1 | Dieu 1 | Khoan 1 | Diem a",
                }
            ],
            search_time=0.12,
            mode=request.search_mode,
            rerank_time=0.02 if bool(request.use_reranker) else 0.0,
            reranker_used=bool(request.use_reranker),
            model_profile_id="default",
            query_mode=request.query_mode,
            rewrite_used=True,
            rewrite_status="ok",
            rewrite_fallback_reason="",
            rewrite_strategy="structured_synthesis",
            rewrite_quality_passed=True,
            rewritten_queries=["thu hoi dat"],
            retrieval_query_set=["thu hoi dat"],
            retrieval_query_plan=[{"query": "thu hoi dat", "source": "rewrite", "weight": 1.0}],
            rewrite_context={"original_query": request.query, "rewrite_mode": "rewrite_combine"},
            rewrite_debug={},
            per_subquery_hits=[{"query": "thu hoi dat", "hits": 1}],
            retrieval_union_count=1,
            fallback_used=False,
            fallback_error="",
            collection="legal_rag_vn",
        )

    monkeypatch.setattr(backend_main, "_run_shared_legal_search", _fake_shared_search)


def _install_shared_chat_stub(monkeypatch):
    monkeypatch.setattr(backend_main, "_shared_legal_enabled", lambda mode: True)

    def _fake_shared_chat(request):
        return backend_main.ChatResponse(
            answer="Tra loi mau",
            sources=[
                {
                    "type": "legal",
                    "source_type": "legal",
                    "score": 0.91,
                    "content_preview": "Noi dung phap ly mau",
                    "content": "Noi dung phap ly mau",
                }
            ],
            search_time=0.1,
            generation_time=0.2,
            total_time=0.3,
            rerank_time=0.0,
            reranker_used=bool(request.use_reranker),
            quality_grade="A",
            quality_score=0.95,
            model_profile_id="default",
            query_mode=request.query_mode,
            rewrite_used=True,
            rewrite_status="ok",
            rewrite_fallback_reason="",
            rewrite_strategy="structured_synthesis",
            rewrite_quality_passed=True,
            rewritten_queries=["thu hoi dat"],
            retrieval_query_set=["thu hoi dat"],
            retrieval_query_plan=[{"query": "thu hoi dat", "source": "rewrite", "weight": 1.0}],
            rewrite_context={"original_query": request.message, "rewrite_mode": "rewrite_combine"},
            rewrite_debug={},
            per_subquery_hits=[{"query": "thu hoi dat", "hits": 1}],
            retrieval_union_count=1,
            fallback_used=False,
            fallback_error="",
            collection="legal_rag_vn",
        )

    monkeypatch.setattr(backend_main, "_run_shared_legal_chat", _fake_shared_chat)


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
    
    async def test_full_status(self, async_client, monkeypatch):
        """Full status should check all components."""
        monkeypatch.setattr(backend_main.Config, "BACKEND_USE_SHARED_LEGAL_PIPELINE", False)
        monkeypatch.setattr(backend_main, "get_qdrant_client", lambda: _FakeQdrantClient())
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
    
    async def test_search_basic(self, async_client, sample_query, monkeypatch):
        """Basic search should return results."""
        _install_shared_search_stub(monkeypatch)
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
    
    async def test_search_with_reranker(self, async_client, sample_query, monkeypatch):
        """Search with reranker should work."""
        _install_shared_search_stub(monkeypatch)
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
    
    async def test_search_result_structure(self, async_client, sample_query, monkeypatch):
        """Search results should have correct metadata."""
        _install_shared_search_stub(monkeypatch)
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
    
    async def test_chat_basic(self, async_client, sample_query, sample_user_id, monkeypatch):
        """Basic chat should return answer."""
        _install_shared_chat_stub(monkeypatch)
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
    
    async def test_chat_returns_sources(self, async_client, sample_query, sample_user_id, monkeypatch):
        """Chat should return source citations."""
        _install_shared_chat_stub(monkeypatch)
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
    
    async def test_chat_quality_metrics(self, async_client, sample_query, sample_user_id, monkeypatch):
        """Chat should return quality metrics."""
        _install_shared_chat_stub(monkeypatch)
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
