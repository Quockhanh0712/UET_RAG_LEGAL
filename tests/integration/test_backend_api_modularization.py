"""Contract tests for Task 02 backend API modularization."""

from __future__ import annotations

import pytest
from fastapi import FastAPI


class _FakeCollectionInfo:
    def __init__(self, points_count: int):
        self.points_count = points_count


class _FakeQdrantClient:
    def get_collection(self, name: str):
        from backend import main as backend_main

        if name == backend_main.Config.LEGAL_COLLECTION:
            return _FakeCollectionInfo(points_count=123)
        return _FakeCollectionInfo(points_count=0)


def _install_shared_search_stub(monkeypatch):
    from backend import main as backend_main

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
            rerank_time=0.0,
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
    from backend import main as backend_main

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


@pytest.mark.integration
def test_main_exports_fastapi_app():
    from backend.main import app

    assert isinstance(app, FastAPI)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_status_route_contract(async_client, monkeypatch):
    from backend import main as backend_main

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_route_contract(async_client, monkeypatch):
    _install_shared_search_stub(monkeypatch)

    response = await async_client.post(
        "/api/search",
        json={
            "query": "thu hoi dat",
            "top_k": 5,
            "search_mode": "legal",
            "use_reranker": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "search_time" in data
    assert "mode" in data
    assert "reranker_used" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_route_contract(async_client, monkeypatch):
    _install_shared_chat_stub(monkeypatch)

    response = await async_client.post(
        "/api/chat",
        json={
            "message": "thu hoi dat",
            "user_id": "test_user",
            "search_mode": "legal",
            "use_reranker": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "search_time" in data
    assert "generation_time" in data
    assert "total_time" in data
