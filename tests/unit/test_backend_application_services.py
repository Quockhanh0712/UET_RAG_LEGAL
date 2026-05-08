"""Task 03 tests for backend application services and delegation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.api.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from backend.services.shared_legal_services import (
    SharedLegalChatService,
    SharedLegalSearchService,
)
from src.adapters.backend_adapter import build_chat_payload, build_search_payload


def test_shared_legal_search_service_builds_contract_response():
    calls = {"apply_env": 0, "payload": 0}

    class _Settings:
        def apply_env(self):
            calls["apply_env"] += 1

    class _Pipeline:
        embedding_model = SimpleNamespace(embedding_dim=1024)

        def search(self, **kwargs):
            return {"search_result": True, "kwargs": kwargs}

    def _build_payload(result, *, mode, collection, model_profile_id, query_mode):
        calls["payload"] += 1
        assert mode == "legal"
        assert collection == "legal_rag_vn"
        assert model_profile_id == "default"
        assert query_mode == "original"
        assert result["search_result"] is True
        return {
            "results": [{"content": "x", "score": 1.0, "source_type": "legal"}],
            "search_time": 0.11,
            "mode": mode,
            "model_profile_id": model_profile_id,
            "query_mode": query_mode,
        }

    service = SharedLegalSearchService(
        normalize_query_mode=lambda mode: "original",
        backend_runtime_settings=lambda **kwargs: _Settings(),
        get_legal_pipeline=lambda settings: _Pipeline(),
        validate_qdrant_contract=lambda settings, dim: SimpleNamespace(collection_name="legal_rag_vn"),
        build_query_rewrite_service=lambda profile_id, query_mode: None,
        build_search_payload=_build_payload,
        get_model_profile=lambda profile_id: SimpleNamespace(id="default"),
        get_setting=lambda name, default: default,
    )

    response = service.run(
        SearchRequest(
            query="thu hoi dat",
            search_mode="legal",
            top_k=5,
            use_reranker=False,
        )
    )

    assert isinstance(response, SearchResponse)
    assert response.mode == "legal"
    assert calls["apply_env"] == 1
    assert calls["payload"] == 1


def test_shared_legal_chat_service_persists_history():
    saved = {"count": 0}

    class _Settings:
        def apply_env(self):
            return None

    class _Pipeline:
        embedding_model = SimpleNamespace(embedding_dim=1024)

        def query(self, **kwargs):
            return SimpleNamespace(answer="Tra loi dich vu")

    def _save_chat_history(**kwargs):
        saved["count"] += 1
        assert kwargs["answer"] == "Tra loi dich vu"
        assert isinstance(kwargs["sources"], list)

    service = SharedLegalChatService(
        normalize_query_mode=lambda mode: "original",
        backend_runtime_settings=lambda **kwargs: _Settings(),
        get_legal_pipeline=lambda settings: _Pipeline(),
        validate_qdrant_contract=lambda settings, dim: SimpleNamespace(collection_name="legal_rag_vn"),
        build_query_rewrite_service=lambda profile_id, query_mode: None,
        build_remote_llm=lambda profile_id: object(),
        build_chat_payload=lambda result, **kwargs: {
            "answer": kwargs["answer"],
            "sources": [{"type": "legal", "score": 1.0}],
            "search_time": 0.1,
            "generation_time": 0.2,
            "total_time": 0.3,
            "rerank_time": 0.0,
            "reranker_used": False,
            "model_profile_id": kwargs["model_profile_id"],
            "query_mode": kwargs["query_mode"],
            "fallback_used": kwargs["fallback_used"],
            "fallback_error": kwargs["fallback_error"],
            "collection": kwargs["collection"],
        },
        get_model_profile=lambda profile_id: SimpleNamespace(id="default"),
        get_setting=lambda name, default: default,
        save_chat_history=_save_chat_history,
    )

    response = service.run(
        ChatRequest(
            message="thu hoi dat",
            search_mode="legal",
            top_k=5,
            use_reranker=False,
        )
    )

    assert isinstance(response, ChatResponse)
    assert response.answer == "Tra loi dich vu"
    assert saved["count"] == 1


@pytest.mark.asyncio
async def test_chat_route_delegates_to_service(monkeypatch):
    from backend import main as backend_main

    class _Service:
        def __init__(self):
            self.called = False

        def run(self, request):
            self.called = True
            return ChatResponse(
                answer="delegated",
                sources=[],
                search_time=0.0,
                generation_time=0.0,
                total_time=0.0,
            )

    service = _Service()
    monkeypatch.setattr(backend_main, "_shared_legal_enabled", lambda mode: True)
    monkeypatch.setattr(backend_main, "get_shared_legal_chat_service", lambda: service)

    response = await backend_main.chat(ChatRequest(message="x", search_mode="legal"))
    assert response.answer == "delegated"
    assert service.called is True


@pytest.mark.asyncio
async def test_search_route_delegates_to_service(monkeypatch):
    from backend import main as backend_main

    class _Service:
        def __init__(self):
            self.called = False

        def run(self, request):
            self.called = True
            return SearchResponse(
                results=[],
                search_time=0.0,
                mode=request.search_mode,
            )

    service = _Service()
    monkeypatch.setattr(backend_main, "_shared_legal_enabled", lambda mode: True)
    monkeypatch.setattr(backend_main, "get_shared_legal_search_service", lambda: service)

    response = await backend_main.search(SearchRequest(query="x", search_mode="legal"))
    assert response.mode == "legal"
    assert service.called is True


def test_shared_runtime_gate_supports_legal_user_and_hybrid(monkeypatch):
    from backend import main as backend_main

    monkeypatch.setattr(backend_main.Config, "BACKEND_USE_SHARED_LEGAL_PIPELINE", True)

    assert backend_main._shared_legal_enabled("legal") is True
    assert backend_main._shared_legal_enabled("user") is True
    assert backend_main._shared_legal_enabled("hybrid") is True
    assert backend_main._shared_legal_enabled("unknown") is False


def test_shared_search_service_handles_user_mode_user_wide_results():
    rewrite_calls = {"count": 0}

    class _Settings:
        def apply_env(self):
            return None

    class _IntentClassifier:
        @staticmethod
        def classify(question):
            return SimpleNamespace(intent=SimpleNamespace(value="factual"))

    class _Pipeline:
        embedding_model = SimpleNamespace(embedding_dim=1024)

        def __init__(self):
            self.intent_classifier = _IntentClassifier()

        def initialize(self):
            return None

        @staticmethod
        def _merge_query_rankings_rrf(rankings, limit):
            return rankings[0][:limit]

    class _UserStore:
        @staticmethod
        def search(**kwargs):
            assert kwargs["session_id"] == "session-b"
            assert kwargs["document_scope"] == "current_session"
            assert kwargs["query"] == "dieu 47 la gi"
            return [
                {
                    "content": "Dieu 47 noi dung",
                    "score": 0.91,
                    "filename": "35_2024_QH15_588811.pdf",
                    "doc_id": "doc-1",
                    "page": 2,
                    "payload": {
                        "chunk_id": "chunk-1",
                        "content": "Dieu 47 noi dung",
                        "doc_id": "doc-1",
                        "filename": "35_2024_QH15_588811.pdf",
                        "page_number": 2,
                        "source_type": "user",
                        "citation_full": "PDF USER_UPLOAD | Dieu 47",
                        "legal_path": "Dieu 47",
                        "ten_van_ban": "35_2024_QH15_588811.pdf",
                    },
                }
            ]

        @staticmethod
        def get_user_documents(user_id):
            return [SimpleNamespace(id="doc-1", session_id="session-b", status="ready")]

    service = SharedLegalSearchService(
        normalize_query_mode=lambda mode: "rewrite_combine",
        backend_runtime_settings=lambda **kwargs: _Settings(),
        get_legal_pipeline=lambda settings: _Pipeline(),
        validate_qdrant_contract=lambda settings, dim: SimpleNamespace(collection_name="legal_documents__d1024"),
        build_query_rewrite_service=lambda profile_id, query_mode: SimpleNamespace(
            rewrite=lambda question: rewrite_calls.__setitem__("count", rewrite_calls["count"] + 1)
        ),
        build_search_payload=build_search_payload,
        get_model_profile=lambda profile_id: SimpleNamespace(id="default"),
        get_setting=lambda name, default: default,
        get_user_store=lambda: _UserStore(),
    )

    response = service.run(
        SearchRequest(
            query="dieu 47 la gi",
            user_id="user-1",
            session_id="session-b",
            search_mode="user",
            top_k=5,
            use_reranker=False,
        )
    )

    assert response.mode == "user"
    assert response.collection == "user_docs_private"
    assert response.results[0]["type"] == "user"
    assert response.results[0]["citation_full"] == "PDF USER_UPLOAD | Dieu 47"
    assert response.retrieval_debug["document_scope"] == "current_session"
    assert response.retrieval_debug["user_query_strategy"] == "original_only"
    assert rewrite_calls["count"] == 0


def test_shared_chat_service_handles_hybrid_mode_with_only_user_hits():
    saved = {"count": 0}

    class _Settings:
        def apply_env(self):
            return None

    class _Pipeline:
        embedding_model = SimpleNamespace(embedding_dim=1024)

        def __init__(self):
            self.intent_classifier = SimpleNamespace(
                classify=lambda question: SimpleNamespace(intent=SimpleNamespace(value="factual"))
            )

        def _retrieve(self, **kwargs):
            return [], {
                "intent": "factual",
                "retrieval_time": 0.02,
                "rerank_time": 0.0,
                "used_reranker": False,
                "graph_expansion_used": False,
                "graph_mode": "none",
                "rewrite_time": 0.0,
                "rewrite_used": True,
                "rewrite_status": "ok",
                "rewrite_fallback_reason": "",
                "rewritten_queries": ["dieu 47"],
                "missing_numbers": [],
                "per_subquery_hits": [{"query": "dieu 47", "hits": 0}],
                "retrieval_union_count": 0,
                "rewrite_mode": "rewrite_combine",
                "rewrite_original_query": "hoi dieu 47",
                "rewrite_queries_compact": ["dieu 47"],
                "rewrite_strategy": "structured_synthesis",
                "rewrite_quality_passed": True,
                "retrieval_query_set": ["dieu 47"],
                "retrieval_query_plan": [{"query": "dieu 47", "source": "rewrite", "weight": 1.0}],
                "query_facets": {},
                "rewrite_debug": {},
            }

        @staticmethod
        def _merge_query_rankings_rrf(rankings, limit):
            return rankings[0][:limit]

        @staticmethod
        def _merge_results(primary, secondary):
            return list(primary) + list(secondary)

    class _UserStore:
        @staticmethod
        def search(**kwargs):
            assert kwargs["query"] == "hoi dieu 47"
            assert kwargs["document_scope"] == "current_session"
            return [
                {
                    "content": "Dieu 47 noi dung",
                    "score": 0.91,
                    "filename": "35_2024_QH15_588811.pdf",
                    "doc_id": "doc-1",
                    "page": 2,
                    "payload": {
                        "chunk_id": "chunk-1",
                        "content": "Dieu 47 noi dung",
                        "doc_id": "doc-1",
                        "filename": "35_2024_QH15_588811.pdf",
                        "page_number": 2,
                        "source_type": "user",
                        "citation_full": "PDF USER_UPLOAD | Dieu 47",
                        "legal_path": "Dieu 47",
                        "ten_van_ban": "35_2024_QH15_588811.pdf",
                    },
                }
            ]

        @staticmethod
        def get_user_documents(user_id):
            return [SimpleNamespace(id="doc-1", session_id="session-b", status="ready")]

    service = SharedLegalChatService(
        normalize_query_mode=lambda mode: "rewrite_combine",
        backend_runtime_settings=lambda **kwargs: _Settings(),
        get_legal_pipeline=lambda settings: _Pipeline(),
        validate_qdrant_contract=lambda settings, dim: SimpleNamespace(collection_name="legal_documents__d1024"),
        build_query_rewrite_service=lambda profile_id, query_mode: SimpleNamespace(rewrite=lambda question: None),
        build_remote_llm=lambda profile_id: SimpleNamespace(generate=lambda question, context: "Tra loi tu hybrid"),
        build_chat_payload=build_chat_payload,
        get_model_profile=lambda profile_id: SimpleNamespace(id="default"),
        get_setting=lambda name, default: default,
        save_chat_history=lambda **kwargs: saved.__setitem__("count", saved["count"] + 1),
        get_user_store=lambda: _UserStore(),
    )

    response = service.run(
        ChatRequest(
            message="hoi dieu 47",
            user_id="user-1",
            session_id="session-b",
            search_mode="hybrid",
            top_k=5,
            use_reranker=False,
        )
    )

    assert response.answer == "Tra loi tu hybrid"
    assert response.sources[0]["type"] == "user"
    assert response.retrieval_debug["user_query_strategy"] == "original_only"
    assert saved["count"] == 1


def test_shared_search_service_flags_scope_broaden_when_ready_docs_exist_outside_session():
    class _Settings:
        def apply_env(self):
            return None

    class _Pipeline:
        embedding_model = SimpleNamespace(embedding_dim=1024)

        def __init__(self):
            self.intent_classifier = SimpleNamespace(
                classify=lambda question: SimpleNamespace(intent=SimpleNamespace(value="factual"))
            )

        def initialize(self):
            return None

        @staticmethod
        def _merge_query_rankings_rrf(rankings, limit):
            return rankings[0][:limit]

    class _UserStore:
        @staticmethod
        def search(**kwargs):
            return []

        @staticmethod
        def get_user_documents(user_id):
            return [
                SimpleNamespace(id="doc-1", session_id="session-other", status="ready"),
                SimpleNamespace(id="doc-2", session_id="session-current", status="running"),
            ]

    service = SharedLegalSearchService(
        normalize_query_mode=lambda mode: "original",
        backend_runtime_settings=lambda **kwargs: _Settings(),
        get_legal_pipeline=lambda settings: _Pipeline(),
        validate_qdrant_contract=lambda settings, dim: SimpleNamespace(collection_name="legal_documents__d1024"),
        build_query_rewrite_service=lambda profile_id, query_mode: SimpleNamespace(rewrite=lambda question: None),
        build_search_payload=build_search_payload,
        get_model_profile=lambda profile_id: SimpleNamespace(id="default"),
        get_setting=lambda name, default: default,
        get_user_store=lambda: _UserStore(),
    )

    response = service.run(
        SearchRequest(
            query="noi dung can tim",
            user_id="user-1",
            session_id="session-current",
            search_mode="user",
            top_k=5,
            use_reranker=False,
            document_scope="current_session",
        )
    )

    assert response.retrieval_debug["scope_broaden_available"] is True
    assert response.retrieval_debug["no_result_reason"] == "scope_filtered_out"
