"""
Unit tests for Legal RAG Pipeline (src/legal_rag_pipeline.py).
Tests core pipeline logic, citation formatting, LLM error handling,
Qdrant fallback, reranker fallback, and graph expansion.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.legal_rag_pipeline import LegalAnswer, LegalLLM, LegalRAGPipeline
from src.qdrant_store import SearchResult
from src.query_intent import IntentResult, QueryIntent


# ============================================================================
# 1. LegalAnswer & Citation Formatting Tests
# ============================================================================

class TestLegalAnswer:
    def test_format_with_citations_empty(self):
        answer = LegalAnswer(
            answer="Đây là câu trả lời.",
            citations=[],
            intent="concept",
            retrieval_time=0.1,
            generation_time=0.5,
        )
        assert answer.format_with_citations() == "Đây là câu trả lời."

    def test_format_with_citations_full(self):
        answer = LegalAnswer(
            answer="Hình phạt tù có thời hạn.",
            citations=[
                {
                    "doc_id": "doc_01",
                    "dieu_so": "38",
                    "khoan_so": "1",
                    "diem_so": "",
                    "citation_full": "Khoản 1 Điều 38 Bộ luật Hình sự 2015",
                }
            ],
            intent="penalty",
            retrieval_time=0.1,
            generation_time=0.5,
        )
        formatted = answer.format_with_citations()
        assert "Hình phạt tù có thời hạn." in formatted
        assert "**CAN CU PHAP LY:**" in formatted
        assert "1. Khoản 1 Điều 38 Bộ luật Hình sự 2015" in formatted

    def test_format_with_citations_structured(self):
        answer = LegalAnswer(
            answer="Quy định về thời hiệu.",
            citations=[
                {
                    "doc_id": "doc_02",
                    "ten_van_ban": "Bộ luật Dân sự",
                    "loai_van_ban": "Luật",
                    "co_quan": "Quốc hội",
                    "dieu_so": "149",
                    "khoan_so": "2",
                    "diem_so": "a",
                }
            ],
            intent="lookup",
            retrieval_time=0.1,
            generation_time=0.5,
        )
        formatted = answer.format_with_citations()
        assert "Dieu 149 - Khoan 2 - Diem a)" in formatted
        assert "[Luật] Bộ luật Dân sự - Quốc hội" in formatted

    def test_format_with_citations_deduplication(self):
        citations = [
            {
                "doc_id": "doc_01",
                "dieu_so": "10",
                "khoan_so": "1",
                "diem_so": "",
                "citation_full": "Điều 10 Luật Đất đai",
            },
            {
                "doc_id": "doc_01",
                "dieu_so": "10",
                "khoan_so": "1",
                "diem_so": "",
                "citation_full": "Điều 10 Luật Đất đai (trùng lặp)",
            },
        ]
        answer = LegalAnswer(
            answer="Phân loại đất.",
            citations=citations,
            intent="lookup",
            retrieval_time=0.1,
            generation_time=0.5,
        )
        formatted = answer.format_with_citations()
        assert formatted.count("Điều 10 Luật Đất đai") == 1


# ============================================================================
# 2. LegalLLM Error Handling Tests ("nếu LLM lỗi ko sao")
# ============================================================================

class TestLegalLLM:
    def test_llm_ollama_network_error(self):
        llm = LegalLLM()
        llm.provider = "ollama"

        with patch("requests.post", side_effect=Exception("Connection refused")):
            result = llm.generate("query", "context")
            assert "Loi khi goi LLM" in result or "Loi" in result
            assert "Connection refused" in result

    def test_llm_gemini_error_handling(self):
        llm = LegalLLM()
        llm.provider = "gemini"

        with patch("src.llm_client.generate_answer", side_effect=Exception("Quota exceeded")):
            result = llm.generate("query", "context")
            assert "Loi khi goi LLM" in result or "Quota exceeded" in result

    def test_llm_fallback_message(self):
        llm = LegalLLM()
        llm.provider = "unknown_provider"

        result = llm.generate("query", "context")
        assert "LLM khong kha dung" in result


# ============================================================================
# 3. LegalRAGPipeline Tests
# ============================================================================

class TestLegalRAGPipeline:
    def _create_mock_pipeline(self):
        pipeline = LegalRAGPipeline()
        pipeline._initialized = True

        pipeline.embedding_model = MagicMock()
        pipeline.embedding_model.encode_query.return_value = [0.1] * 768
        pipeline.embedding_model.embedding_dim = 768
        pipeline.embedding_model.model_name = "test-model"

        pipeline.intent_classifier = MagicMock()
        pipeline.intent_classifier.classify.return_value = IntentResult(
            intent=QueryIntent.GENERAL,
            confidence=0.9,
            top_k=3,
            dense_weight=0.6,
            sparse_weight=0.4,
            graph_mode="none",
        )

        pipeline.qdrant_store = MagicMock()
        pipeline.graph_store = MagicMock()
        pipeline.graph_store.enabled = False
        pipeline.llm = MagicMock()
        pipeline.llm.provider = "test"
        pipeline.llm.model = "test-model"

        return pipeline

    def test_query_success(self):
        pipeline = self._create_mock_pipeline()
        pipeline.qdrant_store.hybrid_search.return_value = [
            SearchResult(
                chunk_id="chunk_1",
                content="Khoản 1 Điều 12: Quy định bảo vệ môi trường.",
                score=0.95,
                metadata={
                    "doc_id": "law_01",
                    "dieu_so": "12",
                    "khoan_so": "1",
                    "citation_full": "Khoản 1 Điều 12 Luật Bảo vệ môi trường",
                },
            )
        ]
        pipeline.llm.generate.return_value = "Theo Khoản 1 Điều 12, phải bảo vệ môi trường."

        answer = pipeline.query("Bảo vệ môi trường như thế nào?", use_reranker=False)

        assert isinstance(answer, LegalAnswer)
        assert answer.answer == "Theo Khoản 1 Điều 12, phải bảo vệ môi trường."
        assert len(answer.citations) == 1
        assert answer.citations[0]["doc_id"] == "law_01"
        assert answer.retrieval_time >= 0
        assert answer.generation_time >= 0

    def test_query_when_llm_fails(self):
        """Pipeline must NOT crash if LLM throws an error; answer contains error description."""
        pipeline = self._create_mock_pipeline()
        pipeline.qdrant_store.hybrid_search.return_value = [
            SearchResult(
                chunk_id="chunk_1",
                content="Nội dung điều luật.",
                score=0.9,
                metadata={"doc_id": "law_01", "citation_full": "Điều 1 Luật A"},
            )
        ]
        pipeline.llm.generate.side_effect = RuntimeError("LLM service timeout")

        answer = pipeline.query("Câu hỏi pháp lý?", use_reranker=False)

        assert isinstance(answer, LegalAnswer)
        assert "Loi khi goi LLM" in answer.answer
        assert "LLM service timeout" in answer.answer
        assert len(answer.citations) == 1

    def test_query_when_qdrant_fails(self):
        """Pipeline must NOT crash if Qdrant throws connection/server error."""
        pipeline = self._create_mock_pipeline()
        pipeline.qdrant_store.hybrid_search.side_effect = RuntimeError("Qdrant connection refused")

        answer = pipeline.query("Câu hỏi khi Qdrant chết?", use_reranker=False)

        assert isinstance(answer, LegalAnswer)
        assert "Khong tim thay thong tin phap ly" in answer.answer
        assert len(answer.citations) == 0

    def test_query_is_active_filter_fallback(self):
        """If is_active filter causes search failure, pipeline retries without is_active."""
        pipeline = self._create_mock_pipeline()

        call_count = 0

        def hybrid_search_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            filters = kwargs.get("filter_conditions", {})
            if "is_active" in filters:
                raise ValueError("Index has no keyword index for is_active")
            return [
                SearchResult(
                    chunk_id="chunk_legacy",
                    content="Nội dung từ index cũ.",
                    score=0.88,
                    metadata={"doc_id": "law_legacy"},
                )
            ]

        pipeline.qdrant_store.hybrid_search.side_effect = hybrid_search_side_effect
        pipeline.llm.generate.return_value = "Câu trả lời từ index cũ."

        answer = pipeline.query("Truy vấn index cũ", use_reranker=False)

        assert call_count == 2
        assert len(answer.citations) == 1
        assert answer.citations[0]["doc_id"] == "law_legacy"

    def test_query_reranker_fallback_on_error(self):
        """If reranker throws an error, pipeline safely falls back to base results."""
        pipeline = self._create_mock_pipeline()
        pipeline.qdrant_store.hybrid_search.return_value = [
            SearchResult(
                chunk_id="c1",
                content="Văn bản 1",
                score=0.9,
                metadata={"doc_id": "doc1"},
            ),
            SearchResult(
                chunk_id="c2",
                content="Văn bản 2",
                score=0.8,
                metadata={"doc_id": "doc2"},
            ),
        ]
        pipeline.llm.generate.return_value = "Trả lời"

        with patch.object(pipeline, "_rerank_results", side_effect=RuntimeError("GPU OOM")):
            answer = pipeline.query("Câu hỏi", use_reranker=True, top_k=2)
            assert answer.used_reranker is False
            assert len(answer.citations) == 2

    def test_get_stats_resilience(self):
        pipeline = self._create_mock_pipeline()
        pipeline.qdrant_store.get_stats.side_effect = RuntimeError("Qdrant offline")

        stats = pipeline.get_stats()
        assert "qdrant" in stats
        assert "error" in stats["qdrant"]
        assert stats["embedding_dim"] == 768
