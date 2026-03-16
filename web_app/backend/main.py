"""
Legal RAG Web API - FastAPI Backend

Kiến trúc Dual-Store:
1. legal_rag_hybrid - Kho Luật chính (100k+ điều luật)
2. user_docs_private - Kho Cá nhân (per user/session)

Endpoints:
- POST /api/chat - Chat với AI (hỗ trợ reranker toggle)
- POST /api/search - Tìm kiếm hybrid
- POST /api/upload - Upload file người dùng
- GET /api/documents - Lấy danh sách tài liệu user
- DELETE /api/documents/{doc_id} - Xóa tài liệu
- GET /api/history - Lịch sử chat
- POST /api/score - Chấm điểm câu trả lời
- GET/PUT /api/settings/reranker - Toggle reranker

Author: Legal RAG System
Date: 2024-12
Updated: 2024-12 - Added Vietnamese Reranker & Answer Scoring
"""

import os
import sys
import uuid
import time
import json
import shutil
import base64
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

# Add parent to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Local imports
from user_store import get_user_store, UserDocumentStore
from smart_chunker import SmartRecursiveChunker
from vietnamese_reranker import get_reranker, get_answer_scorer
from quality_monitor import get_quality_monitor
from database import (
    get_db,
    save_chat_session,
    save_message,
    save_answer_metrics,
    save_message_sources,
    get_session_messages,
    get_metrics_stats
)
from src.adapters.backend_adapter import build_chat_payload, build_search_payload
from src.config.runtime_settings import load_runtime_settings
from src.factories.legal_pipeline_factory import get_legal_pipeline
from src.legal_rag_pipeline import LegalAnswer
from src.runtime.qdrant_contract import validate_qdrant_contract
from src.services.model_profile_registry import (
    get_default_model_profile_id,
    get_model_profile,
    list_model_profiles,
)
from src.services.query_rewrite_service import QueryRewriteService
from src.services.remote_llm_client import RemoteLLMClient

# Environment setup
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("OLLAMA_NUM_GPU", "999")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Qdrant Cloud
    QDRANT_USE_CLOUD = os.getenv("QDRANT_USE_CLOUD", "false").lower() == "true"
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    
    # Collections
    LEGAL_COLLECTION = os.getenv("QDRANT_LEGAL_COLLECTION", "legal_rag_vn")  # Core legal docs
    USER_COLLECTION = "user_docs_private"   # User uploads
    
    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "huyydangg/DEk21_hcmute_embedding")
    
    # LLM
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    BACKEND_USE_SHARED_LEGAL_PIPELINE = os.getenv("BACKEND_USE_SHARED_LEGAL_PIPELINE", "false").lower() == "true"
    BACKEND_RUNTIME_PROFILE = os.getenv("BACKEND_RUNTIME_PROFILE", "chainlit_demo")
    
    # Paths
    UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
    BM25_CACHE_PATH = PROJECT_ROOT / "data" / "qdrant_db" / "bm25_encoder.json"
    
    # Search
    TOP_K = int(os.getenv("TOP_K", "10"))
    
    # Reranker Settings
    RERANKER_ENABLED = True  # Default enabled
    RERANKER_ALPHA = 1     # Weight: 0.5 = balanced (50% reranker + 50% original)
    
    # Quality Evaluation Settings
    TOP_CONTEXTS_FOR_LLM = 10  # Number of contexts sent to LLM and used for quality eval


# Global settings state
_app_settings = {
    "reranker_enabled": Config.RERANKER_ENABLED,
    "reranker_alpha": Config.RERANKER_ALPHA,
    "top_contexts_for_llm": Config.TOP_CONTEXTS_FOR_LLM
}


# Ensure upload directory exists
Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    search_mode: str = "hybrid"  # legal, user, hybrid
    top_k: int = 10
    use_reranker: Optional[bool] = None  # None = use global setting
    model_profile_id: Optional[str] = None
    query_mode: str = "original"
    response_mode: str = "chat"
    require_active_only: bool = True
    use_graph: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    search_time: float
    generation_time: float
    total_time: float
    rerank_time: float = 0.0  # Time spent on reranking
    reranker_used: bool = False  # Whether reranker was applied
    
    # Quality metrics (added for consistency)
    quality_grade: Optional[str] = None
    quality_score: Optional[float] = None
    bert_score: Optional[float] = None
    hallucination_score: Optional[float] = None
    factuality_score: Optional[float] = None
    context_relevance: Optional[float] = None
    quality_feedback: Optional[str] = None
    model_profile_id: Optional[str] = None
    query_mode: Optional[str] = None
    rewrite_used: bool = False
    rewrite_status: str = "disabled"
    rewrite_fallback_reason: str = ""
    rewrite_strategy: str = "structured_synthesis"
    rewrite_quality_passed: bool = False
    rewritten_queries: List[str] = Field(default_factory=list)
    retrieval_query_set: List[str] = Field(default_factory=list)
    retrieval_query_plan: List[Dict[str, Any]] = Field(default_factory=list)
    rewrite_context: Dict[str, Any] = Field(default_factory=dict)
    rewrite_debug: Dict[str, Any] = Field(default_factory=dict)
    per_subquery_hits: List[Dict] = Field(default_factory=list)
    retrieval_union_count: int = 0
    fallback_used: bool = False
    fallback_error: str = ""
    collection: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    session_id: Optional[str] = None
    search_mode: str = "hybrid"  # legal, user, hybrid
    top_k: int = 10
    use_reranker: Optional[bool] = None  # None = use global setting
    model_profile_id: Optional[str] = None
    query_mode: str = "original"
    require_active_only: bool = True

class SearchResponse(BaseModel):
    results: List[Dict]
    search_time: float
    mode: str
    rerank_time: float = 0.0
    reranker_used: bool = False
    model_profile_id: Optional[str] = None
    query_mode: Optional[str] = None
    rewrite_used: bool = False
    rewrite_status: str = "disabled"
    rewrite_fallback_reason: str = ""
    rewrite_strategy: str = "structured_synthesis"
    rewrite_quality_passed: bool = False
    rewritten_queries: List[str] = Field(default_factory=list)
    retrieval_query_set: List[str] = Field(default_factory=list)
    retrieval_query_plan: List[Dict[str, Any]] = Field(default_factory=list)
    rewrite_context: Dict[str, Any] = Field(default_factory=dict)
    rewrite_debug: Dict[str, Any] = Field(default_factory=dict)
    per_subquery_hits: List[Dict] = Field(default_factory=list)
    retrieval_union_count: int = 0
    fallback_used: bool = False
    fallback_error: str = ""
    collection: Optional[str] = None


class ScoreRequest(BaseModel):
    """Request để chấm điểm câu trả lời"""
    query: str
    answer: str
    contexts: List[Dict]  # List of context documents
    top_k: int = 5  # Số contexts để đánh giá


class ScoreResponse(BaseModel):
    """Response chấm điểm câu trả lời"""
    overall_score: float  # 0-100
    grade: str  # A, B, C, D, F
    feedback: str
    details: Dict  # Detailed scoring breakdown


class RerankerSettingsResponse(BaseModel):
    """Reranker settings"""
    enabled: bool
    alpha: float
    model_name: str
    model_loaded: bool

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    upload_time: str
    status: str

class UploadResponse(BaseModel):
    success: bool
    doc_id: str
    filename: str
    chunk_count: int
    message: str

class ChatHistoryItem(BaseModel):
    id: str
    title: str
    timestamp: str
    preview: str


# ============================================================================
# Global Resources (Singleton Pattern)
# ============================================================================

_qdrant_client = None
_embedding_model = None
_bm25_encoder = None
_ollama_client = None
_chat_history: Dict[str, List[Dict]] = {}  # session_id -> messages


def _shared_legal_enabled(search_mode: str) -> bool:
    return bool(Config.BACKEND_USE_SHARED_LEGAL_PIPELINE and str(search_mode or "").strip().lower() == "legal")


def _normalize_query_mode(query_mode: str | None) -> str:
    value = str(query_mode or "original").strip().lower()
    if value in {"rewrite_combine", "rewrite+combine", "rewrite"}:
        return "rewrite_combine"
    return "original"


def _backend_runtime_settings(query_rewriter_enabled: bool, query_rewriter_force: bool):
    base_settings = load_runtime_settings(Config.BACKEND_RUNTIME_PROFILE)
    return replace(
        base_settings,
        query_rewriter_enabled=bool(query_rewriter_enabled),
        query_rewriter_force=bool(query_rewriter_force),
    )


def _build_remote_llm(profile_id: str | None) -> RemoteLLMClient:
    profile = get_model_profile(profile_id)
    return RemoteLLMClient(
        provider_type=profile.provider_type,
        base_url=profile.base_url,
        model=profile.model,
        timeout_s=profile.timeout_s,
        system_prompt=(
            "Ban la tro ly phap ly Viet Nam. "
            "Chi duoc tra loi dua tren EVIDENCE_CONTEXT. "
            "Khong duoc them ghi chu, luu y, hoac lap lai doan van. "
            "Neu khong du thong tin, phai noi ro trong muc CHUA_DU_CAN_CU."
        ),
    )


def _build_query_rewrite_service(profile_id: str | None, query_mode: str) -> QueryRewriteService:
    normalized_mode = _normalize_query_mode(query_mode)
    profile = get_model_profile(profile_id)
    enabled = normalized_mode == "rewrite_combine" and profile.supports_rewrite
    rewriter = None
    if enabled:
        from src.legal_query_rewriter import LegalQueryRewriter

        rewriter = LegalQueryRewriter(
            provider_type=profile.provider_type,
            base_url=profile.base_url,
            model_name=profile.model,
            timeout_s=profile.timeout_s,
        )
    return QueryRewriteService(
        enabled=enabled,
        force=enabled,
        rewriter=rewriter,
    )


def _save_shared_chat_history(
    *,
    request: ChatRequest,
    answer: str,
    sources: List[Dict[str, Any]],
    payload: Dict[str, Any],
) -> None:
    session_id = request.session_id
    user_id = request.user_id or "default"
    user_message_id = str(uuid.uuid4())
    assistant_message_id = str(uuid.uuid4())

    try:
        save_chat_session(
            session_id=session_id,
            user_id=user_id,
            title=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        save_message(
            message_id=user_message_id,
            session_id=session_id,
            role="user",
            content=request.message,
        )
        save_message(
            message_id=assistant_message_id,
            session_id=session_id,
            role="assistant",
            content=answer,
        )
        if sources:
            save_message_sources(assistant_message_id, sources)
    except Exception as exc:
        logger.error("Shared chat database save failed: %s", exc)

    if session_id not in _chat_history:
        _chat_history[session_id] = []
    _chat_history[session_id].append(
        {
            "id": user_message_id,
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat(),
        }
    )
    _chat_history[session_id].append(
        {
            "id": assistant_message_id,
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "search_time": payload.get("search_time"),
            "generation_time": payload.get("generation_time"),
            "total_time": payload.get("total_time"),
            "rerank_time": payload.get("rerank_time"),
            "reranker_used": payload.get("reranker_used"),
            "model_profile_id": payload.get("model_profile_id"),
            "query_mode": payload.get("query_mode"),
            "rewrite_used": payload.get("rewrite_used"),
            "rewrite_status": payload.get("rewrite_status"),
            "rewrite_fallback_reason": payload.get("rewrite_fallback_reason"),
            "rewrite_strategy": payload.get("rewrite_strategy"),
            "rewrite_quality_passed": payload.get("rewrite_quality_passed"),
            "rewritten_queries": payload.get("rewritten_queries", []),
            "retrieval_query_set": payload.get("retrieval_query_set", []),
            "retrieval_query_plan": payload.get("retrieval_query_plan", []),
            "rewrite_context": payload.get("rewrite_context", {}),
            "rewrite_debug": payload.get("rewrite_debug", {}),
            "per_subquery_hits": payload.get("per_subquery_hits", []),
            "retrieval_union_count": payload.get("retrieval_union_count", 0),
            "fallback_used": payload.get("fallback_used", False),
            "fallback_error": payload.get("fallback_error", ""),
            "collection": payload.get("collection"),
        }
    )


def _run_shared_legal_search(request: SearchRequest) -> SearchResponse:
    query_mode = _normalize_query_mode(request.query_mode)
    settings = _backend_runtime_settings(
        query_rewriter_enabled=(query_mode == "rewrite_combine"),
        query_rewriter_force=(query_mode == "rewrite_combine"),
    )
    settings.apply_env()
    pipeline = get_legal_pipeline(settings)
    contract = validate_qdrant_contract(settings, int(pipeline.embedding_model.embedding_dim))
    query_rewrite_service = _build_query_rewrite_service(request.model_profile_id, query_mode)
    result = pipeline.search(
        question=request.query,
        top_k=request.top_k,
        filter_conditions=None,
        use_reranker=bool(request.use_reranker if request.use_reranker is not None else _app_settings.get("reranker_enabled", True)),
        include_inactive=not request.require_active_only,
        use_graph_context=False,
        graph_hops=1,
        graph_expand_docs=20,
        query_rewrite_service=query_rewrite_service,
    )
    payload = build_search_payload(
        result,
        mode=request.search_mode,
        collection=contract.collection_name,
        model_profile_id=get_model_profile(request.model_profile_id).id,
        query_mode=query_mode,
    )
    return SearchResponse(**payload)


def _run_shared_legal_chat(request: ChatRequest) -> ChatResponse:
    query_mode = _normalize_query_mode(request.query_mode)
    settings = _backend_runtime_settings(
        query_rewriter_enabled=(query_mode == "rewrite_combine"),
        query_rewriter_force=(query_mode == "rewrite_combine"),
    )
    settings.apply_env()
    pipeline = get_legal_pipeline(settings)
    contract = validate_qdrant_contract(settings, int(pipeline.embedding_model.embedding_dim))
    query_rewrite_service = _build_query_rewrite_service(request.model_profile_id, query_mode)
    llm_client = _build_remote_llm(request.model_profile_id)

    fallback_used = False
    fallback_error = ""
    try:
        result = pipeline.query(
            question=request.message,
            top_k=request.top_k,
            filter_conditions=None,
            use_reranker=bool(request.use_reranker if request.use_reranker is not None else _app_settings.get("reranker_enabled", True)),
            include_inactive=not request.require_active_only,
            use_graph_context=bool(request.use_graph),
            graph_hops=1,
            graph_expand_docs=20,
            query_rewrite_service=query_rewrite_service,
            llm_client=llm_client,
        )
        answer = str(result.answer or "").strip()
    except Exception as exc:
        fallback_used = True
        fallback_error = str(exc)
        search_result = pipeline.search(
            question=request.message,
            top_k=request.top_k,
            filter_conditions=None,
            use_reranker=bool(request.use_reranker if request.use_reranker is not None else _app_settings.get("reranker_enabled", True)),
            include_inactive=not request.require_active_only,
            use_graph_context=bool(request.use_graph),
            graph_hops=1,
            graph_expand_docs=20,
            query_rewrite_service=query_rewrite_service,
        )
        result = LegalAnswer(
            answer="",
            citations=list(search_result.citations or []),
            intent=str(search_result.intent),
            retrieval_time=float(search_result.retrieval_time),
            generation_time=0.0,
            rerank_time=float(search_result.rerank_time),
            used_reranker=bool(search_result.used_reranker),
            graph_expansion_used=bool(search_result.graph_expansion_used),
            graph_mode=str(search_result.graph_mode),
            rewrite_time=float(search_result.rewrite_time),
            rewrite_used=bool(search_result.rewrite_used),
            rewrite_status=str(search_result.rewrite_status),
            rewrite_fallback_reason=str(search_result.rewrite_fallback_reason),
            rewrite_strategy=str(search_result.rewrite_strategy),
            rewrite_quality_passed=bool(search_result.rewrite_quality_passed),
            rewritten_queries=list(search_result.rewritten_queries),
            rewrite_mode=str(search_result.rewrite_mode),
            rewrite_original_query=str(search_result.rewrite_original_query),
            rewrite_queries_compact=list(search_result.rewrite_queries_compact),
            retrieval_query_set=list(search_result.retrieval_query_set),
            retrieval_query_plan=list(search_result.retrieval_query_plan),
            query_facets=dict(search_result.query_facets),
            rewrite_debug=dict(search_result.rewrite_debug),
            missing_numbers=list(search_result.missing_numbers),
            per_subquery_hits=list(search_result.per_subquery_hits),
            retrieval_union_count=int(search_result.retrieval_union_count),
        )
        answer = ""
    if not answer or answer.lower().startswith("loi khi goi llm") or answer.lower().startswith("llm khong kha dung"):
        fallback_used = True
        fallback_error = fallback_error or answer or "answer_generation_failed"
        answer = (
            "LLM unavailable or failed. Returning retrieval-only sources. "
            "Please inspect citations below."
        )

    payload = build_chat_payload(
        result,
        answer=answer,
        collection=contract.collection_name,
        model_profile_id=get_model_profile(request.model_profile_id).id,
        query_mode=query_mode,
        fallback_used=fallback_used,
        fallback_error=fallback_error,
    )
    _save_shared_chat_history(
        request=request,
        answer=answer,
        sources=payload["sources"],
        payload=payload,
    )
    return ChatResponse(**payload)


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        
        if Config.QDRANT_USE_CLOUD and Config.QDRANT_CLOUD_URL:
            _qdrant_client = QdrantClient(
                url=Config.QDRANT_CLOUD_URL,
                api_key=Config.QDRANT_API_KEY
            )
            logger.info(f"[QDRANT] Connected to Qdrant Cloud")
        else:
            _qdrant_client = QdrantClient(
                host=Config.QDRANT_HOST,
                port=Config.QDRANT_PORT
            )
            logger.info(f"[QDRANT] Connected to {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
    return _qdrant_client


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[EMBED] Loading {Config.EMBEDDING_MODEL} on {device}...")
        
        _embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)
        
        if device == "cuda":
            _embedding_model.half()
            import torch
            logger.info(f"[EMBED] GPU: {torch.cuda.get_device_name(0)}, FP16 enabled")
    
    return _embedding_model


def get_bm25_encoder():
    global _bm25_encoder
    if _bm25_encoder is None:
        if Config.BM25_CACHE_PATH.exists():
            with open(Config.BM25_CACHE_PATH, 'r', encoding='utf-8') as f:
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
            
            _bm25_encoder = SimpleBM25(data)
            logger.info(f"[BM25] Loaded with {len(data['vocab'])} vocab")
        else:
            logger.warning("[BM25] Cache not found, sparse search disabled")
    
    return _bm25_encoder


def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import requests
        
        class OllamaClient:
            def __init__(self, model, host):
                self.model = model
                self.host = host.rstrip("/")
            
            def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.1) -> str:
                try:
                    response = requests.post(
                        f"{self.host}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "num_predict": max_tokens,
                                "temperature": temperature,
                                "num_ctx": 32768
                            }
                        },
                        timeout=300
                    )
                    response.raise_for_status()
                    return response.json().get("response", "").strip()
                except Exception as e:
                    return f"❌ Lỗi LLM: {str(e)}"
            
            def chat_generate(self, model: str, prompt: str, images: List[str] = None, temperature: float = 0.1) -> str:
                try:
                    payload = {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                                "images": images if images else []
                            }
                        ],
                        "stream": False,
                        "format": "json",
                        "temperature": temperature
                    }
                    response = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
                    response.raise_for_status()
                    return response.json().get("message", {}).get("content", "").strip()
                except Exception as e:
                    return f'{{"error": "{str(e)}" }}'
        
        _ollama_client = OllamaClient(Config.OLLAMA_MODEL, Config.OLLAMA_URL)
        logger.info(f"[LLM] Initialized Ollama HTTP API: {Config.OLLAMA_MODEL} at {Config.OLLAMA_URL}")
    
    return _ollama_client


# ============================================================================
# Search Functions
# ============================================================================

def search_legal_collection(
    query: str,
    top_k: int = 10
) -> List[Dict]:
    """Search trong Kho Luật chính (read-only)"""
    from qdrant_client.http import models
    
    client = get_qdrant_client()
    embed_model = get_embedding_model()
    bm25 = get_bm25_encoder()
    
    # Generate query vectors
    query_vector = embed_model.encode(query).tolist()
    
    if bm25:
        sparse_indices, sparse_values = bm25.encode(query)
        
        results = client.query_points(
            collection_name=Config.LEGAL_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=top_k * 2
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    ),
                    using="sparse",
                    limit=top_k * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        )
    else:
        results = client.query_points(
            collection_name=Config.LEGAL_COLLECTION,
            query=query_vector,
            using="dense",
            limit=top_k,
            with_payload=True
        )
    
    documents = []
    for point in (results.points if hasattr(results, 'points') else results):
        doc = {
            "content": point.payload.get("content", ""),
            "score": point.score,
            "so_hieu": point.payload.get("so_hieu", ""),
            "dieu": point.payload.get("dieu", ""),
            "source_type": "legal",  # Đánh dấu nguồn từ Kho Luật
            "source_label": "Văn bản pháp luật"
        }
        documents.append(doc)
    
    return documents


def search_user_collection(
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    top_k: int = 5
) -> List[Dict]:
    """Search trong Kho Cá nhân (với filter user_id)"""
    user_store = get_user_store()
    
    results = user_store.search(
        query=query,
        user_id=user_id,
        session_id=session_id,
        top_k=top_k,
        use_hybrid=True
    )
    
    # Add source labels
    for doc in results:
        doc["source_type"] = "user"  # Đánh dấu nguồn từ User Upload
        doc["source_label"] = f"📄 {doc.get('filename', 'Tài liệu người dùng')}"
    
    return results


def hybrid_search(
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    top_k: int = 10,
    search_mode: str = "hybrid",
    use_reranker: Optional[bool] = None  # None = use global setting
) -> tuple:
    """
    Tìm kiếm kết hợp từ cả 2 kho với optional reranking
    
    search_mode:
    - "legal": Chỉ tìm trong Kho Luật
    - "user": Chỉ tìm trong Kho Cá nhân
    - "hybrid": Tìm cả 2 kho và merge kết quả
    
    Returns:
    - (results, rerank_time, reranker_used)
    """
    legal_results = []
    user_results = []
    rerank_time = 0.0
    reranker_used = False
    
    if search_mode in ["legal", "hybrid"]:
        legal_results = search_legal_collection(query, top_k=top_k * 2)  # Fetch more for reranking
    
    if search_mode in ["user", "hybrid"]:
        user_results = search_user_collection(
            query=query,
            user_id=user_id,
            session_id=session_id,
            top_k=min(5, top_k)  # Limit user results
        )
    
    # Merge all results
    all_results = legal_results + user_results
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Determine if reranker should be used
    should_rerank = use_reranker if use_reranker is not None else _app_settings.get("reranker_enabled", False)
    
    if should_rerank and len(all_results) > 0:
        try:
            reranker = get_reranker()
            if reranker.load():
                rerank_start = time.time()
                
                # Prepare documents for reranking
                docs_to_rerank = [r.get("content", "") for r in all_results]
                
                # Rerank with alpha blending
                alpha = _app_settings.get("reranker_alpha", 0.5)
                reranked, _ = reranker.rerank(
                    query=query,
                    results=all_results,
                    top_k=top_k,
                    alpha=alpha
                )
                
                # Rebuild results from RerankedResult objects
                reranked_results = []
                for i, rr in enumerate(reranked):
                    # Find original result by content matching
                    original = next((r for r in all_results if r.get("content", "") == rr.content), {})
                    
                    # Create new result dict with rerank info
                    new_result = original.copy()
                    new_result["original_score"] = rr.original_score
                    new_result["score"] = rr.final_score
                    new_result["rerank_score"] = rr.rerank_score
                    new_result["rerank_rank"] = i + 1
                    
                    reranked_results.append(new_result)
                
                all_results = reranked_results
                reranker_used = True
                rerank_time = time.time() - rerank_start
                
                logger.info(f"[RERANKER] Reranked {len(all_results)} docs in {rerank_time:.3f}s")
            else:
                logger.warning("[RERANKER] Failed to load, using original ranking")
        except Exception as e:
            logger.error(f"[RERANKER] Error: {e}, using original ranking")
    
    return all_results[:top_k], rerank_time, reranker_used


# ============================================================================
# Prompt Building
# ============================================================================
def build_legal_prompt(query: str, contexts: List[Dict], top_k: int = None) -> str:
    if top_k is None:
        top_k = Config.TOP_CONTEXTS_FOR_LLM
    
    context_text = ""
    for i, ctx in enumerate(contexts[:top_k], 1):
        source_type = ctx.get("source_type", "legal")
        # Lấy thêm metadata quan trọng cho việc so sánh
        ngay_bh = ctx.get("ngay_ban_hanh", "Không rõ")
        tinh_trang = ctx.get("tinh_trang", "Chưa xác định")
        
        if source_type == "legal":
            so_hieu = ctx.get("so_hieu", "")
            dieu = ctx.get("dieu", "")
            header = f"[📘 LUẬT] {dieu} - {so_hieu} (Ban hành: {ngay_bh} | Tình trạng: {tinh_trang})"
        else:
            filename = ctx.get("filename", "Tài liệu")
            header = f"[📄 TÀI LIỆU] {filename} (Ngày: {ngay_bh})"
        
        content = ctx.get("content", "")
        context_text += f"\n---\n**NGUỒN {i}** {header}:\n{content}\n"

    prompt = f"""Bạn là Trợ lý Pháp lý AI chuyên nghiệp tại Việt Nam. Nhiệm vụ của bạn là phân tích thông tin từ phần [TÀI LIỆU CUNG CẤP] để trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn và có căn cứ hiện hành.

<nguyen_tac_cot_loi>
1. TRUNG THÀNH TUYỆT ĐỐI VỚI DỮ LIỆU: 
   - CHỈ trả lời dựa trên thông tin có trong [TÀI LIỆU CUNG CẤP]. 
   - Nếu tài liệu không chứa câu trả lời, BẮT BUỘC nói: "Dựa trên các tài liệu được cung cấp, tôi không tìm thấy thông tin để trả lời câu hỏi này." TUYỆT ĐỐI không tự bịa ra luật.

2. QUY TẮC ƯU TIÊN HIỆU LỰC & THỜI GIAN (BẮT BUỘC TUÂN THỦ):
   - BƯỚC 1 (Ưu tiên tuyệt đối): Áp dụng quy định từ văn bản có ghi "Tình trạng: Còn hiệu lực".
   - BƯỚC 2 (Ép buộc so sánh năm): Nếu các văn bản đều có tình trạng "Chưa xác định", bạn PHẢI tự động trích xuất năm ban hành từ Số hiệu văn bản (Ví dụ: '91/2015/QH13' là năm 2015, '33/2005/QH11' là năm 2005). BẠN PHẢI SO SÁNH và CHỈ SỬ DỤNG văn bản có năm GẦN HIỆN TẠI NHẤT. 
   - BƯỚC 3 (Loại bỏ): LẬP TỨC BỎ QUA hoàn toàn các văn bản có năm cũ hơn. TUYỆT ĐỐI KHÔNG đưa văn bản cũ vào phần [CĂN CỨ PHÁP LÝ] hay [NỘI DUNG CHI TIẾT].
   - BƯỚC 4 (Thứ bậc): Nếu cùng năm, ưu tiên: Luật/Bộ luật > Nghị định > Thông tư.

3. KHÔNG LAN MAN LỊCH SỬ:
   - Chỉ trình bày duy nhất quy định của văn bản mới nhất đã được lọc qua Quy tắc 2. Không giải thích lý do loại bỏ văn bản cũ.
</nguyen_tac_cot_loi>

<cau_truc_tra_loi>
- [KẾT LUẬN]: Trả lời trực diện vào trọng tâm câu hỏi.
- [CĂN CỨ PHÁP LÝ]: Liệt kê chính xác Điều, Khoản, và Số hiệu văn bản MỚI NHẤT.
- [NỘI DUNG CHI TIẾT]: Phân tích ngắn gọn quy định (Sử dụng Bullet point hoặc Bảng).
- [LƯU Ý]: Thêm dòng chữ: "*Lưu ý: Tư vấn này được tạo bởi AI dựa trên tài liệu tham khảo, chỉ mang tính chất hỗ trợ thông tin.*"
</cau_truc_tra_loi>

---
[TÀI LIỆU CUNG CẤP]
{context_text}
---

[CÂU HỎI CỦA NGƯỜI DÙNG]: {query}

[TRẢ LỜI CỦA TRỢ LÝ]:"""
    
    return prompt

# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan - initialize resources on startup"""
    logger.info("🚀 Starting Legal RAG API...")
    
    # Pre-load resources
    get_qdrant_client()
    get_embedding_model()
    get_bm25_encoder()
    get_ollama_client()
    get_user_store()
    
    logger.info("✅ All resources loaded")
    yield
    
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="Legal RAG API",
    description="Vietnamese Legal RAG Chatbot API with Dual-Store Architecture",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Legal RAG API",
        "version": "1.0.0"
    }


@app.get("/api/status")
async def get_status():
    """Get system status"""
    try:
        if Config.BACKEND_USE_SHARED_LEGAL_PIPELINE:
            settings = _backend_runtime_settings(query_rewriter_enabled=False, query_rewriter_force=False)
            settings.apply_env()
            pipeline = await asyncio.to_thread(get_legal_pipeline, settings)
            contract = await asyncio.to_thread(
                validate_qdrant_contract,
                settings,
                int(pipeline.embedding_model.embedding_dim),
            )
            legal_count = int(pipeline.qdrant_store.get_stats().get("points_count", 0) or 0)
            client = get_qdrant_client()
            try:
                user_info = client.get_collection(Config.USER_COLLECTION)
                user_count = user_info.points_count
            except Exception:
                user_count = 0
            return {
                "status": "ok",
                "backend_mode": "shared_legal_pipeline",
                "shared_pipeline_enabled": True,
                "legal_collection": contract.collection_name,
                "legal_documents": legal_count,
                "user_collection": Config.USER_COLLECTION,
                "user_documents": user_count,
                "embedding_model": settings.embedding_model,
                "llm_model": Config.OLLAMA_MODEL,
                "available_model_profiles": [profile.to_public_dict() for profile in list_model_profiles()],
                "collection_resolved": contract.collection_name,
                "collection_status": contract.collection_status,
            }

        client = get_qdrant_client()
        legal_info = client.get_collection(Config.LEGAL_COLLECTION)
        try:
            user_info = client.get_collection(Config.USER_COLLECTION)
            user_count = user_info.points_count
        except Exception:
            user_count = 0
        return {
            "status": "ok",
            "backend_mode": "legacy",
            "shared_pipeline_enabled": False,
            "legal_collection": Config.LEGAL_COLLECTION,
            "legal_documents": legal_info.points_count,
            "user_collection": Config.USER_COLLECTION,
            "user_documents": user_count,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.OLLAMA_MODEL,
            "available_model_profiles": [profile.to_public_dict() for profile in list_model_profiles()],
            "collection_resolved": Config.LEGAL_COLLECTION,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings/model-profiles")
async def get_model_profiles():
    return {
        "profiles": [profile.to_public_dict() for profile in list_model_profiles()],
        "default_profile_id": get_default_model_profile_id(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the legal AI assistant"""
    if _shared_legal_enabled(request.search_mode):
        try:
            return await asyncio.to_thread(_run_shared_legal_chat, request)
        except Exception as exc:
            logger.error("Shared legal chat error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    start_time = time.time()
    
    try:
        # Step 1: Search with optional reranking
        search_start = time.time()
        contexts, rerank_time, reranker_used = hybrid_search(
            query=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            top_k=request.top_k,
            search_mode=request.search_mode,
            use_reranker=request.use_reranker
        )
        search_time = time.time() - search_start - rerank_time  # Exclude rerank time
        
        if not contexts:
            # Build helpful error message based on search mode
            try:
                user_store = get_user_store()
                user_docs = user_store.get_user_documents(request.user_id, request.session_id)
                doc_count = len(user_docs)
            except:
                doc_count = 0
            
            if request.search_mode == "user":
                if doc_count == 0:
                    error_msg = "❌ Bạn chưa tải lên tài liệu nào. Vui lòng tải lên tài liệu trước khi tìm kiếm trong 'Tài liệu của tôi'."
                else:
                    error_msg = f"❌ Không tìm thấy nội dung phù hợp trong {doc_count} tài liệu của bạn.\n\n💡 Gợi ý:\n• Chuyển sang chế độ 'Kết hợp' (tìm cả tài liệu của bạn và văn bản luật)\n• Chuyển sang 'Văn bản pháp luật' (tìm trong kho luật)\n• Đặt câu hỏi khác liên quan đến nội dung tài liệu đã tải lên"
            elif request.search_mode == "legal":
                error_msg = "❌ Không tìm thấy văn bản pháp luật phù hợp.\n\n💡 Gợi ý:\n• Thử câu hỏi khác với từ khóa rõ ràng hơn\n• Sử dụng thuật ngữ pháp lý chính xác\n• Tải lên tài liệu riêng và chuyển sang chế độ 'Tài liệu của tôi'"
            else:  # hybrid
                error_msg = f"❌ Không tìm thấy tài liệu phù hợp (đã tìm trong {doc_count} tài liệu riêng và kho luật).\n\n💡 Gợi ý:\n• Đặt câu hỏi rõ ràng hơn với từ khóa cụ thể\n• Kiểm tra chính tả và ngữ pháp\n• Tải lên thêm tài liệu liên quan nếu cần"
            
            logger.warning(f"[CHAT] No results found. Mode={request.search_mode}, User docs={doc_count}, Total contexts={len(contexts)}")
            
            return ChatResponse(
                answer=error_msg,
                sources=[],
                search_time=search_time,
                generation_time=0,
                total_time=time.time() - start_time,
                rerank_time=rerank_time,
                reranker_used=reranker_used
            )
        
        # Step 2: Generate answer (use TOP_CONTEXTS_FOR_LLM)
        gen_start = time.time()
        llm = get_ollama_client()
        contexts_for_llm = contexts[:Config.TOP_CONTEXTS_FOR_LLM]
        prompt = build_legal_prompt(request.message, contexts_for_llm, top_k=Config.TOP_CONTEXTS_FOR_LLM)
        answer = llm.generate(prompt, max_tokens=4096, temperature=0.1)
        gen_time = time.time() - gen_start
        
        # Step 3: Format sources for frontend (same as contexts used)
        sources = []
        seen = set()
        
        for ctx in contexts_for_llm:  # Same contexts used in LLM
            source_type = ctx.get("source_type", "legal")
            
            if source_type == "legal":
                so_hieu = ctx.get("so_hieu", "")
                dieu = ctx.get("dieu", "")
                key = (so_hieu, dieu)
                
                if key not in seen and (so_hieu or dieu):
                    seen.add(key)
                    sources.append({
                        "type": "legal",
                        "label": f"{dieu}" if dieu else so_hieu,
                        "detail": so_hieu if dieu else "",
                        "score": ctx.get("score", 0),
                        "content_preview": ctx.get("content", ""),  # Full content for preview
                        "content": ctx.get("content", "")  # Full content for modal
                    })
            else:
                filename = ctx.get("filename", "")
                page = ctx.get("page", "")
                key = (filename, page)
                
                if key not in seen and filename:
                    seen.add(key)
                    sources.append({
                        "type": "user",
                        "label": filename,
                        "detail": f"Trang {page}" if page else "",
                        "score": ctx.get("score", 0),
                        "content_preview": ctx.get("content", ""),  # Full content for preview
                        "content": ctx.get("content", "")  # Full content for modal
                    })
        
        # Step 4: Quality Monitoring (automatic evaluation with SAME contexts)
        quality_metrics = None
        try:
            monitor = get_quality_monitor()
            # Use EXACT SAME contexts that were sent to LLM
            context_texts = [ctx.get("content", "") for ctx in contexts_for_llm]
            quality_metrics = monitor.evaluate_answer(
                query=request.message,
                answer=answer,
                contexts=context_texts,
                prompt=prompt,
                top_k=Config.TOP_CONTEXTS_FOR_LLM  # Match contexts_for_llm
            )
            logger.info(f"✅ Quality metrics: Grade={quality_metrics.grade}, Score={quality_metrics.overall_score:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ Quality monitoring failed: {e}")
        
        # Step 5: Save to database
        session_id = request.session_id
        user_id = request.user_id or "default"
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        
        try:
            # Ensure session exists
            save_chat_session(
                session_id=session_id,
                user_id=user_id,
                title=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            # Save user message
            save_message(
                message_id=user_message_id,
                session_id=session_id,
                role="user",
                content=request.message
            )
            
            # Save assistant message
            save_message(
                message_id=assistant_message_id,
                session_id=session_id,
                role="assistant",
                content=answer
            )
            
            # Save sources
            if sources:
                save_message_sources(assistant_message_id, sources)
            
            # Save quality metrics if available
            if quality_metrics:
                save_answer_metrics(
                    message_id=assistant_message_id,
                    overall_score=quality_metrics.overall_score,
                    grade=quality_metrics.grade,
                    feedback=quality_metrics.feedback,
                    bert_score=quality_metrics.bert_score,
                    hallucination_score=quality_metrics.hallucination_score,
                    factuality_score=quality_metrics.factuality_score,
                    context_relevance=quality_metrics.context_relevance,
                    query=request.message,
                    answer=answer,
                    top_k_contexts=Config.TOP_CONTEXTS_FOR_LLM,  # Actual contexts used
                    contexts_used=len(contexts_for_llm),  # Actual contexts count
                    reranker_used=reranker_used
                )
                logger.info(f"💾 Metrics saved to database")
            
        except Exception as e:
            logger.error(f"❌ Database save failed: {e}")
        
        # Save to in-memory history (for backward compatibility)
        if session_id not in _chat_history:
            _chat_history[session_id] = []
        
        _chat_history[session_id].append({
            "id": user_message_id,
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        _chat_history[session_id].append({
            "id": assistant_message_id,
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "quality_grade": quality_metrics.grade if quality_metrics else None,
            "quality_score": quality_metrics.overall_score if quality_metrics else None
        })
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            search_time=search_time,
            generation_time=gen_time,
            total_time=time.time() - start_time,
            rerank_time=rerank_time,
            reranker_used=reranker_used,
            # Quality metrics
            quality_grade=quality_metrics.grade if quality_metrics else None,
            quality_score=quality_metrics.overall_score if quality_metrics else None,
            bert_score=quality_metrics.bert_score if quality_metrics else None,
            hallucination_score=quality_metrics.hallucination_score if quality_metrics else None,
            factuality_score=quality_metrics.factuality_score if quality_metrics else None,
            context_relevance=quality_metrics.context_relevance if quality_metrics else None,
            quality_feedback=quality_metrics.feedback if quality_metrics else None
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search documents"""
    if _shared_legal_enabled(request.search_mode):
        try:
            return await asyncio.to_thread(_run_shared_legal_search, request)
        except Exception as exc:
            logger.error("Shared legal search error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    start_time = time.time()
    
    try:
        results, rerank_time, reranker_used = hybrid_search(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            top_k=request.top_k,
            search_mode=request.search_mode,
            use_reranker=request.use_reranker
        )
        
        return SearchResponse(
            results=results,
            search_time=time.time() - start_time - rerank_time,
            mode=request.search_mode,
            rerank_time=rerank_time,
            reranker_used=reranker_used
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(default="default_user"),
    session_id: str = Form(default=None)
):
    """Upload user document"""
    
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Validate file type
    allowed_types = ['.pdf', '.docx', '.doc', '.txt']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Save file temporarily
    temp_path = Config.UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    
    try:
        logger.info(f"[UPLOAD] Receiving file: {file.filename} ({file_ext})")
        
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"[UPLOAD] Saved to: {temp_path}, size: {temp_path.stat().st_size} bytes")
        
        # Process file
        user_store = get_user_store()
        success, doc_id, chunk_count = await user_store.upload_file(
            file_path=str(temp_path),
            user_id=user_id,
            session_id=session_id,
            filename=file.filename
        )
        
        logger.info(f"[UPLOAD] Processing result: success={success}, chunks={chunk_count}")
        
        if success:
            return UploadResponse(
                success=True,
                doc_id=doc_id,
                filename=file.filename,
                chunk_count=chunk_count,
                message=f"Đã tải lên và xử lý {chunk_count} đoạn văn bản"
            )
        else:
            logger.error(f"[UPLOAD] Processing failed for {file.filename}")
            
            # Check if it's a .doc file that couldn't be processed
            if file_ext == '.doc':
                raise HTTPException(
                    status_code=400, 
                    detail="Không thể xử lý file .doc (Word 97-2003 cũ). Vui lòng chuyển đổi sang .docx: Mở file trong Word → Save As → chọn .docx"
                )
            
            raise HTTPException(status_code=500, detail=f"Không thể xử lý file {file.filename}")
            
    except Exception as e:
        logger.error(f"[UPLOAD] Upload error for {file.filename}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/documents")
async def get_documents(
    user_id: str = "default_user",
    session_id: Optional[str] = None
):
    """Get user's uploaded documents"""
    try:
        user_store = get_user_store()
        documents = user_store.get_user_documents(user_id, session_id)
        
        return {
            "documents": [doc.to_dict() for doc in documents]
        }
    except Exception as e:
        logger.error(f"Get documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user_id: str = "default_user"
):
    """Delete user document"""
    try:
        user_store = get_user_store()
        success = user_store.delete_document(doc_id, user_id)
        
        if success:
            return {"message": f"Document {doc_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history(session_id: str):
    """Get chat history for session"""
    if session_id in _chat_history:
        return {"messages": _chat_history[session_id]}
    return {"messages": []}


@app.get("/api/sessions")
async def get_sessions(user_id: str = "default_user"):
    """Get list of chat sessions"""
    sessions = []
    
    for session_id, messages in _chat_history.items():
        if messages:
            # Get first user message as title
            first_msg = next(
                (m for m in messages if m.get("role") == "user"),
                None
            )
            if first_msg:
                sessions.append({
                    "id": session_id,
                    "title": first_msg["content"][:50] + "..." if len(first_msg["content"]) > 50 else first_msg["content"],
                    "timestamp": first_msg.get("timestamp", ""),
                    "message_count": len(messages)
                })
    
    # Sort by timestamp descending
    sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return {"sessions": sessions}


# ============================================================================
# Reranker & Scoring Endpoints
# ============================================================================

@app.get("/api/settings/reranker", response_model=RerankerSettingsResponse)
async def get_reranker_settings():
    """Get current reranker settings"""
    reranker = get_reranker()
    
    return RerankerSettingsResponse(
        enabled=_app_settings.get("reranker_enabled", False),
        alpha=_app_settings.get("reranker_alpha", 0.5),
        model_name=reranker.MODEL_NAME,
        model_loaded=reranker.model is not None
    )


@app.put("/api/settings/reranker")
async def update_reranker_settings(
    enabled: bool = None,
    alpha: float = None
):
    """Update reranker settings"""
    global _app_settings
    
    if enabled is not None:
        _app_settings["reranker_enabled"] = enabled
        logger.info(f"[SETTINGS] Reranker enabled: {enabled}")
    
    if alpha is not None:
        if 0.0 <= alpha <= 1.0:
            _app_settings["reranker_alpha"] = alpha
            logger.info(f"[SETTINGS] Reranker alpha: {alpha}")
        else:
            raise HTTPException(status_code=400, detail="Alpha must be between 0.0 and 1.0")
    
    # Preload model if enabling
    if enabled:
        try:
            reranker = get_reranker()
            reranker.load()
        except Exception as e:
            logger.warning(f"[RERANKER] Preload failed: {e}")
    
    return {
        "enabled": _app_settings["reranker_enabled"],
        "alpha": _app_settings["reranker_alpha"],
        "message": "Settings updated successfully"
    }


@app.post("/api/score", response_model=ScoreResponse)
async def score_answer(request: ScoreRequest):
    """
    Score answer quality using QualityMonitor (BERTScore-based)
    
    [PRIMARY SCORER] Uses QualityMonitor for consistent evaluation across all features.
    This ensures chat, dashboard, and API scoring use the same methodology.
    
    Returns:
    - overall_score: Weighted combination of all metrics
    - grade: A/B/C/D/F letter grade
    - bert_score: Semantic similarity (BERTScore F1)
    - hallucination_score: Claim verification score
    - factuality_score: Fact accuracy score
    - context_relevance: Query-context relevance
    """
    try:
        # Use QualityMonitor as primary scorer
        monitor = get_quality_monitor()
        
        # Extract content strings from contexts
        context_texts = []
        for ctx in request.contexts[:request.top_k]:
            if isinstance(ctx, dict):
                content = ctx.get("content", "")
            else:
                content = str(ctx)
            if content:
                context_texts.append(content)
        
        if not context_texts:
            return ScoreResponse(
                overall_score=0.0,
                grade="F",
                feedback="Không có tài liệu nguồn để đánh giá.",
                details={}
            )
        
        # Build dummy prompt for evaluation
        prompt = f"Question: {request.query}\\n\\nAnswer: {request.answer}"
        
        # Evaluate with QualityMonitor
        metrics = monitor.evaluate_answer(
            query=request.query,
            answer=request.answer,
            contexts=context_texts,
            prompt=prompt,
            top_k=request.top_k
        )
        
        return ScoreResponse(
            overall_score=metrics.overall_score,
            grade=metrics.grade,
            feedback=metrics.feedback,
            details={
                "bert_score": metrics.bert_score,
                "hallucination_score": metrics.hallucination_score,
                "factuality_score": metrics.factuality_score,
                "context_relevance": metrics.context_relevance,
                **metrics.details
            }
        )
        
    except Exception as e:
        logger.error(f"Score error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/preload-reranker")
async def preload_reranker():
    """Preload reranker model to GPU"""
    try:
        reranker = get_reranker()
        success = reranker.load()
        
        if success:
            return {
                "status": "ok",
                "message": "Reranker model loaded successfully",
                "model": reranker.MODEL_NAME,
                "device": str(reranker.device)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load reranker model")
            
    except Exception as e:
        logger.error(f"Preload reranker error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/stats")
async def get_metrics_statistics(
    user_id: Optional[str] = None,
    days: int = 7
):
    """
    Get aggregated quality metrics statistics
    
    Args:
        user_id: Filter by user (optional)
        days: Number of days to look back (default: 7)
    
    Returns:
        Statistics: average scores, grade distribution, trends
    """
    try:
        stats = get_metrics_stats(user_id=user_id, days=days)
        
        return {
            "status": "ok",
            "user_id": user_id,
            "days": days,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Get metrics stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/history")
async def get_metrics_history(
    session_id: str
):
    """
    Get metrics history for a session
    
    Args:
        session_id: Session ID
    
    Returns:
        List of messages with quality metrics
    """
    try:
        messages = get_session_messages(session_id=session_id)
        
        # Format for frontend
        history = []
        for msg in messages:
            item = {
                "id": msg.get("id", ""),
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "timestamp": msg.get("created_at", ""),
                "sources": msg.get("sources", [])
            }
            
            # Add metrics if available
            if "metrics" in msg and msg["metrics"]:
                item["quality"] = {
                    "overall_score": msg["metrics"].get("overall_score"),
                    "grade": msg["metrics"].get("grade"),
                    "feedback": msg["metrics"].get("feedback"),
                    "bert_score": msg["metrics"].get("bert_score"),
                    "hallucination_score": msg["metrics"].get("hallucination_score"),
                    "factuality_score": msg["metrics"].get("factuality_score"),
                    "context_relevance": msg["metrics"].get("context_relevance")
                }
            
            history.append(item)
        
        return {
            "status": "ok",
            "session_id": session_id,
            "count": len(history),
            "messages": history
        }
        
    except Exception as e:
        logger.error(f"Get metrics history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/all")
async def get_all_metrics(
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    grade_filter: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """
    Get all quality metrics for dashboard
    
    Args:
        user_id: Filter by user (optional)
        limit: Max records to return (default: 100)
        offset: Pagination offset (default: 0)
        grade_filter: Filter by grade A/B/C/D/F (optional)
        sort_by: Field to sort by (default: created_at)
        sort_order: asc or desc (default: desc)
    
    Returns:
        List of all metrics with full details
    """
    try:
        from database import AnswerMetrics, Message, ChatSession
        
        with get_db() as db:
            # Build query
            query = db.query(AnswerMetrics, Message, ChatSession)\
                .join(Message, AnswerMetrics.message_id == Message.id)\
                .join(ChatSession, Message.session_id == ChatSession.id)
            
            # Apply filters
            if user_id:
                query = query.filter(ChatSession.user_id == user_id)
            
            if grade_filter and grade_filter in ['A', 'B', 'C', 'D', 'F']:
                query = query.filter(AnswerMetrics.grade == grade_filter)
            
            # Apply sorting
            if sort_by == "created_at":
                order_col = AnswerMetrics.created_at
            elif sort_by == "overall_score":
                order_col = AnswerMetrics.overall_score
            elif sort_by == "grade":
                order_col = AnswerMetrics.grade
            else:
                order_col = AnswerMetrics.created_at
            
            if sort_order == "desc":
                order_col = order_col.desc()
            else:
                order_col = order_col.asc()
            
            query = query.order_by(order_col)
            
            # Get total count before pagination
            total_count = query.count()
            
            # Apply pagination
            results = query.offset(offset).limit(limit).all()
            
            # Format results
            from database import MessageSource
            metrics_list = []
            for metric, message, session in results:
                # Get sources for this message
                sources = db.query(MessageSource)\
                    .filter(MessageSource.message_id == message.id)\
                    .order_by(MessageSource.rank)\
                    .all()
                
                item = {
                    "id": metric.id,
                    "message_id": metric.message_id,
                    "session_id": session.id,
                    "user_id": session.user_id,
                    "timestamp": metric.created_at.isoformat() if metric.created_at else None,
                    
                    # Quality metrics
                    "overall_score": metric.overall_score,
                    "grade": metric.grade,
                    "feedback": metric.feedback,
                    "bert_score": metric.bert_score,
                    "hallucination_score": metric.hallucination_score,
                    "factuality_score": metric.factuality_score,
                    "context_relevance": metric.context_relevance,
                    
                    # Content
                    "query": metric.query,
                    "answer": metric.answer,
                    
                    # Sources
                    "sources": [
                        {
                            "rank": s.rank,
                            "score": s.score,
                            "label": s.label,
                            "detail": s.detail,
                            "content": s.content_preview
                        }
                        for s in sources
                    ],
                    
                    # Metadata
                    "top_k_contexts": metric.top_k_contexts,
                    "contexts_used": metric.contexts_used,
                    "reranker_used": metric.reranker_used,
                    
                    # Timing from message
                    "search_time": message.search_time,
                    "generation_time": message.generation_time,
                    "rerank_time": message.rerank_time,
                    "total_time": message.total_time
                }
                
                metrics_list.append(item)
            
            return {
                "status": "ok",
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "count": len(metrics_list),
                "metrics": metrics_list
            }
        
    except Exception as e:
        logger.error(f"Get all metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/export")
async def export_all_metrics(
    format: str = "json",
    user_id: Optional[str] = None
):
    """
    Export all metrics to file
    
    Args:
        format: Export format (json, csv, txt)
        user_id: Filter by user (optional)
    
    Returns:
        File download
    """
    try:
        from database import AnswerMetrics, Message, ChatSession, MessageSource
        import io
        import csv
        
        with get_db() as db:
            # Get all metrics with related data
            query = db.query(AnswerMetrics, Message, ChatSession)\
                .join(Message, AnswerMetrics.message_id == Message.id)\
                .join(ChatSession, Message.session_id == ChatSession.id)
            
            if user_id:
                query = query.filter(ChatSession.user_id == user_id)
            
            results = query.order_by(AnswerMetrics.created_at.desc()).all()
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == "json":
                # JSON export
                data = {
                    "export_time": datetime.now().isoformat(),
                    "total_records": len(results),
                    "user_id": user_id,
                    "metrics": []
                }
                
                for metric, message, session in results:
                    # Get sources
                    sources = db.query(MessageSource)\
                        .filter(MessageSource.message_id == message.id)\
                        .all()
                    
                    item = {
                        "id": metric.id,
                        "timestamp": metric.created_at.isoformat() if metric.created_at else None,
                        "session_id": session.id,
                        "user_id": session.user_id,
                        "quality_metrics": {
                            "overall_score": metric.overall_score,
                            "grade": metric.grade,
                            "feedback": metric.feedback,
                            "bert_score": metric.bert_score,
                            "hallucination_score": metric.hallucination_score,
                            "factuality_score": metric.factuality_score,
                            "context_relevance": metric.context_relevance
                        },
                        "message_content": {
                            "query": metric.query,
                            "answer": metric.answer
                        },
                        "context_info": {
                            "top_k": metric.top_k_contexts,
                            "used": metric.contexts_used,
                            "reranker_used": metric.reranker_used
                        },
                        "timing": {
                            "search_time": message.search_time,
                            "generation_time": message.generation_time,
                            "rerank_time": message.rerank_time,
                            "total_time": message.total_time
                        },
                        "sources": [
                            {
                                "rank": s.rank,
                                "score": s.score,
                                "label": s.label,
                                "details": s.details
                            }
                            for s in sources
                        ]
                    }
                    
                    data["metrics"].append(item)
                
                content = json.dumps(data, ensure_ascii=False, indent=2)
                media_type = "application/json"
                filename = f"quality_metrics_export_{timestamp}.json"
                
            elif format == "csv":
                # CSV export
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Headers
                writer.writerow([
                    "ID", "Timestamp", "Session ID", "User ID",
                    "Overall Score", "Grade", "Feedback",
                    "BERTScore", "Hallucination Score", "Factuality Score", "Context Relevance",
                    "Query", "Answer",
                    "Top K", "Contexts Used", "Reranker Used",
                    "Search Time", "Generation Time", "Rerank Time", "Total Time"
                ])
                
                # Data rows
                for metric, message, session in results:
                    writer.writerow([
                        metric.id,
                        metric.created_at.isoformat() if metric.created_at else "",
                        session.id,
                        session.user_id,
                        metric.overall_score,
                        metric.grade,
                        metric.feedback,
                        metric.bert_score,
                        metric.hallucination_score,
                        metric.factuality_score,
                        metric.context_relevance,
                        (metric.query or "")[:100],  # Truncate for CSV
                        (metric.answer or "")[:200],  # Truncate for CSV
                        metric.top_k_contexts,
                        metric.contexts_used,
                        metric.reranker_used,
                        message.search_time,
                        message.generation_time,
                        message.rerank_time,
                        message.total_time
                    ])
                
                content = output.getvalue()
                media_type = "text/csv"
                filename = f"quality_metrics_export_{timestamp}.csv"
                
            elif format == "txt":
                # TXT export
                lines = []
                lines.append("=" * 80)
                lines.append("QUALITY METRICS EXPORT REPORT")
                lines.append("=" * 80)
                lines.append(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append(f"Total Records: {len(results)}")
                if user_id:
                    lines.append(f"User ID: {user_id}")
                lines.append("=" * 80)
                lines.append("")
                
                for idx, (metric, message, session) in enumerate(results, 1):
                    lines.append(f"\n{'=' * 80}")
                    lines.append(f"RECORD #{idx}")
                    lines.append(f"{'=' * 80}")
                    lines.append(f"ID: {metric.id}")
                    lines.append(f"Timestamp: {metric.created_at.strftime('%Y-%m-%d %H:%M:%S') if metric.created_at else 'N/A'}")
                    lines.append(f"Session: {session.id}")
                    lines.append(f"User: {session.user_id}")
                    lines.append("")
                    lines.append("QUALITY METRICS:")
                    lines.append(f"  Overall Score: {metric.overall_score:.2%} (Grade: {metric.grade})")
                    lines.append(f"  BERTScore: {metric.bert_score:.2%}" if metric.bert_score else "  BERTScore: N/A")
                    lines.append(f"  Hallucination Score: {metric.hallucination_score:.2%}" if metric.hallucination_score else "  Hallucination Score: N/A")
                    lines.append(f"  Factuality Score: {metric.factuality_score:.2%}" if metric.factuality_score else "  Factuality Score: N/A")
                    lines.append(f"  Context Relevance: {metric.context_relevance:.2%}" if metric.context_relevance else "  Context Relevance: N/A")
                    lines.append(f"  Feedback: {metric.feedback}")
                    lines.append("")
                    lines.append("QUERY:")
                    lines.append(f"  {metric.query}")
                    lines.append("")
                    lines.append("ANSWER:")
                    lines.append(f"  {metric.answer}")
                    lines.append("")
                    lines.append("METADATA:")
                    lines.append(f"  Top K: {metric.top_k_contexts}")
                    lines.append(f"  Contexts Used: {metric.contexts_used}")
                    lines.append(f"  Reranker Used: {'Yes' if metric.reranker_used else 'No'}")
                    lines.append("")
                    lines.append("TIMING:")
                    lines.append(f"  Search: {message.search_time:.3f}s" if message.search_time else "  Search: N/A")
                    lines.append(f"  Generation: {message.generation_time:.3f}s" if message.generation_time else "  Generation: N/A")
                    lines.append(f"  Rerank: {message.rerank_time:.3f}s" if message.rerank_time else "  Rerank: N/A")
                    lines.append(f"  Total: {message.total_time:.3f}s" if message.total_time else "  Total: N/A")
                
                lines.append(f"\n{'=' * 80}")
                lines.append("END OF REPORT")
                lines.append("=" * 80)
                
                content = "\n".join(lines)
                media_type = "text/plain"
                filename = f"quality_metrics_export_{timestamp}.txt"
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            
            # Return file download
            return StreamingResponse(
                io.BytesIO(content.encode('utf-8')),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        
    except Exception as e:
        logger.error(f"Export metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Legal OCR Endpoints
# ============================================================================

@app.post("/api/ocr/preview")
async def ocr_preview(file: UploadFile = File(...)):
    try:
        content = await file.read()
        import fitz
        import base64
        
        if file.filename.lower().endswith(".pdf"):
            doc = fitz.open(stream=content, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                b64 = base64.b64encode(img_bytes).decode('utf-8')
                doc.close()
                return {"success": True, "image_b64": b64, "type": "pdf"}
        else:
            b64 = base64.b64encode(content).decode('utf-8')
            return {"success": True, "image_b64": b64, "type": "image"}
            
        return {"success": False, "error": "Unsupported file format"}
    except Exception as e:
        logger.error(f"[OCR] Preview error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/ocr/extract")
async def ocr_extract(
    file: UploadFile = File(...),
    mode: str = Form("Fast"),
    prompt_json: str = Form("{}")
):
    try:
        content = await file.read()
        import fitz
        import base64
        import json
        llm = get_ollama_client()
        results = []
        
        is_pdf = file.filename.lower().endswith((".pdf"))
        
        if "fast" in mode.lower() and is_pdf:
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                text = page.get_text()
                full_prompt = f"Bạn phải tuân thủ chặt chẽ nội dung được cung cấp. Không được đoán hoặc thêm thông tin ngoài. Chỉ trích xuất dữ liệu từ văn bản sau:\n\nCấu trúc JSON: {prompt_json}\n\nNội dung tài liệu: {text}"
                res_str = llm.chat_generate(model="qwen3:8b", prompt=full_prompt)
                try:
                    res_json = json.loads(res_str)
                except:
                    res_json = {"raw": res_str}
                results.append({f"Trang {page.number + 1}": res_json})
            doc.close()
            
        else:
            # Accurate Mode
            imgs_b64 = []
            if is_pdf:
                doc = fitz.open(stream=content, filetype="pdf")
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    imgs_b64.append(base64.b64encode(img_data).decode('utf-8'))
                doc.close()
            else:
                imgs_b64 = [base64.b64encode(content).decode('utf-8')]
                
            for i, b64 in enumerate(imgs_b64):
                full_prompt = f"Bạn phải tuân thủ chặt chẽ nội dung trong ảnh. Không được đoán hoặc thêm thông tin ngoài. Chỉ trích xuất dữ liệu từ ảnh theo cấu trúc này:\n\n{prompt_json}"
                res_str = llm.chat_generate(model="qwen3-vl:8b", prompt=full_prompt, images=[b64])
                try:
                    res_json = json.loads(res_str)
                except:
                    res_json = {"raw": res_str}
                results.append({f"Trang {i+1}": res_json})
                
        return {"success": True, "results": results}
        
    except Exception as e:
        logger.error(f"[OCR] Extract error: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# Main Entry
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
