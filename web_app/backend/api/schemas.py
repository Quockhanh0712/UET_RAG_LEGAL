"""API schema models extracted from backend.main for modular routing."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    search_mode: str = "hybrid"
    document_scope: str = "current_session"
    top_k: int = 10
    use_reranker: Optional[bool] = None
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
    rerank_time: float = 0.0
    reranker_used: bool = False
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
    retrieval_debug: Dict[str, Any] = Field(default_factory=dict)
    per_subquery_hits: List[Dict] = Field(default_factory=list)
    retrieval_union_count: int = 0
    fallback_used: bool = False
    fallback_error: str = ""
    collection: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    session_id: Optional[str] = None
    search_mode: str = "hybrid"
    document_scope: str = "current_session"
    top_k: int = 10
    use_reranker: Optional[bool] = None
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
    retrieval_debug: Dict[str, Any] = Field(default_factory=dict)
    per_subquery_hits: List[Dict] = Field(default_factory=list)
    retrieval_union_count: int = 0
    fallback_used: bool = False
    fallback_error: str = ""
    collection: Optional[str] = None


class ScoreRequest(BaseModel):
    query: str
    answer: str
    contexts: List[Dict]
    top_k: int = 5


class ScoreResponse(BaseModel):
    overall_score: float
    grade: str
    feedback: str
    details: Dict


class RerankerSettingsResponse(BaseModel):
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
    job_id: Optional[str] = None
    ingest_mode: Optional[str] = None
    page_count: Optional[int] = None
    text_chars: Optional[int] = None
    parser_used: Optional[str] = None
    ocr_provider: Optional[str] = None
    error_message: Optional[str] = None


class UploadResponse(BaseModel):
    success: bool
    doc_id: str
    filename: str
    chunk_count: int
    message: str
    job_id: Optional[str] = None
    status: str = "queued"


class DocumentStatusResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int = 0
    job_id: Optional[str] = None
    ingest_mode: Optional[str] = None
    page_count: Optional[int] = None
    text_chars: Optional[int] = None
    parser_used: Optional[str] = None
    ocr_provider: Optional[str] = None
    error_message: Optional[str] = None
    job: Dict[str, Any] = Field(default_factory=dict)


class ChatHistoryItem(BaseModel):
    id: str
    title: str
    timestamp: str
    preview: str
