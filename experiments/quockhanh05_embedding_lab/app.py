from __future__ import annotations

import asyncio
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.quockhanh05_embedding_lab.config import (  # noqa: E402
    BM25_PATH,
    COLLECTION_NAME,
    DENSE_DIM,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    NORMALIZE_EMBEDDINGS,
    apply_lab_environment,
)
from src.embeddings import get_embedding_dimension, get_model  # noqa: E402
from src.legal_rag_pipeline import LegalRAGPipeline  # noqa: E402


STATIC_DIR = LAB_DIR / "static"

app = FastAPI(title="Quockhanh05 Embedding Lab", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    include_inactive: bool = False


class SafeAnswerClient:
    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.fallback_used = False
        self.fallback_error = ""

    def generate(self, question: str, structured_context: Dict[str, str]) -> str:
        try:
            return self.delegate.generate(question, structured_context)
        except Exception as exc:
            self.fallback_used = True
            self.fallback_error = str(exc)
            evidence = str(structured_context.get("evidence_context") or "").strip()
            if not evidence:
                return "LLM generation failed and retrieval returned no evidence."
            preview = evidence[:1800]
            return (
                "LLM generation failed; showing retrieval evidence only.\n\n"
                f"Question: {question}\n\n"
                f"Top evidence preview:\n{preview}"
            )


def _set_sentence_transformer_max_length(max_seq_length: int = MAX_SEQ_LENGTH) -> None:
    model = get_model(MODEL_ID)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = int(max_seq_length)
    try:
        first_module = model._first_module()
    except Exception:
        first_module = None
    if first_module is not None and hasattr(first_module, "max_seq_length"):
        first_module.max_seq_length = int(max_seq_length)


@lru_cache(maxsize=1)
def get_lab_pipeline() -> LegalRAGPipeline:
    apply_lab_environment(require_qdrant=True)
    _set_sentence_transformer_max_length(MAX_SEQ_LENGTH)

    dense_dim = int(get_embedding_dimension(MODEL_ID))
    if dense_dim != DENSE_DIM:
        raise RuntimeError(f"Unexpected embedding dimension: expected {DENSE_DIM}, got {dense_dim}")
    if not BM25_PATH.exists():
        raise RuntimeError(
            f"BM25 artifact not found: {BM25_PATH}. Run index_lab_collection.py first."
        )

    pipeline = LegalRAGPipeline(
        embedding_model_name=MODEL_ID,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        query_rewrite_service=None,
    )
    pipeline.initialize()
    pipeline.configure_runtime(
        collection_name=COLLECTION_NAME,
        bm25_path=str(BM25_PATH),
    )
    return pipeline


def _source_payload(citation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "citation_full": citation.get("citation_full") or "",
        "content": citation.get("content") or "",
        "score": citation.get("score"),
        "doc_id": citation.get("doc_id") or citation.get("van_ban_id") or "",
        "so_ky_hieu": citation.get("so_ky_hieu") or "",
        "ten_van_ban": citation.get("ten_van_ban") or "",
        "dieu_so": citation.get("dieu_so") or "",
        "tieu_de_dieu": citation.get("tieu_de_dieu") or "",
        "chunk_id": citation.get("chunk_id") or "",
        "token_len_est": citation.get("token_len_est") or 0,
        "token_len_model": citation.get("token_len_model") or 0,
        "encoder_profile": citation.get("encoder_profile") or "",
    }


def _run_chat(request: ChatRequest) -> Dict[str, Any]:
    pipeline = get_lab_pipeline()
    answer_client = SafeAnswerClient(pipeline.llm)
    started = time.time()
    result = pipeline.query(
        question=request.message.strip(),
        top_k=request.top_k,
        filter_conditions=None,
        use_reranker=request.use_reranker,
        include_inactive=request.include_inactive,
        use_graph_context=False,
        query_rewrite_service=None,
        llm_client=answer_client,
    )
    total_time = time.time() - started
    return {
        "answer": result.answer,
        "sources": [_source_payload(citation) for citation in result.citations],
        "collection": COLLECTION_NAME,
        "embedding_model": MODEL_ID,
        "max_seq_length": MAX_SEQ_LENGTH,
        "search_time": result.retrieval_time,
        "generation_time": result.generation_time,
        "total_time": total_time,
        "rerank_time": result.rerank_time,
        "reranker_used": result.used_reranker,
        "fallback_used": answer_client.fallback_used,
        "fallback_error": answer_client.fallback_error,
        "retrieval_union_count": result.retrieval_union_count,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> Dict[str, Any]:
    configured = True
    error = ""
    try:
        apply_lab_environment(require_qdrant=True)
    except Exception as exc:
        configured = False
        error = str(exc)

    return {
        "configured": configured,
        "error": error,
        "collection": COLLECTION_NAME,
        "embedding_model": MODEL_ID,
        "max_seq_length": MAX_SEQ_LENGTH,
        "bm25_path": str(BM25_PATH),
        "bm25_exists": BM25_PATH.exists(),
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(_run_chat, request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
