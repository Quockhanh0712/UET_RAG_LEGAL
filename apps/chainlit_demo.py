"""
Chainlit demo app for hybrid legal retrieval on Qdrant Cloud.

Run:
  chainlit run apps/chainlit_demo.py -w
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import chainlit as cl
from dotenv import load_dotenv
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# Locked defaults for this demo profile.
os.environ["EMBEDDING_MODEL"] = "truro7/vn-law-embedding"
os.environ["EMBEDDING_NORMALIZE"] = "true"
os.environ["EMBEDDING_DEVICE"] = "cuda"
os.environ["USE_FP16"] = "true"
os.environ["QDRANT_COLLECTION"] = "legal_documents"
os.environ["QDRANT_COLLECTION_BY_MODEL"] = "true"
os.environ["QDRANT_USE_CLOUD"] = "true"
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_MODEL"] = os.getenv("OLLAMA_MODEL", "qwen3:8b")
os.environ["OLLAMA_URL"] = os.getenv("OLLAMA_URL", "http://222.255.250.24:11434")
os.environ["RERANKER_MODEL"] = "AITeamVN/Vietnamese_Reranker"
os.environ["RERANKER_ALPHA"] = "1.0"
os.environ.setdefault("RERANKER_MAX_LENGTH", "2304")
os.environ.setdefault("RERANKER_BATCH_SIZE", "8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.legal_rag_pipeline import LegalRAGPipeline
from src.qdrant_store import collection_name_for_model


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _require_env(name: str) -> str:
    value = str(os.getenv(name, "")).strip()
    if not value:
        raise RuntimeError(f"Missing required env: {name}")
    return value


def _validate_runtime_config() -> Dict[str, object]:
    use_cloud = _env_bool("QDRANT_USE_CLOUD", True)
    if not use_cloud:
        raise RuntimeError("QDRANT_USE_CLOUD must be true for this demo profile.")

    qdrant_url = _require_env("QDRANT_CLOUD_URL")
    _require_env("QDRANT_API_KEY")

    collection_base = str(os.getenv("QDRANT_COLLECTION", "")).strip()
    if collection_base != "legal_documents":
        raise RuntimeError("QDRANT_COLLECTION must be legal_documents.")

    if not _env_bool("QDRANT_COLLECTION_BY_MODEL", True):
        raise RuntimeError("QDRANT_COLLECTION_BY_MODEL must be true.")

    model_id = str(os.getenv("EMBEDDING_MODEL", "")).strip()
    if model_id != "truro7/vn-law-embedding":
        raise RuntimeError("EMBEDDING_MODEL must be truro7/vn-law-embedding.")

    if not _env_bool("EMBEDDING_NORMALIZE", True):
        raise RuntimeError("EMBEDDING_NORMALIZE must be true for this demo.")

    llm_provider = str(os.getenv("LLM_PROVIDER", "")).strip().lower()
    if llm_provider != "ollama":
        raise RuntimeError("LLM_PROVIDER must be ollama for this demo profile.")

    _require_env("OLLAMA_URL")
    _require_env("OLLAMA_MODEL")

    return {
        "qdrant_url": qdrant_url,
        "model_id": model_id,
        "collection_base": collection_base,
    }


def _validate_collection_green(qdrant_url: str, api_key: str, collection_name: str) -> None:
    client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=30)
    names = [c.name for c in client.get_collections().collections]
    if collection_name not in names:
        preview = ", ".join(names[:20]) if names else "<none>"
        raise RuntimeError(
            f"Collection not found on cloud: {collection_name} | url={qdrant_url} | available={preview} "
            "=> Set QDRANT_CLOUD_URL to the indexed cluster or set CHAINLIT_COLLECTION_OVERRIDE."
        )
    info = client.get_collection(collection_name)
    status = str(getattr(info, "status", "")).lower()
    if status != "green":
        raise RuntimeError(f"Collection {collection_name} is not green (status={status}).")


def _resolve_collection_name(collection_base: str, model_id: str, embedding_dim: int) -> str:
    override = str(os.getenv("CHAINLIT_COLLECTION_OVERRIDE", "")).strip()
    if override:
        return override
    return collection_name_for_model(collection_base, model_id, embedding_dim)


def _validate_bm25_contract(collection_name: str) -> Path:
    bm25_path = Path(os.getenv("CHAINLIT_BM25_PATH", str(PROJECT_ROOT / "data" / "qdrant_db" / "bm25_encoder.json")))
    if not bm25_path.exists():
        raise RuntimeError(f"BM25 encoder not found: {bm25_path}")

    manifest_env = str(os.getenv("CHAINLIT_BM25_MANIFEST", "")).strip()
    if manifest_env:
        manifest_path = Path(manifest_env)
        if not manifest_path.exists():
            raise RuntimeError(f"BM25 manifest path not found: {manifest_path}")
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        collections = payload.get("collections", {})
        if collection_name not in collections:
            raise RuntimeError(
                "BM25 manifest mismatch: target collection not present in manifest. "
                "Set correct CHAINLIT_BM25_MANIFEST or disable it."
            )
    return bm25_path


@lru_cache(maxsize=1)
def _get_pipeline() -> LegalRAGPipeline:
    config = _validate_runtime_config()
    pipeline = LegalRAGPipeline(
        embedding_model_name=str(config["model_id"]),
        normalize_embeddings=True,
    )
    pipeline.initialize()

    collection = _resolve_collection_name(
        collection_base=str(config["collection_base"]),
        model_id=str(config["model_id"]),
        embedding_dim=int(pipeline.embedding_model.embedding_dim),
    )
    pipeline.qdrant_store.collection_name = collection
    pipeline.qdrant_store.dense_dim = int(pipeline.embedding_model.embedding_dim)

    bm25_path = _validate_bm25_contract(collection)
    pipeline.qdrant_store.set_bm25_path(str(bm25_path), load_if_exists=True)

    _validate_collection_green(
        qdrant_url=str(config["qdrant_url"]),
        api_key=_require_env("QDRANT_API_KEY"),
        collection_name=collection,
    )
    return pipeline


def _is_llm_failure(answer: str) -> bool:
    text = str(answer or "").strip().lower()
    if not text:
        return True
    markers = [
        "loi khi goi llm",
        "llm khong kha dung",
        "error calling ollama api",
        "failed",
    ]
    return any(marker in text for marker in markers)


def _to_citation_items(results: List) -> List[Dict]:
    items: List[Dict] = []
    for r in results:
        meta = r.metadata or {}
        items.append(
            {
                "doc_id": meta.get("doc_id", meta.get("van_ban_id", "")),
                "citation_full": meta.get("citation_full", ""),
                "legal_path": meta.get("legal_path", meta.get("path", "")),
                "chunk_type": meta.get("chunk_type", ""),
                "score": float(getattr(r, "score", 0.0)),
                "is_active": bool(meta.get("is_active", True)),
            }
        )
    return items


def _retrieve_only(pipeline: LegalRAGPipeline, query: str, top_k: int, require_active_only: bool) -> List[Dict]:
    filters: Dict[str, object] = {}
    if require_active_only:
        filters["is_active"] = True

    query_embedding = pipeline.embedding_model.encode_query(query)
    try:
        results = pipeline.qdrant_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            dense_weight=0.6,
            sparse_weight=0.4,
            filter_conditions=filters,
        )
    except Exception:
        results = pipeline.qdrant_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            dense_weight=0.6,
            sparse_weight=0.4,
            filter_conditions={},
        )
    if not results and require_active_only:
        results = pipeline.qdrant_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            dense_weight=0.6,
            sparse_weight=0.4,
            filter_conditions={},
        )
    return _to_citation_items(results[:top_k])


def _format_sources(citations: List[Dict], top_k: int) -> str:
    lines = []
    for idx, c in enumerate(citations[:top_k], start=1):
        citation = str(c.get("citation_full", "")).replace("\n", " ").strip()
        if len(citation) > 260:
            citation = citation[:260].rstrip() + "..."
        lines.append(
            f"{idx}. doc_id=`{c.get('doc_id','')}` | path=`{c.get('legal_path','')}` "
            f"| type=`{c.get('chunk_type','')}` | score=`{float(c.get('score', 0.0)):.4f}`\n"
            f"   - {citation}"
        )
    return "\n".join(lines) if lines else "No sources."


@cl.on_chat_start
async def on_chat_start() -> None:
    top_k = int(os.getenv("CHAINLIT_TOP_K", "5"))
    use_reranker = _env_bool("CHAINLIT_USE_RERANKER", True)
    use_graph = _env_bool("CHAINLIT_USE_GRAPH", False)
    require_active_only = _env_bool("CHAINLIT_REQUIRE_ACTIVE_ONLY", True)

    try:
        pipeline = await asyncio.to_thread(_get_pipeline)
        collection_name = pipeline.qdrant_store.collection_name
        await cl.Message(
            content=(
                "Hybrid Legal Demo is ready.\n\n"
                f"- model: `{os.getenv('EMBEDDING_MODEL')}`\n"
                f"- qdrant_url: `{os.getenv('QDRANT_CLOUD_URL')}`\n"
                f"- collection: `{collection_name}`\n"
                f"- top_k: `{top_k}`\n"
                f"- use_reranker: `{use_reranker}`\n"
                f"- reranker_model: `{os.getenv('RERANKER_MODEL')}`\n"
                f"- reranker_alpha: `{os.getenv('RERANKER_ALPHA')}`\n"
                f"- use_graph: `{use_graph}`\n"
                f"- require_active_only: `{require_active_only}`"
            )
        ).send()
    except Exception as exc:
        await cl.Message(content=f"Startup error: {exc}").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = str(message.content or "").strip()
    if not query:
        await cl.Message(content="Empty query.").send()
        return

    top_k = int(os.getenv("CHAINLIT_TOP_K", "5"))
    use_reranker = _env_bool("CHAINLIT_USE_RERANKER", True)
    use_graph = _env_bool("CHAINLIT_USE_GRAPH", False)
    require_active_only = _env_bool("CHAINLIT_REQUIRE_ACTIVE_ONLY", True)

    try:
        pipeline = await asyncio.to_thread(_get_pipeline)
    except Exception as exc:
        await cl.Message(content=f"Configuration error: {exc}").send()
        return

    retrieval_ms = 0.0
    generation_ms = 0.0
    rerank_ms = 0.0
    used_reranker = False
    answer_text = ""
    citations: List[Dict] = []
    fallback_error = ""

    started = time.perf_counter()
    try:
        legal_answer = await asyncio.to_thread(
            pipeline.query,
            query,
            top_k,
            None,
            use_reranker,
            not require_active_only,
            use_graph,
            1,
            20,
        )
        retrieval_ms = float(legal_answer.retrieval_time) * 1000.0
        generation_ms = float(legal_answer.generation_time) * 1000.0
        rerank_ms = float(getattr(legal_answer, "rerank_time", 0.0)) * 1000.0
        used_reranker = bool(getattr(legal_answer, "used_reranker", False))
        answer_text = str(legal_answer.answer or "").strip()
        citations = list(legal_answer.citations or [])
        if _is_llm_failure(answer_text):
            raise RuntimeError(answer_text or "LLM generation failed.")
    except Exception as exc:
        fallback_error = str(exc)
        fallback_start = time.perf_counter()
        citations = await asyncio.to_thread(_retrieve_only, pipeline, query, top_k, require_active_only)
        retrieval_ms = (time.perf_counter() - fallback_start) * 1000.0
        generation_ms = 0.0
        rerank_ms = 0.0
        used_reranker = False
        answer_text = (
            "LLM unavailable or failed. Returning retrieval-only sources. "
            "Please inspect citations below."
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    source_text = _format_sources(citations, top_k=top_k)
    content = (
        f"{answer_text}\n\n"
        f"---\n\n"
        f"**Metrics**\n"
        f"- retrieval_time_ms: `{retrieval_ms:.2f}`\n"
        f"- rerank_time_ms: `{rerank_ms:.2f}`\n"
        f"- used_reranker: `{used_reranker}`\n"
        f"- generation_time_ms: `{generation_ms:.2f}`\n"
        f"- end_to_end_time_ms: `{elapsed_ms:.2f}`\n"
        f"- fallback_used: `{bool(fallback_error)}`\n"
        f"- fallback_error: `{fallback_error or ''}`\n\n"
        f"**Top Sources**\n{source_text}"
    )
    await cl.Message(content=content).send()
