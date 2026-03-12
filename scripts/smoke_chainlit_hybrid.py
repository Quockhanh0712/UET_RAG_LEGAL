"""
Smoke test for Chainlit demo profile (hybrid retrieval + optional LLM answer).

Output artifacts:
  data/gold/research/index_runs/<run_id>/chainlit_smoke_10q.json
  data/gold/research/index_runs/<run_id>/chainlit_smoke_10q.md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# Locked defaults for this smoke profile.
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
os.environ.setdefault("CHAINLIT_TOP_K", "5")
os.environ.setdefault("CHAINLIT_USE_RERANKER", "true")
os.environ.setdefault("CHAINLIT_USE_GRAPH", "false")
os.environ.setdefault("CHAINLIT_REQUIRE_ACTIVE_ONLY", "true")

from src.legal_rag_pipeline import LegalRAGPipeline
from src.qdrant_store import collection_name_for_model


DEFAULT_QUERIES = [
    "Cong an xa xu phat loi khong mang bang lai xe co dung khong?",
    "Phai thuc hien thao tac nap mau vao binh chua va xu ly mau so bo bang hoa chat nhu the nao?",
    "Trinh tu danh gia chat luong hang nam doi voi Kiem soat vien duoc quy dinh nhu the nao?",
    "Tu tu chet sau khi tiem thuoc doc ma khong co than nhan nhan xac thi duoc ho tro mai tang nhu the nao?",
    "Trach nhiem cua Bo Khoa hoc va Cong nghe ve quan ly va phat trien cong nghiep an ninh duoc quy dinh nhu the nao?",
    "Dap pha bien bao khu vuc bien gioi bi phat the nao?",
    "Muc tieu phat trien khu cong nghiep ho tro duoc quy dinh nhu the nao?",
    "Thua phat lai duoc tong dat nhung giay to ho so tai lieu nao?",
    "Muc phat khi su dung trai phep long duong do thi he pho de rua xe",
    "Hai cong ty con chung cong ty me co duoc tham du goi thau cua nhau khong?",
]


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
        raise RuntimeError("QDRANT_USE_CLOUD must be true for this smoke profile.")

    qdrant_url = _require_env("QDRANT_CLOUD_URL")
    qdrant_api_key = _require_env("QDRANT_API_KEY")
    collection_base = str(os.getenv("QDRANT_COLLECTION", "")).strip()
    if collection_base != "legal_documents":
        raise RuntimeError("QDRANT_COLLECTION must be legal_documents.")
    if not _env_bool("QDRANT_COLLECTION_BY_MODEL", True):
        raise RuntimeError("QDRANT_COLLECTION_BY_MODEL must be true.")

    model_id = str(os.getenv("EMBEDDING_MODEL", "")).strip()
    if model_id != "truro7/vn-law-embedding":
        raise RuntimeError("EMBEDDING_MODEL must be truro7/vn-law-embedding.")
    if not _env_bool("EMBEDDING_NORMALIZE", True):
        raise RuntimeError("EMBEDDING_NORMALIZE must be true.")

    llm_provider = str(os.getenv("LLM_PROVIDER", "")).strip().lower()
    if llm_provider != "ollama":
        raise RuntimeError("LLM_PROVIDER must be ollama for this smoke profile.")
    _require_env("OLLAMA_URL")
    _require_env("OLLAMA_MODEL")

    return {
        "qdrant_url": qdrant_url,
        "qdrant_api_key": qdrant_api_key,
        "collection_base": collection_base,
        "model_id": model_id,
    }


def _validate_bm25_contract(collection_name: str) -> Path:
    bm25_path = Path(os.getenv("CHAINLIT_BM25_PATH", str(PROJECT_ROOT / "data" / "qdrant_db" / "bm25_encoder.json")))
    if not bm25_path.exists():
        raise RuntimeError(f"BM25 encoder not found: {bm25_path}")

    manifest_env = str(os.getenv("CHAINLIT_BM25_MANIFEST", "")).strip()
    if manifest_env:
        manifest_path = Path(manifest_env)
        if not manifest_path.exists():
            raise RuntimeError(f"BM25 manifest path not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        collections = payload.get("collections", {})
        if collection_name not in collections:
            raise RuntimeError(
                "BM25 manifest mismatch: target collection not present in manifest. "
                "Set correct CHAINLIT_BM25_MANIFEST or disable it."
            )
    return bm25_path


def _validate_collection_green(qdrant_url: str, api_key: str, collection_name: str) -> Tuple[int, str]:
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
    points_count = int(getattr(info, "points_count", 0) or 0)
    return points_count, status


def _resolve_collection_name(collection_base: str, model_id: str, embedding_dim: int) -> str:
    override = str(os.getenv("CHAINLIT_COLLECTION_OVERRIDE", "")).strip()
    if override:
        return override
    return collection_name_for_model(collection_base, model_id, embedding_dim)


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
    rows = []
    for r in results:
        meta = r.metadata or {}
        rows.append(
            {
                "doc_id": meta.get("doc_id", meta.get("van_ban_id", "")),
                "citation_full": meta.get("citation_full", ""),
                "legal_path": meta.get("legal_path", meta.get("path", "")),
                "chunk_type": meta.get("chunk_type", ""),
                "score": float(getattr(r, "score", 0.0)),
                "is_active": bool(meta.get("is_active", True)),
            }
        )
    return rows


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


def _build_pipeline(collection_base: str, model_id: str) -> Tuple[LegalRAGPipeline, str]:
    pipeline = LegalRAGPipeline(embedding_model_name=model_id, normalize_embeddings=True)
    pipeline.initialize()
    collection_name = _resolve_collection_name(
        collection_base=collection_base,
        model_id=model_id,
        embedding_dim=int(pipeline.embedding_model.embedding_dim),
    )
    pipeline.qdrant_store.collection_name = collection_name
    pipeline.qdrant_store.dense_dim = int(pipeline.embedding_model.embedding_dim)
    bm25_path = _validate_bm25_contract(collection_name)
    pipeline.qdrant_store.set_bm25_path(str(bm25_path), load_if_exists=True)
    return pipeline, collection_name


def _pctl(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * q)
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def _write_markdown(path: Path, payload: Dict) -> None:
    rows = payload.get("rows", [])
    summary = payload.get("summary", {})
    lines = [
        "# Chainlit Hybrid Smoke (10 queries)",
        "",
        f"- run_id: `{payload.get('run_id','')}`",
        f"- created_at: `{payload.get('created_at','')}`",
        f"- model: `{payload.get('model','')}`",
        f"- collection: `{payload.get('collection','')}`",
        f"- top_k: `{payload.get('top_k', 0)}`",
        f"- use_reranker: `{payload.get('use_reranker', False)}`",
        f"- reranker_model: `{payload.get('reranker_model', '')}`",
        f"- reranker_alpha: `{payload.get('reranker_alpha', '')}`",
        f"- use_graph: `{payload.get('use_graph', False)}`",
        f"- require_active_only: `{payload.get('require_active_only', False)}`",
        "",
        "## Summary",
        "",
        f"- query_total: `{summary.get('query_total', 0)}`",
        f"- status_ok: `{summary.get('status_ok', 0)}`",
        f"- fallback_count: `{summary.get('fallback_count', 0)}`",
        f"- hard_fail_count: `{summary.get('hard_fail_count', 0)}`",
        f"- retrieval_p50_ms: `{summary.get('retrieval_p50_ms', 0.0):.2f}`",
        f"- retrieval_p95_ms: `{summary.get('retrieval_p95_ms', 0.0):.2f}`",
        f"- generation_p50_ms: `{summary.get('generation_p50_ms', 0.0):.2f}`",
        f"- generation_p95_ms: `{summary.get('generation_p95_ms', 0.0):.2f}`",
        f"- rerank_used_count: `{summary.get('rerank_used_count', 0)}`",
        f"- rerank_p50_ms: `{summary.get('rerank_p50_ms', 0.0):.2f}`",
        f"- rerank_p95_ms: `{summary.get('rerank_p95_ms', 0.0):.2f}`",
        "",
        "## Gate Checks",
        "",
        f"- at_least_3_results_each_query: `{summary.get('gate_min3_results', False)}`",
        f"- top1_citation_non_empty_gte_9: `{summary.get('gate_top1_citation', False)}`",
        f"- llm_non_empty_or_fallback_gte_9: `{summary.get('gate_llm_or_fallback', False)}`",
        f"- reranker_used_when_enabled: `{summary.get('gate_reranker_used', False)}`",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(f"### Q{idx}. {row.get('query','')}")
        lines.append(f"- status: `{row.get('status','')}`")
        lines.append(f"- retrieval_time_ms: `{float(row.get('retrieval_time_ms', 0.0)):.2f}`")
        lines.append(f"- rerank_time_ms: `{float(row.get('rerank_time_ms', 0.0)):.2f}`")
        lines.append(f"- used_reranker: `{bool(row.get('used_reranker', False))}`")
        lines.append(f"- generation_time_ms: `{float(row.get('generation_time_ms', 0.0)):.2f}`")
        lines.append(f"- error: `{row.get('error','')}`")
        results = row.get("results", [])
        if not results:
            lines.append("- results: `[]`")
            lines.append("")
            continue
        for ridx, res in enumerate(results[:3], start=1):
            citation = str(res.get("citation_full", "")).replace("\n", " ").strip()
            if len(citation) > 220:
                citation = citation[:220].rstrip() + "..."
            lines.append(
                f"- R{ridx}: score=`{float(res.get('score', 0.0)):.4f}` "
                f"doc_id=`{res.get('doc_id','')}` path=`{res.get('legal_path','')}` type=`{res.get('chunk_type','')}`"
            )
            lines.append(f"  citation: {citation}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_queries(path: Path | None) -> List[str]:
    if path is None:
        return list(DEFAULT_QUERIES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries", [])
    if not isinstance(queries, list) or not queries:
        raise RuntimeError("Invalid queries file: expected {\"queries\": [...]} with non-empty list.")
    return [str(q).strip() for q in queries if str(q).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test chainlit demo hybrid retrieval.")
    parser.add_argument("--run-id", default="", help="Optional run id. Default uses UTC timestamp.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "gold" / "research" / "index_runs"),
        help="Output root folder for artifacts.",
    )
    parser.add_argument("--queries-json", default="", help="Optional JSON file with {'queries': [...]} .")
    args = parser.parse_args()

    config = _validate_runtime_config()

    run_id = str(args.run_id).strip() or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(os.getenv("CHAINLIT_TOP_K", "5"))
    use_reranker = _env_bool("CHAINLIT_USE_RERANKER", True)
    use_graph = _env_bool("CHAINLIT_USE_GRAPH", False)
    require_active_only = _env_bool("CHAINLIT_REQUIRE_ACTIVE_ONLY", True)
    queries = _load_queries(Path(args.queries_json) if str(args.queries_json).strip() else None)

    pipeline, collection_name = _build_pipeline(
        collection_base=str(config["collection_base"]),
        model_id=str(config["model_id"]),
    )

    points_count, status = _validate_collection_green(
        qdrant_url=str(config["qdrant_url"]),
        api_key=str(config["qdrant_api_key"]),
        collection_name=collection_name,
    )

    rows: List[Dict] = []
    retrieval_latencies: List[float] = []
    generation_latencies: List[float] = []
    rerank_latencies: List[float] = []
    fallback_count = 0
    hard_fail_count = 0
    top1_citation_non_empty = 0
    llm_non_empty_or_fallback = 0
    rerank_used_count = 0
    min3_results_every_query = True

    for query in queries:
        record = {
            "run_id": run_id,
            "query": query,
            "status": "ok",
            "top_k": top_k,
            "results": [],
            "retrieval_time_ms": 0.0,
            "rerank_time_ms": 0.0,
            "used_reranker": False,
            "generation_time_ms": 0.0,
            "error": "",
        }
        try:
            answer = pipeline.query(
                question=query,
                top_k=top_k,
                filter_conditions=None,
                use_reranker=use_reranker,
                include_inactive=not require_active_only,
                use_graph_context=use_graph,
                graph_hops=1,
                graph_expand_docs=20,
            )
            record["retrieval_time_ms"] = float(answer.retrieval_time) * 1000.0
            record["rerank_time_ms"] = float(getattr(answer, "rerank_time", 0.0)) * 1000.0
            record["used_reranker"] = bool(getattr(answer, "used_reranker", False))
            record["generation_time_ms"] = float(answer.generation_time) * 1000.0
            record["results"] = list(answer.citations or [])
            if record["used_reranker"]:
                rerank_used_count += 1

            answer_text = str(answer.answer or "").strip()
            if _is_llm_failure(answer_text):
                raise RuntimeError(answer_text or "LLM failure.")
            if answer_text:
                llm_non_empty_or_fallback += 1
        except Exception as exc:
            fallback_count += 1
            record["status"] = "fallback"
            record["error"] = str(exc)
            fallback_start = time.perf_counter()
            try:
                record["results"] = _retrieve_only(
                    pipeline=pipeline,
                    query=query,
                    top_k=top_k,
                    require_active_only=require_active_only,
                )
            except Exception as fallback_exc:
                hard_fail_count += 1
                record["status"] = "error"
                record["error"] = f"{exc} | fallback_error: {fallback_exc}"
                record["results"] = []
            record["retrieval_time_ms"] = (time.perf_counter() - fallback_start) * 1000.0
            record["rerank_time_ms"] = 0.0
            record["used_reranker"] = False
            record["generation_time_ms"] = 0.0
            if record["results"]:
                llm_non_empty_or_fallback += 1

        results = record["results"]
        if len(results) < 3:
            min3_results_every_query = False
        if results and str(results[0].get("citation_full", "")).strip():
            top1_citation_non_empty += 1

        retrieval_latencies.append(float(record["retrieval_time_ms"]))
        rerank_latencies.append(float(record["rerank_time_ms"]))
        generation_latencies.append(float(record["generation_time_ms"]))
        rows.append(record)

    summary = {
        "query_total": len(queries),
        "status_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "fallback_count": fallback_count,
        "hard_fail_count": hard_fail_count,
        "retrieval_p50_ms": statistics.median(retrieval_latencies) if retrieval_latencies else 0.0,
        "retrieval_p95_ms": _pctl(retrieval_latencies, 0.95) if retrieval_latencies else 0.0,
        "generation_p50_ms": statistics.median(generation_latencies) if generation_latencies else 0.0,
        "generation_p95_ms": _pctl(generation_latencies, 0.95) if generation_latencies else 0.0,
        "rerank_used_count": rerank_used_count,
        "rerank_p50_ms": statistics.median(rerank_latencies) if rerank_latencies else 0.0,
        "rerank_p95_ms": _pctl(rerank_latencies, 0.95) if rerank_latencies else 0.0,
        "gate_min3_results": bool(min3_results_every_query),
        "gate_top1_citation": bool(top1_citation_non_empty >= 9),
        "gate_llm_or_fallback": bool(llm_non_empty_or_fallback >= 9),
        "gate_reranker_used": (rerank_used_count > 0) if use_reranker else True,
    }

    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(config["model_id"]),
        "collection": collection_name,
        "collection_status": status,
        "collection_points_count": points_count,
        "qdrant_url": str(config["qdrant_url"]),
        "top_k": top_k,
        "use_reranker": use_reranker,
        "reranker_model": os.getenv("RERANKER_MODEL", ""),
        "reranker_alpha": os.getenv("RERANKER_ALPHA", ""),
        "use_graph": use_graph,
        "require_active_only": require_active_only,
        "rows": rows,
        "summary": summary,
    }

    json_path = output_dir / "chainlit_smoke_10q.json"
    md_path = output_dir / "chainlit_smoke_10q.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)

    print(str(json_path))
    print(str(md_path))
    print(json.dumps(summary, ensure_ascii=False))

    if hard_fail_count > 0:
        return 2
    if not summary["gate_min3_results"]:
        return 3
    if not summary["gate_top1_citation"]:
        return 4
    if not summary["gate_llm_or_fallback"]:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
