"""Smoke test for Chainlit demo profile (hybrid retrieval + optional LLM answer)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.chainlit_adapter import execute_chat_query
from src.config.runtime_settings import load_runtime_settings
from src.factories.legal_pipeline_factory import get_legal_pipeline
from src.runtime.qdrant_contract import validate_qdrant_contract

SETTINGS = load_runtime_settings("chainlit_smoke")
SETTINGS.apply_env()

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
        f"- run_id: `{payload.get('run_id', '')}`",
        f"- created_at: `{payload.get('created_at', '')}`",
        f"- model: `{payload.get('model', '')}`",
        f"- collection: `{payload.get('collection', '')}`",
        f"- collection_status: `{payload.get('collection_status', '')}`",
        f"- top_k: `{payload.get('top_k', 0)}`",
        f"- use_reranker: `{payload.get('use_reranker', False)}`",
        f"- reranker_model: `{payload.get('reranker_model', '')}`",
        f"- reranker_alpha: `{payload.get('reranker_alpha', '')}`",
        f"- query_rewriter_enabled: `{payload.get('query_rewriter_enabled', False)}`",
        f"- query_rewriter_force: `{payload.get('query_rewriter_force', False)}`",
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
        f"- rewrite_used_count: `{summary.get('rewrite_used_count', 0)}`",
        f"- rewrite_p50_ms: `{summary.get('rewrite_p50_ms', 0.0):.2f}`",
        f"- rewrite_p95_ms: `{summary.get('rewrite_p95_ms', 0.0):.2f}`",
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
        lines.append(f"### Q{idx}. {row.get('query', '')}")
        lines.append(f"- status: `{row.get('status', '')}`")
        lines.append(f"- retrieval_time_ms: `{float(row.get('retrieval_time_ms', 0.0)):.2f}`")
        lines.append(f"- rewrite_time_ms: `{float(row.get('rewrite_time_ms', 0.0)):.2f}`")
        lines.append(f"- rewrite_used: `{bool(row.get('rewrite_used', False))}`")
        lines.append(f"- rewrite_status: `{row.get('rewrite_status', '')}`")
        lines.append(f"- rewrite_query_count: `{int(row.get('rewrite_query_count', 0) or 0)}`")
        lines.append(f"- retrieval_union_count: `{int(row.get('retrieval_union_count', 0) or 0)}`")
        lines.append(f"- rerank_time_ms: `{float(row.get('rerank_time_ms', 0.0)):.2f}`")
        lines.append(f"- used_reranker: `{bool(row.get('used_reranker', False))}`")
        lines.append(f"- generation_time_ms: `{float(row.get('generation_time_ms', 0.0)):.2f}`")
        lines.append(f"- error: `{row.get('error', '')}`")
        lines.append(f"- rewritten_queries: `{row.get('rewritten_queries', [])}`")
        lines.append(f"- missing_numbers: `{row.get('missing_numbers', [])}`")
        lines.append(f"- per_subquery_hits: `{row.get('per_subquery_hits', [])}`")
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
                f"doc_id=`{res.get('doc_id', '')}` path=`{res.get('legal_path', '')}` type=`{res.get('chunk_type', '')}`"
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
        raise RuntimeError("Invalid queries file: expected {'queries': [...]} with non-empty list.")
    return [str(q).strip() for q in queries if str(q).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Chainlit demo hybrid retrieval.")
    parser.add_argument("--run-id", default="", help="Optional run id. Default uses UTC timestamp.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "gold" / "research" / "index_runs"),
        help="Output root folder for artifacts.",
    )
    parser.add_argument("--queries-json", default="", help="Optional JSON file with {'queries': [...]} .")
    args = parser.parse_args()

    run_id = str(args.run_id).strip() or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = SETTINGS.chainlit_top_k
    use_reranker = SETTINGS.chainlit_use_reranker
    use_graph = SETTINGS.chainlit_use_graph
    require_active_only = SETTINGS.chainlit_require_active_only
    queries = _load_queries(Path(args.queries_json) if str(args.queries_json).strip() else None)

    pipeline = get_legal_pipeline(SETTINGS)
    contract = validate_qdrant_contract(SETTINGS, int(pipeline.embedding_model.embedding_dim))
    qdrant_stats = pipeline.qdrant_store.get_stats()

    rows: List[Dict] = []
    retrieval_latencies: List[float] = []
    generation_latencies: List[float] = []
    rerank_latencies: List[float] = []
    rewrite_latencies: List[float] = []
    fallback_count = 0
    hard_fail_count = 0
    top1_citation_non_empty = 0
    llm_non_empty_or_fallback = 0
    rerank_used_count = 0
    rewrite_used_count = 0
    min3_results_every_query = True

    for query in queries:
        execution = execute_chat_query(
            pipeline=pipeline,
            query=query,
            top_k=top_k,
            use_reranker=use_reranker,
            require_active_only=require_active_only,
            use_graph=use_graph,
        )
        row = execution.to_smoke_row(run_id=run_id, query=query, top_k=top_k)
        rows.append(row)

        if execution.rewrite_used:
            rewrite_used_count += 1
        if execution.used_reranker:
            rerank_used_count += 1
        if execution.fallback_used:
            fallback_count += 1
        if execution.status == "error":
            hard_fail_count += 1
        if execution.answer_text or execution.fallback_used:
            llm_non_empty_or_fallback += 1

        results = row["results"]
        if len(results) < 3:
            min3_results_every_query = False
        if results and str(results[0].get("citation_full", "")).strip():
            top1_citation_non_empty += 1

        retrieval_latencies.append(float(row["retrieval_time_ms"]))
        rewrite_latencies.append(float(row["rewrite_time_ms"]))
        rerank_latencies.append(float(row["rerank_time_ms"]))
        generation_latencies.append(float(row["generation_time_ms"]))

    summary = {
        "query_total": len(queries),
        "status_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "fallback_count": fallback_count,
        "hard_fail_count": hard_fail_count,
        "retrieval_p50_ms": statistics.median(retrieval_latencies) if retrieval_latencies else 0.0,
        "retrieval_p95_ms": _pctl(retrieval_latencies, 0.95) if retrieval_latencies else 0.0,
        "generation_p50_ms": statistics.median(generation_latencies) if generation_latencies else 0.0,
        "generation_p95_ms": _pctl(generation_latencies, 0.95) if generation_latencies else 0.0,
        "rerank_used_count": rerank_used_count,
        "rerank_p50_ms": statistics.median(rerank_latencies) if rerank_latencies else 0.0,
        "rerank_p95_ms": _pctl(rerank_latencies, 0.95) if rerank_latencies else 0.0,
        "rewrite_used_count": rewrite_used_count,
        "rewrite_p50_ms": statistics.median(rewrite_latencies) if rewrite_latencies else 0.0,
        "rewrite_p95_ms": _pctl(rewrite_latencies, 0.95) if rewrite_latencies else 0.0,
        "gate_min3_results": bool(min3_results_every_query),
        "gate_top1_citation": bool(top1_citation_non_empty >= 9),
        "gate_llm_or_fallback": bool(llm_non_empty_or_fallback >= 9),
        "gate_reranker_used": (rerank_used_count > 0) if use_reranker else True,
    }

    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": SETTINGS.embedding_model,
        "collection": contract.collection_name,
        "collection_status": contract.collection_status,
        "collection_points_count": int(qdrant_stats.get("points_count", 0) or 0),
        "qdrant_url": SETTINGS.qdrant_url,
        "top_k": top_k,
        "use_reranker": use_reranker,
        "reranker_model": SETTINGS.reranker_model,
        "reranker_alpha": str(SETTINGS.reranker_alpha),
        "query_rewriter_enabled": SETTINGS.query_rewriter_enabled,
        "query_rewriter_force": SETTINGS.query_rewriter_force,
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
