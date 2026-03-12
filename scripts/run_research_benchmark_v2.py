"""
Run research benchmark variants and export contract-compliant results.

Extended output columns:
run_id, variant_id, embedding_model_id, embedding_dim, normalize_embeddings,
status, num_queries, Recall@20, nDCG@10, MRR@10, citation_valid_rate,
latency_p95_ms, encode_p95_ms, notes, created_at
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import DEFAULT_MODEL, get_embedding_dimension
from src.legal_rag_pipeline import LegalRAGPipeline


DEFAULT_COLLECTION_BASE = os.getenv("QDRANT_COLLECTION", "legal_documents")


@dataclass
class VariantConfig:
    variant_id: str
    dense_weight: float
    sparse_weight: float
    reranker: bool
    graph_mode: str


@dataclass
class EmbeddingConfig:
    model_id: str
    embedding_dim: int
    normalize_embeddings: Optional[bool]
    collection_name: str


BENCHMARK_VARIANTS: List[VariantConfig] = [
    VariantConfig("B0", 0.0, 1.0, False, "none"),
    VariantConfig("B1", 1.0, 0.0, False, "none"),
    VariantConfig("B2", 0.6, 0.4, False, "none"),
    VariantConfig("B3", 0.6, 0.4, True, "none"),
    VariantConfig("G1", 0.6, 0.4, True, "local"),
    VariantConfig("G2", 0.6, 0.4, True, "legal_impact"),
    VariantConfig("G3", 0.6, 0.4, True, "temporal"),
]


def _dcg(relevances: List[float]) -> float:
    total = 0.0
    for i, rel in enumerate(relevances, start=1):
        denom = math.log2(i + 1)
        total += (2**rel - 1) / denom
    return total


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def _load_queries(path: Path, max_queries: int = 0) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    if max_queries > 0:
        rows = rows[:max_queries]
    return rows


def _load_labels(path: Path) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    if "query_id" not in df.columns or "doc_id" not in df.columns:
        return {}

    labels: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for _, row in df.iterrows():
        query_id = str(row.get("query_id", "")).strip()
        doc_id = str(row.get("doc_id", "")).strip().upper()
        if not query_id or not doc_id:
            continue
        rel = _safe_float(row.get("relevance"))
        citation_valid = _safe_float(row.get("citation_valid"))
        labels.setdefault(query_id, {})[doc_id] = {
            "relevance": rel,
            "citation_valid": citation_valid,
        }
    return labels


def _compute_query_metrics(
    retrieved_doc_ids: List[str],
    label_map: Dict[str, Dict[str, Optional[float]]],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not label_map:
        return None, None, None, None

    relevant_docs = {doc_id for doc_id, value in label_map.items() if (value.get("relevance") or 0) > 0}
    if not relevant_docs:
        return None, None, None, None

    top20 = retrieved_doc_ids[:20]
    top10 = retrieved_doc_ids[:10]

    hits_top20 = sum(1 for doc_id in top20 if doc_id in relevant_docs)
    recall20 = hits_top20 / len(relevant_docs) if relevant_docs else None

    reciprocal_rank = 0.0
    for idx, doc_id in enumerate(top10, start=1):
        if doc_id in relevant_docs:
            reciprocal_rank = 1.0 / idx
            break
    mrr10 = reciprocal_rank

    rels = []
    for doc_id in top10:
        rels.append(float(label_map.get(doc_id, {}).get("relevance") or 0.0))
    dcg_val = _dcg(rels)
    ideal_rels = sorted([float(v.get("relevance") or 0.0) for v in label_map.values()], reverse=True)[:10]
    idcg_val = _dcg(ideal_rels)
    ndcg10 = (dcg_val / idcg_val) if idcg_val > 0 else 0.0

    citation_marks = []
    for doc_id in top10:
        val = label_map.get(doc_id, {}).get("citation_valid")
        if val is not None:
            citation_marks.append(1.0 if float(val) >= 1 else 0.0)
    citation_valid_rate = (sum(citation_marks) / len(citation_marks)) if citation_marks else None

    return recall20, ndcg10, mrr10, citation_valid_rate


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    idx = max(0, min(len(values_sorted) - 1, int(0.95 * (len(values_sorted) - 1))))
    return values_sorted[idx]


def _run_variant(
    pipeline: LegalRAGPipeline,
    queries: List[Dict[str, str]],
    labels: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    variant: VariantConfig,
    top_k: int,
) -> Dict[str, Optional[float]]:
    graph_available = bool(pipeline.graph_store and getattr(pipeline.graph_store, "enabled", False))
    if variant.graph_mode != "none" and not graph_available:
        return {
            "status": "graph-pending",
            "num_queries": len(queries),
            "Recall@20": None,
            "nDCG@10": None,
            "MRR@10": None,
            "citation_valid_rate": None,
            "latency_p95_ms": None,
            "encode_p95_ms": None,
            "notes": "Neo4j graph store unavailable; graph variants deferred.",
        }

    recall_vals: List[float] = []
    ndcg_vals: List[float] = []
    mrr_vals: List[float] = []
    citation_vals: List[float] = []
    latency_ms: List[float] = []
    encode_ms: List[float] = []
    unlabeled_queries = 0

    try:
        for q in queries:
            query_id = str(q.get("query_id", "")).strip()
            query_text = str(q.get("query_text", "")).strip()
            if not query_text:
                continue

            filters = {"is_active": True}
            start = datetime.now().timestamp()

            encode_start = datetime.now().timestamp()
            query_embedding = pipeline.embedding_model.encode_query(query_text)
            encode_elapsed_ms = (datetime.now().timestamp() - encode_start) * 1000.0
            encode_ms.append(encode_elapsed_ms)

            base = pipeline.qdrant_store.hybrid_search(
                query=query_text,
                query_embedding=query_embedding,
                top_k=100 if variant.reranker else max(top_k * 2, top_k),
                dense_weight=variant.dense_weight,
                sparse_weight=variant.sparse_weight,
                filter_conditions=filters,
            )

            merged = base
            if variant.graph_mode != "none":
                shortlist = base[: max(top_k, 20)]
                graph_results = pipeline._graph_expand(
                    base_results=shortlist,
                    filter_conditions=filters,
                    graph_hops=1,
                    graph_expand_docs=20,
                    top_k=top_k,
                    graph_mode=variant.graph_mode,
                )
                merged = pipeline._merge_results(base, graph_results)

            if variant.reranker:
                ranked, _ = pipeline._rerank_results(query_text, merged, top_k=top_k)
            else:
                ranked = merged[:top_k]

            elapsed_ms = (datetime.now().timestamp() - start) * 1000.0
            latency_ms.append(elapsed_ms)

            retrieved_doc_ids = []
            for item in ranked:
                meta = item.metadata or {}
                doc_id = str(meta.get("doc_id") or meta.get("van_ban_id") or "").upper().strip()
                if doc_id:
                    retrieved_doc_ids.append(doc_id)

            label_map = labels.get(query_id, {})
            recall20, ndcg10, mrr10, citation_valid_rate = _compute_query_metrics(retrieved_doc_ids, label_map)
            if recall20 is None or ndcg10 is None or mrr10 is None:
                unlabeled_queries += 1
            else:
                recall_vals.append(recall20)
                ndcg_vals.append(ndcg10)
                mrr_vals.append(mrr10)
            if citation_valid_rate is not None:
                citation_vals.append(citation_valid_rate)
    except Exception as exc:
        return {
            "status": "infra-pending",
            "num_queries": len(queries),
            "Recall@20": None,
            "nDCG@10": None,
            "MRR@10": None,
            "citation_valid_rate": None,
            "latency_p95_ms": None,
            "encode_p95_ms": None,
            "notes": f"Retrieval failed: {exc}",
        }

    notes = ""
    if unlabeled_queries > 0:
        notes = f"{unlabeled_queries}/{len(queries)} queries missing labels."

    return {
        "status": "ok",
        "num_queries": len(queries),
        "Recall@20": statistics.mean(recall_vals) if recall_vals else None,
        "nDCG@10": statistics.mean(ndcg_vals) if ndcg_vals else None,
        "MRR@10": statistics.mean(mrr_vals) if mrr_vals else None,
        "citation_valid_rate": statistics.mean(citation_vals) if citation_vals else None,
        "latency_p95_ms": _p95(latency_ms),
        "encode_p95_ms": _p95(encode_ms),
        "notes": notes,
    }


def _append_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "run_id",
        "variant_id",
        "embedding_model_id",
        "embedding_dim",
        "normalize_embeddings",
        "status",
        "num_queries",
        "Recall@20",
        "nDCG@10",
        "MRR@10",
        "citation_valid_rate",
        "latency_p95_ms",
        "encode_p95_ms",
        "notes",
        "created_at",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-migrate legacy benchmark contract to the new schema when needed.
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            current_fields = reader.fieldnames or []
            existing_rows = list(reader)
        if current_fields != fields:
            migrated_rows = []
            for old_row in existing_rows:
                migrated = {field: old_row.get(field, "") for field in fields}
                migrated_rows.append(migrated)
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for migrated_row in migrated_rows:
                    writer.writerow(migrated_row)

    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_embedding_models(raw: str) -> List[str]:
    models = [m.strip() for m in str(raw or "").split(",") if m.strip()]
    return models or [DEFAULT_MODEL]


def _parse_normalize_options(raw: str) -> List[Optional[bool]]:
    value = str(raw or "auto").strip().lower()
    if value == "both":
        return [True, False]
    if value == "true":
        return [True]
    if value == "false":
        return [False]
    return [None]


def _build_embedding_configs(
    models: List[str],
    normalize_options: List[Optional[bool]],
    collection_base: str,
    collection_by_model: bool,
) -> List[EmbeddingConfig]:
    configs: List[EmbeddingConfig] = []
    for model_id in models:
        try:
            dim = int(get_embedding_dimension(model_id))
        except Exception:
            dim = 0
        for normalize_opt in normalize_options:
            if collection_by_model:
                collection_name = _collection_name_for_model(collection_base, model_id, dim)
            else:
                collection_name = collection_base
            configs.append(
                EmbeddingConfig(
                    model_id=model_id,
                    embedding_dim=dim,
                    normalize_embeddings=normalize_opt,
                    collection_name=collection_name,
                )
            )
    return configs


def _normalize_label(value: Optional[bool]) -> str:
    if value is None:
        return "auto"
    return "true" if value else "false"


def _model_slug(model_name: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(model_name or "").lower())
    text = "_".join([part for part in text.split("_") if part])
    return text[:80] or "unknown_model"


def _collection_name_for_model(base_name: str, model_name: str, dense_dim: int) -> str:
    return f"{base_name}__{_model_slug(model_name)}__d{int(dense_dim)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legal retrieval benchmark variants (v2 extended)")
    parser.add_argument(
        "--queries-path",
        default=str(Path("data/gold/research/query_set_v1.csv")),
        help="CSV with query_id,query_type,query_text",
    )
    parser.add_argument(
        "--labels-path",
        default=str(Path("data/gold/research/eval_pairs_template.csv")),
        help="CSV with query_id,doc_id,relevance,citation_valid",
    )
    parser.add_argument(
        "--output-path",
        default=str(Path("data/gold/research/benchmark_results.csv")),
        help="Benchmark output CSV",
    )
    parser.add_argument("--max-queries", type=int, default=0, help="Limit query count for quick runs")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k retrieval depth")
    parser.add_argument(
        "--embedding-models",
        default="AITeamVN/Vietnamese_Embedding,huyydangg/DEk21_hcmute_embedding",
        help="Comma-separated SentenceTransformer model ids",
    )
    parser.add_argument(
        "--normalize-options",
        default="auto",
        choices=("auto", "true", "false", "both"),
        help="Embedding normalization options",
    )
    parser.add_argument(
        "--collection-base",
        default=DEFAULT_COLLECTION_BASE,
        help="Base collection name",
    )
    parser.add_argument(
        "--collection-by-model",
        action="store_true",
        help="Use collection naming pattern base__model_slug__d{dim}",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries_path)
    labels_path = Path(args.labels_path)
    output_path = Path(args.output_path)

    queries = _load_queries(queries_path, max_queries=args.max_queries)
    labels = _load_labels(labels_path)
    embedding_models = _parse_embedding_models(args.embedding_models)
    normalize_options = _parse_normalize_options(args.normalize_options)
    embedding_configs = _build_embedding_configs(
        embedding_models,
        normalize_options,
        collection_base=args.collection_base,
        collection_by_model=args.collection_by_model,
    )

    now_utc = datetime.now(timezone.utc)
    run_id = now_utc.strftime("run_%Y%m%d_%H%M%S")
    created_at = now_utc.isoformat()

    rows: List[Dict[str, object]] = []
    for embedding_cfg in embedding_configs:
        try:
            pipeline = LegalRAGPipeline(
                embedding_model_name=embedding_cfg.model_id,
                normalize_embeddings=embedding_cfg.normalize_embeddings,
            )
            pipeline.initialize()
            runtime_dim = int(getattr(pipeline.embedding_model, "embedding_dim", 0) or 0)
            if runtime_dim > 0 and runtime_dim != embedding_cfg.embedding_dim:
                embedding_cfg.embedding_dim = runtime_dim
                if args.collection_by_model:
                    embedding_cfg.collection_name = _collection_name_for_model(
                        args.collection_base,
                        embedding_cfg.model_id,
                        runtime_dim,
                    )
            pipeline.qdrant_store.collection_name = embedding_cfg.collection_name
            pipeline.qdrant_store.dense_dim = int(embedding_cfg.embedding_dim)
        except Exception as exc:
            note = f"Retrieval infra unavailable: {exc}"
            for variant in BENCHMARK_VARIANTS:
                rows.append(
                    {
                        "run_id": run_id,
                        "variant_id": variant.variant_id,
                        "embedding_model_id": embedding_cfg.model_id,
                        "embedding_dim": embedding_cfg.embedding_dim,
                        "normalize_embeddings": _normalize_label(embedding_cfg.normalize_embeddings),
                        "status": "infra-pending",
                        "num_queries": len(queries),
                        "Recall@20": None,
                        "nDCG@10": None,
                        "MRR@10": None,
                        "citation_valid_rate": None,
                        "latency_p95_ms": None,
                        "encode_p95_ms": None,
                        "notes": note,
                        "created_at": created_at,
                    }
                )
            for chunk_variant in ("C1", "C2", "C3", "C4"):
                rows.append(
                    {
                        "run_id": run_id,
                        "variant_id": chunk_variant,
                        "embedding_model_id": embedding_cfg.model_id,
                        "embedding_dim": embedding_cfg.embedding_dim,
                        "normalize_embeddings": _normalize_label(embedding_cfg.normalize_embeddings),
                        "status": "pending-chunk-index",
                        "num_queries": len(queries),
                        "Recall@20": None,
                        "nDCG@10": None,
                        "MRR@10": None,
                        "citation_valid_rate": None,
                        "latency_p95_ms": None,
                        "encode_p95_ms": None,
                        "notes": "Requires dedicated chunked index collection for fair comparison.",
                        "created_at": created_at,
                    }
                )
            continue

        for variant in BENCHMARK_VARIANTS:
            result = _run_variant(
                pipeline=pipeline,
                queries=queries,
                labels=labels,
                variant=variant,
                top_k=args.top_k,
            )
            notes = result["notes"] or ""
            collection_note = f"collection={embedding_cfg.collection_name}"
            notes = f"{notes} | {collection_note}" if notes else collection_note
            rows.append(
                {
                    "run_id": run_id,
                    "variant_id": variant.variant_id,
                    "embedding_model_id": embedding_cfg.model_id,
                    "embedding_dim": embedding_cfg.embedding_dim,
                    "normalize_embeddings": _normalize_label(embedding_cfg.normalize_embeddings),
                    "status": result["status"],
                    "num_queries": result["num_queries"],
                    "Recall@20": result["Recall@20"],
                    "nDCG@10": result["nDCG@10"],
                    "MRR@10": result["MRR@10"],
                    "citation_valid_rate": result["citation_valid_rate"],
                    "latency_p95_ms": result["latency_p95_ms"],
                    "encode_p95_ms": result["encode_p95_ms"],
                    "notes": notes,
                    "created_at": created_at,
                }
            )

        for chunk_variant in ("C1", "C2", "C3", "C4"):
            rows.append(
                {
                    "run_id": run_id,
                    "variant_id": chunk_variant,
                    "embedding_model_id": embedding_cfg.model_id,
                    "embedding_dim": embedding_cfg.embedding_dim,
                    "normalize_embeddings": _normalize_label(embedding_cfg.normalize_embeddings),
                    "status": "pending-chunk-index",
                    "num_queries": len(queries),
                    "Recall@20": None,
                    "nDCG@10": None,
                    "MRR@10": None,
                    "citation_valid_rate": None,
                    "latency_p95_ms": None,
                    "encode_p95_ms": None,
                    "notes": "Requires dedicated chunked index collection for fair comparison.",
                    "created_at": created_at,
                }
            )

    _append_rows(output_path, rows)
    print(f"[DONE] Benchmark run stored: {run_id}")
    print(f"[DONE] Output: {output_path}")


if __name__ == "__main__":
    main()
