"""
Safety-first production indexing orchestration for legal hybrid retrieval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import embed_query, embed_texts, get_embedding_dimension
from src.legal_chunker_v2 import LegalChunkerV2
from src.qdrant_store import QdrantLegalStore, collection_name_for_model, model_slug

PRIMARY_MODEL = "truro7/vn-law-embedding"
FALLBACK_MODEL = "huyydangg/DEk21_hcmute_embedding"
MODEL_BATCH = {PRIMARY_MODEL: 32, FALLBACK_MODEL: 24}
HYBRID_DENSE_WEIGHT = 0.6
HYBRID_SPARSE_WEIGHT = 0.4
DEFAULT_FILTER = {"is_active": True}


@dataclass
class ModelRunResult:
    model_id: str
    dense_dim: int
    collection_name: str
    stage: str
    docs_total: int
    chunks_total: int
    quality: Dict[str, float]
    bm25_path: str
    bm25_checksum: str
    bm25_seconds: float
    encode_seconds: float
    upsert_seconds: float
    total_seconds: float
    end_to_end_chunks_per_s: float
    eta_full_hours: float
    retries: int
    timeouts: int
    errors: int
    upsert_calls: int
    retry_rate: float
    timeout_rate: float
    truncation_rate: Optional[float]
    token_len_p50: Optional[float]
    token_len_p95: Optional[float]
    max_seq_length: Optional[int]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(value: str = "") -> str:
    if value and value.strip():
        text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
        return text.strip("_")
    return _now_utc().strftime("run_%Y%m%d_%H%M%S")


def _load_records(path: Path, source_set: str, expected_docs: int) -> List[Dict]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    selected = df
    source_key = str(source_set or "").strip().lower()
    if source_key:
        if "source_set" in df.columns:
            selected = df[df["source_set"].astype(str).str.lower() == source_key]
        elif source_key != "primary":
            raise RuntimeError(f"source_set column not found; cannot filter source-set={source_key}")

    records = selected.to_dict("records")
    if expected_docs > 0 and len(records) != expected_docs:
        raise RuntimeError(f"Expected {expected_docs} docs but loaded {len(records)} docs")
    return records


def _apply_chunk_policy(chunks: List, chunk_policy: str) -> List:
    policy = str(chunk_policy or "c3").strip().lower()
    if policy != "c3":
        return chunks

    selected: List = []
    by_doc: Dict[str, List] = {}
    for chunk in chunks:
        doc_id = str(getattr(chunk, "doc_id", "") or "UNKNOWN")
        by_doc.setdefault(doc_id, []).append(chunk)

    for doc_chunks in by_doc.values():
        structured = []
        for chunk in doc_chunks:
            chunk_type = str(getattr(chunk, "chunk_type", "") or "")
            segment_level = str(getattr(chunk, "segment_level", "") or "").lower()
            if chunk_type == "ProvisionChunk":
                structured.append(chunk)
            elif chunk_type == "ClauseChunk" and segment_level == "clause":
                structured.append(chunk)
        selected.extend(structured if structured else doc_chunks)
    return selected


def _chunk_count_for_record(chunker: LegalChunkerV2, record: Dict, doc_index: int, chunk_policy: str) -> int:
    chunks = chunker.chunk_document(record, doc_index=doc_index)
    chunks = _apply_chunk_policy(chunks, chunk_policy)
    return len(chunks)


def _compute_doc_chunk_stats(records: List[Dict], chunk_policy: str) -> List[Tuple[int, int]]:
    chunker = LegalChunkerV2()
    stats: List[Tuple[int, int]] = []
    for idx, record in enumerate(records):
        stats.append((idx, _chunk_count_for_record(chunker, record, idx, chunk_policy)))
    return stats


def _pick_even(items: Sequence[int], n: int) -> List[int]:
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    result = []
    last = -1
    for i in range(n):
        pos = round(i * (len(items) - 1) / (n - 1))
        if pos <= last:
            pos = min(last + 1, len(items) - 1)
        result.append(items[pos])
        last = pos
    return result


def _select_stratified_indices(
    doc_chunk_stats: List[Tuple[int, int]],
    short_n: int,
    medium_n: int,
    long_n: int,
) -> List[int]:
    ordered = sorted(doc_chunk_stats, key=lambda x: x[1])
    total = len(ordered)
    if total == 0:
        return []

    third = max(1, total // 3)
    short_pool = [idx for idx, _ in ordered[:third]]
    medium_pool = [idx for idx, _ in ordered[third : min(total, third * 2)]]
    long_pool = [idx for idx, _ in ordered[min(total, third * 2) :]]

    picked = []
    picked.extend(_pick_even(short_pool, short_n))
    picked.extend(_pick_even(medium_pool, medium_n))
    picked.extend(_pick_even(long_pool, long_n))
    return sorted(list(dict.fromkeys(picked)))


def _is_cross_dieu(chunk) -> bool:
    import re

    own = str(getattr(chunk, "dieu_so", "") or "").strip().lower()
    if not own:
        return False
    content = str(getattr(chunk, "content", "") or "")
    matches = re.findall(r"^\s*(?:Điều|Dieu)\s+(\d+[A-Za-z0-9\-]*)\b", content, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return False
    return str(matches[0]).strip().lower() != own


def _compute_quality(chunks: List) -> Dict[str, float]:
    total = len(chunks)
    if total <= 0:
        return {
            "provision_id_valid": 0.0,
            "structured_chunk_rate": 0.0,
            "citation_full_rate": 0.0,
            "non_cross_dieu_rate": 0.0,
        }
    provision = sum(1 for c in chunks if str(getattr(c, "provision_id", "") or "").strip())
    structured = sum(1 for c in chunks if str(getattr(c, "chunk_type", "") or "") in {"ProvisionChunk", "ClauseChunk"})
    citation = sum(1 for c in chunks if str(getattr(c, "citation_full", "") or "").strip())
    non_cross = sum(1 for c in chunks if not _is_cross_dieu(c))
    return {
        "provision_id_valid": provision / total,
        "structured_chunk_rate": structured / total,
        "citation_full_rate": citation / total,
        "non_cross_dieu_rate": non_cross / total,
    }


def _json_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _get_model_max_seq_len(model_id: str) -> Optional[int]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        value = int(getattr(tokenizer, "model_max_length", 0) or 0)
        if value <= 0 or value > 1000000:
            return None
        return value
    except Exception:
        return None


def _token_lengths(texts: List[str], model_id: str) -> List[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    lengths: List[int] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        input_ids = encoded.get("input_ids", [])
        for row in input_ids:
            lengths.append(len(row))
    return lengths


def _pctl(values: List[int], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * q)
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def _estimate_eta_hours(
    total_chunks_full: int,
    bm25_seconds: float,
    end_to_end_chunks_per_s: float,
    retry_overhead_factor: float,
    safety_factor: float,
) -> float:
    if total_chunks_full <= 0 or end_to_end_chunks_per_s <= 0:
        return float("inf")
    runtime = bm25_seconds + (total_chunks_full / end_to_end_chunks_per_s)
    runtime *= float(retry_overhead_factor)
    runtime *= float(safety_factor)
    return runtime / 3600.0


def _run_smoke_queries(
    store: QdrantLegalStore,
    model_id: str,
    normalize_embeddings: bool,
) -> Tuple[int, int]:
    queries = [
        "điều kiện kết hôn theo pháp luật việt nam",
        "thời hạn khiếu nại hành chính",
        "xử phạt vi phạm giao thông đường bộ",
    ]
    hits_total = 0
    passed = 0
    for q in queries:
        emb = embed_query(q, model_name=model_id, normalize_embeddings=normalize_embeddings)
        hits = store.hybrid_search(
            query=q,
            query_embedding=emb,
            top_k=5,
            dense_weight=HYBRID_DENSE_WEIGHT,
            sparse_weight=HYBRID_SPARSE_WEIGHT,
            filter_conditions=DEFAULT_FILTER,
        )
        if hits:
            passed += 1
            hits_total += len(hits)
    return passed, hits_total


def _index_model(
    model_id: str,
    records: List[Dict],
    total_chunks_full: int,
    stage: str,
    run_id: str,
    run_dir: Path,
    qdrant_url: str,
    qdrant_api_key: str,
    collection_base: str,
    chunk_policy: str,
    doc_batch_size: int,
    upsert_ladder: List[int],
    checkpoint_doc_interval: int,
    retry_overhead_factor: float,
    safety_factor: float,
    analyze_truncation: bool,
) -> ModelRunResult:
    normalize_embeddings = True
    dense_dim = int(get_embedding_dimension(model_id))
    model_batch_size = int(MODEL_BATCH.get(model_id, 16))
    base_collection = collection_name_for_model(collection_base, model_id, dense_dim)
    shadow_collection = f"{base_collection}__run_{run_id}__{stage}"
    bm25_path = run_dir / "bm25" / f"{shadow_collection}.json"
    observability_path = run_dir / "logs" / "observability.jsonl"

    store = QdrantLegalStore(
        collection_name=shadow_collection,
        dense_dim=dense_dim,
        use_cloud=True,
        cloud_url=qdrant_url,
        api_key=qdrant_api_key,
        bm25_path=str(bm25_path),
    )
    store.create_collection(force_recreate=True)

    chunker = LegalChunkerV2()
    all_chunks_for_quality: List = []
    bm25_start = time.time()
    chunks_seen = 0

    for start in range(0, len(records), doc_batch_size):
        batch_records = records[start : start + doc_batch_size]
        batch_chunks: List = []
        for offset, record in enumerate(batch_records):
            batch_chunks.extend(
                _apply_chunk_policy(
                    chunker.chunk_document(record, doc_index=start + offset),
                    chunk_policy,
                )
            )
        if not batch_chunks:
            continue
        contents = [str(getattr(c, "content", "") or "") for c in batch_chunks]
        store.bm25_encoder.partial_fit(contents)
        all_chunks_for_quality.extend(batch_chunks)
        chunks_seen += len(batch_chunks)

    if chunks_seen <= 0:
        raise RuntimeError(f"No chunks produced for model={model_id} stage={stage}")

    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    store.bm25_encoder.save(str(bm25_path))
    bm25_seconds = time.time() - bm25_start
    bm25_checksum = _json_sha256(bm25_path)
    quality = _compute_quality(all_chunks_for_quality)

    if quality["citation_full_rate"] < 1.0:
        raise RuntimeError(f"citation_full_rate < 100% for model={model_id}")
    if quality["non_cross_dieu_rate"] < 1.0:
        raise RuntimeError(f"non_cross_dieu_rate < 100% for model={model_id}")

    token_lengths: List[int] = []
    max_seq_len = _get_model_max_seq_len(model_id) if analyze_truncation else None
    trunc_count = 0
    encode_seconds = 0.0
    upsert_seconds = 0.0
    retries = 0
    timeouts = 0
    errors = 0
    upsert_calls = 0

    phase2_start = time.time()
    docs_done = 0
    chunks_done = 0

    for start in range(0, len(records), doc_batch_size):
        batch_records = records[start : start + doc_batch_size]
        batch_chunks: List = []
        for offset, record in enumerate(batch_records):
            batch_chunks.extend(
                _apply_chunk_policy(
                    chunker.chunk_document(record, doc_index=start + offset),
                    chunk_policy,
                )
            )
        docs_done += len(batch_records)
        if not batch_chunks:
            continue

        for chunk in batch_chunks:
            setattr(chunk, "encoder_profile", f"{model_slug(model_id)}|norm=true")

        texts = [str(getattr(c, "content", "") or "") for c in batch_chunks]
        if analyze_truncation:
            lengths = _token_lengths(texts, model_id)
            token_lengths.extend(lengths)
            if max_seq_len:
                trunc_count += sum(1 for n in lengths if n > max_seq_len)

        encode_start = time.time()
        vectors = embed_texts(
            texts,
            batch_size=model_batch_size,
            model_name=model_id,
            normalize_embeddings=normalize_embeddings,
            text_type="passage",
            add_prefix=True,
        )
        encode_elapsed = max(time.time() - encode_start, 1e-9)
        encode_seconds += encode_elapsed

        upsert_start = time.time()
        upsert_metrics = store.add_documents(
            chunks=batch_chunks,
            embeddings=vectors,
            batch_size=upsert_ladder[0],
            fit_bm25=False,
            upsert_batch_ladder=upsert_ladder,
            max_retries_per_batch=3,
            max_consecutive_failures=3,
        )
        upsert_elapsed = max(time.time() - upsert_start, 1e-9)
        upsert_seconds += upsert_elapsed

        batch_chunk_count = len(batch_chunks)
        chunks_done += batch_chunk_count
        retries += int(upsert_metrics.get("retries", 0))
        timeouts += int(upsert_metrics.get("timeouts", 0))
        errors += int(upsert_metrics.get("errors", 0))
        upsert_calls += int(upsert_metrics.get("upsert_calls", 0))

        end_to_end_rate = chunks_done / max(time.time() - phase2_start, 1e-9)
        eta_hours = _estimate_eta_hours(
            total_chunks_full=total_chunks_full,
            bm25_seconds=bm25_seconds,
            end_to_end_chunks_per_s=end_to_end_rate,
            retry_overhead_factor=retry_overhead_factor,
            safety_factor=safety_factor,
        )
        _append_jsonl(
            observability_path,
            {
                "ts_utc": _now_utc().isoformat(),
                "run_id": run_id,
                "stage": stage,
                "model_id": model_id,
                "collection_name": shadow_collection,
                "docs_done": docs_done,
                "docs_total": len(records),
                "chunks_done": chunks_done,
                "encode_rate": batch_chunk_count / encode_elapsed,
                "upsert_rate": batch_chunk_count / upsert_elapsed,
                "end_to_end_chunks_per_s": end_to_end_rate,
                "retries": retries,
                "timeouts": timeouts,
                "errors": errors,
                "upsert_calls": upsert_calls,
                "eta_full_hours": eta_hours,
            },
        )

        if checkpoint_doc_interval > 0 and (docs_done % checkpoint_doc_interval == 0 or docs_done >= len(records)):
            points_count = store.point_count_via_scroll(shadow_collection)
            drift = abs(points_count - chunks_done) / max(chunks_done, 1)
            if drift > 0.001:
                raise RuntimeError(
                    f"Point count drift too high at checkpoint docs={docs_done}: points={points_count}, expected={chunks_done}"
                )

    total_seconds = time.time() - bm25_start
    end_to_end_chunks_per_s = chunks_done / max(total_seconds, 1e-9)
    eta_full_hours = _estimate_eta_hours(
        total_chunks_full=total_chunks_full,
        bm25_seconds=bm25_seconds,
        end_to_end_chunks_per_s=end_to_end_chunks_per_s,
        retry_overhead_factor=retry_overhead_factor,
        safety_factor=safety_factor,
    )
    retry_rate = retries / max(upsert_calls, 1)
    timeout_rate = timeouts / max(upsert_calls, 1)
    trunc_rate = None
    if analyze_truncation and token_lengths and max_seq_len:
        trunc_rate = trunc_count / len(token_lengths)

    smoke_passed, smoke_hits = _run_smoke_queries(store, model_id=model_id, normalize_embeddings=normalize_embeddings)
    _append_jsonl(
        observability_path,
        {
            "ts_utc": _now_utc().isoformat(),
            "run_id": run_id,
            "stage": stage,
            "model_id": model_id,
            "collection_name": shadow_collection,
            "event": "smoke",
            "smoke_passed": smoke_passed,
            "smoke_hits": smoke_hits,
        },
    )

    return ModelRunResult(
        model_id=model_id,
        dense_dim=dense_dim,
        collection_name=shadow_collection,
        stage=stage,
        docs_total=len(records),
        chunks_total=chunks_done,
        quality=quality,
        bm25_path=str(bm25_path),
        bm25_checksum=bm25_checksum,
        bm25_seconds=bm25_seconds,
        encode_seconds=encode_seconds,
        upsert_seconds=upsert_seconds,
        total_seconds=total_seconds,
        end_to_end_chunks_per_s=end_to_end_chunks_per_s,
        eta_full_hours=eta_full_hours,
        retries=retries,
        timeouts=timeouts,
        errors=errors,
        upsert_calls=upsert_calls,
        retry_rate=retry_rate,
        timeout_rate=timeout_rate,
        truncation_rate=trunc_rate,
        token_len_p50=_pctl(token_lengths, 0.5) if token_lengths else None,
        token_len_p95=_pctl(token_lengths, 0.95) if token_lengths else None,
        max_seq_length=max_seq_len,
    )


def _run_benchmark(
    run_id: str,
    run_dir: Path,
    queries_path: Path,
    labels_path: Path,
    max_queries: int,
    collection_base: str,
    collection_suffix: str,
    bm25_manifest_path: Path,
    qdrant_url: str,
    qdrant_api_key: str,
) -> Path:
    output_path = run_dir / "benchmark_results.csv"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_research_benchmark_v2.py"),
        "--queries-path",
        str(queries_path),
        "--labels-path",
        str(labels_path),
        "--output-path",
        str(output_path),
        "--max-queries",
        str(max_queries),
        "--top-k",
        "20",
        "--embedding-models",
        f"{PRIMARY_MODEL},{FALLBACK_MODEL}",
        "--normalize-options",
        "true",
        "--collection-base",
        collection_base,
        "--collection-by-model",
        "--collection-suffix",
        collection_suffix,
        "--bm25-manifest",
        str(bm25_manifest_path),
        "--qdrant-url",
        str(qdrant_url),
        "--qdrant-api-key",
        str(qdrant_api_key),
    ]
    env = os.environ.copy()
    # Pilot phase already loads both models; force benchmark to reuse local cache
    # and avoid transient Hugging Face network calls.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    subprocess.run(cmd, check=True, env=env)
    return output_path


def _load_variant_row(path: Path, run_id: str, variant_id: str, model_id: str) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                str(row.get("run_id", "")).strip() == run_id
                and str(row.get("variant_id", "")).strip() == variant_id
                and str(row.get("embedding_model_id", "")).strip() == model_id
            ):
                return row
    return None


def _as_float(row: Dict[str, str], key: str) -> Optional[float]:
    if not row:
        return None
    text = str(row.get(key, "")).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _gate_benchmark(candidate_row: Dict[str, str], baseline_row: Dict[str, str]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    cand_citation = _as_float(candidate_row, "citation_valid_rate")
    base_citation = _as_float(baseline_row, "citation_valid_rate")
    cand_ndcg = _as_float(candidate_row, "nDCG@10")
    base_ndcg = _as_float(baseline_row, "nDCG@10")
    cand_recall = _as_float(candidate_row, "Recall@20")
    base_recall = _as_float(baseline_row, "Recall@20")
    cand_latency = _as_float(candidate_row, "latency_p95_ms")
    base_latency = _as_float(baseline_row, "latency_p95_ms")

    if cand_citation is None or base_citation is None:
        reasons.append("missing citation_valid_rate for benchmark gating")
    else:
        if cand_citation < base_citation:
            reasons.append(f"citation_valid_rate regression: cand={cand_citation:.4f} < base={base_citation:.4f}")
        if cand_citation < 0.95:
            reasons.append(f"citation_valid_rate below 95%: cand={cand_citation:.4f}")

    if cand_ndcg is None or base_ndcg is None:
        reasons.append("missing nDCG@10 for benchmark gating")
    elif cand_ndcg < (base_ndcg - 0.01):
        reasons.append(f"nDCG@10 too low: cand={cand_ndcg:.4f}, base={base_ndcg:.4f}")

    if cand_recall is None or base_recall is None:
        reasons.append("missing Recall@20 for benchmark gating")
    elif cand_recall < (base_recall - 0.01):
        reasons.append(f"Recall@20 too low: cand={cand_recall:.4f}, base={base_recall:.4f}")

    if cand_latency is None or base_latency is None:
        reasons.append("missing latency_p95_ms for benchmark gating")
    elif base_latency > 0 and cand_latency > base_latency * 1.2:
        reasons.append(f"latency_p95 exceeded +20%: cand={cand_latency:.1f} > base*1.2={base_latency*1.2:.1f}")

    return (len(reasons) == 0), reasons


def _write_manifest(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_progress_log(progress_path: Path, line: str):
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run safe hybrid index production workflow")
    parser.add_argument("--input-path", default="data/gold/vbpl_gold.parquet")
    parser.add_argument("--source-set", default="primary")
    parser.add_argument("--expected-docs", type=int, default=1255)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-root", default="data/gold/research/index_runs")
    parser.add_argument("--collection-base", default="legal_documents")
    parser.add_argument("--chunk-policy", choices=("c3", "raw"), default="c3")
    parser.add_argument("--pilot-docs", type=int, default=100)
    parser.add_argument("--doc-batch-size", type=int, default=50)
    parser.add_argument("--upsert-ladder", default="500,250,100")
    parser.add_argument("--checkpoint-doc-interval", type=int, default=50)
    parser.add_argument("--retry-overhead-factor", type=float, default=1.2)
    parser.add_argument("--safety-factor", type=float, default=1.2)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-api-key", required=True)
    parser.add_argument("--queries-path", default="data/gold/research/query_set_v1.csv")
    parser.add_argument("--labels-path", default="data/gold/research/eval_pairs_template.csv")
    parser.add_argument("--benchmark-max-queries", type=int, default=60)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--enable-alias-swap", action="store_true")
    args = parser.parse_args()

    run_id = _run_id(args.run_id)
    run_root = Path(args.run_root)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(Path(args.input_path), args.source_set, args.expected_docs)
    upsert_ladder = [int(x.strip()) for x in str(args.upsert_ladder).split(",") if x.strip()]
    if not upsert_ladder:
        raise RuntimeError("upsert ladder is empty")

    doc_chunk_stats = _compute_doc_chunk_stats(records, args.chunk_policy)
    total_chunks_full = sum(count for _, count in doc_chunk_stats)
    _write_manifest(
        run_dir / "doc_chunk_stats.json",
        {
            "run_id": run_id,
            "docs_total": len(records),
            "chunks_total_est": total_chunks_full,
            "chunk_policy": args.chunk_policy,
            "doc_chunk_stats": [{"doc_index": idx, "chunk_count": cnt} for idx, cnt in doc_chunk_stats],
        },
    )

    pilot_indices = _select_stratified_indices(
        doc_chunk_stats=doc_chunk_stats,
        short_n=40,
        medium_n=40,
        long_n=20,
    )
    if len(pilot_indices) > args.pilot_docs:
        pilot_indices = pilot_indices[: args.pilot_docs]
    pilot_records = [records[idx] for idx in pilot_indices]
    if len(pilot_records) < min(30, args.pilot_docs):
        raise RuntimeError(f"Pilot records too small after stratification: {len(pilot_records)}")

    manifest_collections: Dict[str, Dict[str, str]] = {}
    model_results: Dict[str, ModelRunResult] = {}
    for model_id in (PRIMARY_MODEL, FALLBACK_MODEL):
        result = _index_model(
            model_id=model_id,
            records=pilot_records,
            total_chunks_full=total_chunks_full,
            stage="pilot",
            run_id=run_id,
            run_dir=run_dir,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            collection_base=args.collection_base,
            chunk_policy=args.chunk_policy,
            doc_batch_size=args.doc_batch_size,
            upsert_ladder=upsert_ladder,
            checkpoint_doc_interval=args.checkpoint_doc_interval,
            retry_overhead_factor=args.retry_overhead_factor,
            safety_factor=args.safety_factor,
            analyze_truncation=True,
        )
        model_results[model_id] = result
        manifest_collections[result.collection_name] = {
            "model_id": result.model_id,
            "stage": result.stage,
            "bm25_path": result.bm25_path,
            "bm25_checksum": result.bm25_checksum,
        }

    bm25_manifest_path = run_dir / "bm25_manifest.json"
    _write_manifest(
        bm25_manifest_path,
        {
            "run_id": run_id,
            "generated_at": _now_utc().isoformat(),
            "collections": manifest_collections,
        },
    )

    for model_id, result in model_results.items():
        if result.timeout_rate >= 0.01:
            raise RuntimeError(f"Pilot timeout_rate failed for {model_id}: {result.timeout_rate:.4f} >= 0.01")
        if result.retry_rate >= 0.05:
            raise RuntimeError(f"Pilot retry_rate failed for {model_id}: {result.retry_rate:.4f} >= 0.05")
        if result.eta_full_hours > 4.0:
            raise RuntimeError(f"Pilot ETA failed for {model_id}: eta_full_hours={result.eta_full_hours:.2f} > 4.0h")

    benchmark_path = None
    if not args.skip_benchmark:
        benchmark_path = _run_benchmark(
            run_id=run_id,
            run_dir=run_dir,
            queries_path=Path(args.queries_path),
            labels_path=Path(args.labels_path),
            max_queries=args.benchmark_max_queries,
            collection_base=args.collection_base,
            collection_suffix=f"__run_{run_id}__pilot",
            bm25_manifest_path=bm25_manifest_path,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
        )
        truro_row = _load_variant_row(benchmark_path, run_id, "B3", PRIMARY_MODEL)
        dek21_row = _load_variant_row(benchmark_path, run_id, "B3", FALLBACK_MODEL)
        if not truro_row or not dek21_row:
            raise RuntimeError("Missing benchmark rows for B3 gating")
        passed, reasons = _gate_benchmark(truro_row, dek21_row)
        if not passed:
            raise RuntimeError("Benchmark gate failed for primary model: " + "; ".join(reasons))

    winner_model = PRIMARY_MODEL
    full_result: Optional[ModelRunResult] = None
    if not args.skip_full:
        full_result = _index_model(
            model_id=winner_model,
            records=records,
            total_chunks_full=total_chunks_full,
            stage="full",
            run_id=run_id,
            run_dir=run_dir,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            collection_base=args.collection_base,
            chunk_policy=args.chunk_policy,
            doc_batch_size=args.doc_batch_size,
            upsert_ladder=upsert_ladder,
            checkpoint_doc_interval=args.checkpoint_doc_interval,
            retry_overhead_factor=args.retry_overhead_factor,
            safety_factor=args.safety_factor,
            analyze_truncation=False,
        )
        manifest_collections[full_result.collection_name] = {
            "model_id": full_result.model_id,
            "stage": full_result.stage,
            "bm25_path": full_result.bm25_path,
            "bm25_checksum": full_result.bm25_checksum,
        }
        _write_manifest(
            bm25_manifest_path,
            {
                "run_id": run_id,
                "generated_at": _now_utc().isoformat(),
                "collections": manifest_collections,
            },
        )

    cutover_info = {"alias_name": "", "old_collection": "", "new_collection": "", "swapped": False}
    if args.enable_alias_swap and full_result is not None:
        alias_name = f"{args.collection_base}__prod__{model_slug(full_result.model_id)}__d{full_result.dense_dim}"
        store = QdrantLegalStore(
            collection_name=full_result.collection_name,
            dense_dim=full_result.dense_dim,
            use_cloud=True,
            cloud_url=args.qdrant_url,
            api_key=args.qdrant_api_key,
            bm25_path=full_result.bm25_path,
        )
        old_collection = store.resolve_alias(alias_name) or ""
        store.swap_alias(alias_name, full_result.collection_name)
        cutover_info = {
            "alias_name": alias_name,
            "old_collection": old_collection,
            "new_collection": full_result.collection_name,
            "swapped": True,
        }

        serving_bm25 = PROJECT_ROOT / "data" / "qdrant_db" / "bm25_encoder.json"
        serving_bm25.parent.mkdir(parents=True, exist_ok=True)
        if serving_bm25.exists():
            backup = serving_bm25.with_name(f"bm25_encoder.backup.{run_id}.json")
            shutil.copy2(serving_bm25, backup)
        shutil.copy2(full_result.bm25_path, serving_bm25)

    summary = {
        "run_id": run_id,
        "generated_at": _now_utc().isoformat(),
        "input_path": args.input_path,
        "source_set": args.source_set,
        "docs_total": len(records),
        "chunks_total_est": total_chunks_full,
        "pilot_records": len(pilot_records),
        "models": {
            model_id: {
                "collection_name": res.collection_name,
                "dense_dim": res.dense_dim,
                "eta_full_hours": res.eta_full_hours,
                "end_to_end_chunks_per_s": res.end_to_end_chunks_per_s,
                "retry_rate": res.retry_rate,
                "timeout_rate": res.timeout_rate,
                "truncation_rate": res.truncation_rate,
                "token_len_p50": res.token_len_p50,
                "token_len_p95": res.token_len_p95,
                "max_seq_length": res.max_seq_length,
            }
            for model_id, res in model_results.items()
        },
        "winner_model": winner_model,
        "full_collection": full_result.collection_name if full_result else "",
        "benchmark_output": str(benchmark_path) if benchmark_path else "",
        "bm25_manifest": str(bm25_manifest_path),
        "cutover": cutover_info,
    }
    _write_manifest(run_dir / "run_summary.json", summary)

    progress_log = PROJECT_ROOT / "data" / "gold" / "research" / "embedding_graph_execution_progress_v3.md"
    if progress_log.exists():
        _append_progress_log(
            progress_log,
            f"| 2026-03-12 | codex | A14 | hybrid production run `{run_id}` | `scripts/run_hybrid_index_production.py` | pilot completed for 2 models, winner=`{winner_model}` | none | full run/cutover follow summary `{run_dir / 'run_summary.json'}` |",
        )

    print(f"[DONE] run_id={run_id}")
    print(f"[DONE] summary={run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
