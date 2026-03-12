"""
Upload legal chunks to Qdrant Cloud with production-oriented hybrid checks.

Key behavior:
- Supports full-dataset indexing without loading all embeddings into memory.
- Two-phase indexing:
  1) BM25 global fit over all selected chunks (streamed by doc batch)
  2) GPU embedding + upsert batches using the pre-fitted BM25 encoder
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import embed_query, embed_texts, get_embedding_dimension
from src.legal_chunker_v2 import LegalChunkerV2
from src.qdrant_store import QdrantLegalStore, collection_name_for_model, model_slug

ARTICLE_PATTERN = re.compile(
    r"^\s*(?:\u0110i\u1ec1u|Dieu)\s+(\d+[A-Za-z0-9\-]*)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _load_records(path: Path, limit_docs: int, source_set: str) -> Tuple[List[Dict], int, int]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    input_total = len(df)

    selected_total = input_total
    if source_set:
        source_set = source_set.strip().lower()
        if "source_set" in df.columns:
            df = df[df["source_set"].astype(str).str.lower() == source_set]
            selected_total = len(df)
        elif source_set != "primary":
            raise ValueError(
                f"Dataset has no source_set column but --source-set={source_set} was requested."
            )

    if limit_docs > 0:
        df = df.head(limit_docs)
    return df.to_dict("records"), input_total, selected_total


def _iter_doc_batches(records: List[Dict], doc_batch_size: int):
    if doc_batch_size <= 0:
        doc_batch_size = len(records)
    for start in range(0, len(records), doc_batch_size):
        end = min(start + doc_batch_size, len(records))
        yield start, records[start:end]


def _build_chunks_for_batch(records: List[Dict], doc_start_index: int, chunk_policy: str) -> List:
    chunker = LegalChunkerV2()
    chunks = []
    for offset, record in enumerate(records):
        chunks.extend(chunker.chunk_document(record, doc_index=doc_start_index + offset))
    return _apply_chunk_policy(chunks, chunk_policy=chunk_policy)


def _apply_chunk_policy(chunks: List, chunk_policy: str) -> List:
    policy = str(chunk_policy or "raw").strip().lower()
    if policy != "c3":
        return chunks

    # C3 production profile:
    # - keep article-level ProvisionChunk
    # - keep clause-level ClauseChunk (Khoan)
    # - drop point-level ClauseChunk (Diem) to avoid vector-space dilution
    grouped: Dict[str, List] = defaultdict(list)
    for chunk in chunks:
        grouped[str(getattr(chunk, "doc_id", "") or "UNKNOWN")].append(chunk)

    selected: List = []
    for _, doc_chunks in grouped.items():
        structured = []
        for c in doc_chunks:
            chunk_type = str(getattr(c, "chunk_type", "") or "")
            segment_level = str(getattr(c, "segment_level", "") or "").lower()
            if chunk_type == "ProvisionChunk":
                structured.append(c)
            elif chunk_type == "ClauseChunk" and segment_level == "clause":
                structured.append(c)
        if structured:
            selected.extend(structured)
        else:
            selected.extend(doc_chunks)
    return selected


def _cross_dieu_violation(chunk) -> bool:
    own_dieu = str(getattr(chunk, "dieu_so", "") or "").strip().lower()
    if not own_dieu:
        return False
    matches = [m.lower() for m in ARTICLE_PATTERN.findall(str(getattr(chunk, "content", "") or ""))]
    if not matches:
        return False
    return matches[0] != own_dieu


def _compute_chunk_quality_from_counts(
    total: int,
    provision_id_valid: int,
    structured: int,
    citation_full: int,
    no_cross_dieu: int,
) -> Dict[str, float]:
    if total <= 0:
        return {
            "total_chunks": 0,
            "provision_id_valid_rate": 0.0,
            "structured_chunk_rate": 0.0,
            "citation_full_rate": 0.0,
            "non_cross_dieu_rate": 0.0,
        }
    return {
        "total_chunks": total,
        "provision_id_valid_rate": provision_id_valid / total,
        "structured_chunk_rate": structured / total,
        "citation_full_rate": citation_full / total,
        "non_cross_dieu_rate": no_cross_dieu / total,
    }


def _assert_chunk_quality(
    metrics: Dict[str, float],
    min_provision_rate: float,
    min_structured_rate: float,
    min_citation_rate: float,
    min_non_cross_rate: float,
) -> None:
    failures = []
    if metrics["provision_id_valid_rate"] < min_provision_rate:
        failures.append(
            f"provision_id_valid_rate={metrics['provision_id_valid_rate']:.2%} < {min_provision_rate:.2%}"
        )
    if metrics["structured_chunk_rate"] < min_structured_rate:
        failures.append(
            f"structured_chunk_rate={metrics['structured_chunk_rate']:.2%} < {min_structured_rate:.2%}"
        )
    if metrics["citation_full_rate"] < min_citation_rate:
        failures.append(
            f"citation_full_rate={metrics['citation_full_rate']:.2%} < {min_citation_rate:.2%}"
        )
    if metrics["non_cross_dieu_rate"] < min_non_cross_rate:
        failures.append(
            f"non_cross_dieu_rate={metrics['non_cross_dieu_rate']:.2%} < {min_non_cross_rate:.2%}"
        )
    if failures:
        raise RuntimeError("Chunk quality gate failed: " + "; ".join(failures))


def _parse_normalize(value: str) -> Optional[bool]:
    text = str(value or "auto").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _trim_to_budget(chunks: List, already_used: int, max_chunks: int) -> List:
    if max_chunks <= 0:
        return chunks
    remaining = max_chunks - already_used
    if remaining <= 0:
        return []
    if len(chunks) <= remaining:
        return chunks
    return chunks[:remaining]


def _run_hybrid_smoke(
    store: QdrantLegalStore,
    model_id: str,
    normalize_embeddings: Optional[bool],
    queries: List[str],
) -> Tuple[int, int]:
    passed = 0
    total_hits = 0
    for query in queries:
        query_embedding = embed_query(
            query,
            model_name=model_id,
            normalize_embeddings=normalize_embeddings,
        )
        hits = store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=5,
            dense_weight=0.6,
            sparse_weight=0.4,
            filter_conditions={"is_active": True},
        )
        if hits:
            passed += 1
            total_hits += len(hits)
    return passed, total_hits


def _upload_for_model(
    model_id: str,
    records: List[Dict],
    cloud_url: str,
    api_key: str,
    collection_base: str,
    embedding_batch_size: int,
    upsert_batch_size: int,
    force_recreate: bool,
    normalize_embeddings: Optional[bool],
    run_smoke: bool,
    doc_batch_size: int,
    chunk_policy: str,
    max_chunks: int,
    min_provision_rate: float,
    min_structured_rate: float,
    min_citation_rate: float,
    min_non_cross_rate: float,
) -> Dict[str, str]:
    normalize_label = "auto" if normalize_embeddings is None else ("true" if normalize_embeddings else "false")

    dense_dim = int(get_embedding_dimension(model_id))
    collection_name = collection_name_for_model(collection_base, model_id, dense_dim)

    store = QdrantLegalStore(
        collection_name=collection_name,
        dense_dim=dense_dim,
        use_cloud=True,
        cloud_url=cloud_url,
        api_key=api_key,
    )
    store.create_collection(force_recreate=force_recreate)

    # Phase 1: global BM25 fit in streamed batches.
    total_chunks = 0
    provision_id_valid = 0
    structured_chunks = 0
    citation_full = 0
    no_cross_dieu = 0

    for batch_no, (doc_start, doc_batch) in enumerate(_iter_doc_batches(records, doc_batch_size), start=1):
        batch_chunks = _build_chunks_for_batch(doc_batch, doc_start, chunk_policy=chunk_policy)
        batch_chunks = _trim_to_budget(batch_chunks, total_chunks, max_chunks=max_chunks)
        if not batch_chunks:
            break

        contents = [str(getattr(c, "content", "") or "") for c in batch_chunks]
        store.bm25_encoder.partial_fit(contents)

        total_chunks += len(batch_chunks)
        provision_id_valid += sum(1 for c in batch_chunks if str(getattr(c, "provision_id", "") or "").strip())
        structured_chunks += sum(
            1 for c in batch_chunks if getattr(c, "chunk_type", "") in {"ProvisionChunk", "ClauseChunk"}
        )
        citation_full += sum(1 for c in batch_chunks if str(getattr(c, "citation_full", "") or "").strip())
        no_cross_dieu += sum(1 for c in batch_chunks if not _cross_dieu_violation(c))

        print(
            f"[BM25-FIT] batch={batch_no} docs={len(doc_batch)} chunks_acc={total_chunks} vocab={len(store.bm25_encoder.vocab)}"
        )
        if max_chunks > 0 and total_chunks >= max_chunks:
            break

    if total_chunks <= 0:
        raise RuntimeError("No chunks generated for indexing")

    store.bm25_encoder.save(store.bm25_path)

    quality_metrics = _compute_chunk_quality_from_counts(
        total=total_chunks,
        provision_id_valid=provision_id_valid,
        structured=structured_chunks,
        citation_full=citation_full,
        no_cross_dieu=no_cross_dieu,
    )
    _assert_chunk_quality(
        metrics=quality_metrics,
        min_provision_rate=min_provision_rate,
        min_structured_rate=min_structured_rate,
        min_citation_rate=min_citation_rate,
        min_non_cross_rate=min_non_cross_rate,
    )

    print(
        "[QUALITY] provision_id={:.2%} structured={:.2%} citation_full={:.2%} non_cross_dieu={:.2%}".format(
            quality_metrics["provision_id_valid_rate"],
            quality_metrics["structured_chunk_rate"],
            quality_metrics["citation_full_rate"],
            quality_metrics["non_cross_dieu_rate"],
        )
    )

    # Phase 2: embed + upsert in streamed batches.
    indexed_chunks = 0
    encoder_profile = f"{model_slug(model_id)}|norm={normalize_label}"

    for batch_no, (doc_start, doc_batch) in enumerate(_iter_doc_batches(records, doc_batch_size), start=1):
        batch_chunks = _build_chunks_for_batch(doc_batch, doc_start, chunk_policy=chunk_policy)
        batch_chunks = _trim_to_budget(batch_chunks, indexed_chunks, max_chunks=max_chunks)
        if not batch_chunks:
            break

        for chunk in batch_chunks:
            setattr(chunk, "encoder_profile", encoder_profile)

        texts = [str(getattr(c, "content", "") or "") for c in batch_chunks]
        vectors = embed_texts(
            texts,
            batch_size=embedding_batch_size,
            model_name=model_id,
            normalize_embeddings=normalize_embeddings,
            text_type="passage",
            add_prefix=True,
        )
        store.add_documents(
            chunks=batch_chunks,
            embeddings=vectors,
            batch_size=upsert_batch_size,
            fit_bm25=False,
        )

        indexed_chunks += len(batch_chunks)
        print(f"[INDEX] batch={batch_no} docs={len(doc_batch)} indexed_chunks={indexed_chunks}")

        if max_chunks > 0 and indexed_chunks >= max_chunks:
            break

    stats = store.get_stats()

    smoke_queries = [
        "dieu kien ket hon theo phap luat viet nam",
        "thoi han khieu nai hanh chinh la bao lau",
        "quy dinh xu phat vi pham giao thong duong bo",
    ]
    smoke_passed = 0
    smoke_total_hits = 0
    if run_smoke:
        smoke_passed, smoke_total_hits = _run_hybrid_smoke(
            store=store,
            model_id=model_id,
            normalize_embeddings=normalize_embeddings,
            queries=smoke_queries,
        )

    return {
        "model_id": model_id,
        "collection_name": collection_name,
        "dense_dim": str(dense_dim),
        "points_count": str(stats.get("points_count", "")),
        "status": str(stats.get("status", "")),
        "normalize_embeddings": normalize_label,
        "chunk_policy": chunk_policy,
        "indexed_chunks": str(indexed_chunks),
        "smoke_passed": str(smoke_passed),
        "smoke_total": str(len(smoke_queries) if run_smoke else 0),
        "smoke_hits": str(smoke_total_hits),
        "quality_provision": f"{quality_metrics['provision_id_valid_rate']:.4f}",
        "quality_structured": f"{quality_metrics['structured_chunk_rate']:.4f}",
        "quality_citation": f"{quality_metrics['citation_full_rate']:.4f}",
        "quality_non_cross": f"{quality_metrics['non_cross_dieu_rate']:.4f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload embeddings to Qdrant Cloud")
    parser.add_argument("--input-path", default="data/gold/vbpl_gold.parquet")
    parser.add_argument("--source-set", default="primary", help="Filter source_set when column exists")
    parser.add_argument("--expected-docs", type=int, default=1255)
    parser.add_argument("--limit-docs", type=int, default=0, help="0 means all docs")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means no chunk cap")
    parser.add_argument("--doc-batch-size", type=int, default=20)
    parser.add_argument("--chunk-policy", choices=("c3", "raw"), default="c3")
    parser.add_argument(
        "--models",
        default="AITeamVN/Vietnamese_Embedding",
        help="Comma-separated embedding model ids",
    )
    parser.add_argument("--embedding-normalize", choices=("auto", "true", "false"), default="true")
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-api-key", required=True)
    parser.add_argument("--collection-base", default="legal_documents")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--upsert-batch-size", type=int, default=50)
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--min-provision-rate", type=float, default=0.90)
    parser.add_argument("--min-structured-rate", type=float, default=0.85)
    parser.add_argument("--min-citation-rate", type=float, default=1.0)
    parser.add_argument("--min-non-cross-rate", type=float, default=1.0)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        raise ValueError("No models provided")
    normalize_embeddings = _parse_normalize(args.embedding_normalize)

    records, input_total, selected_total = _load_records(
        input_path,
        limit_docs=args.limit_docs,
        source_set=args.source_set,
    )
    if args.expected_docs > 0 and args.limit_docs == 0 and len(records) != args.expected_docs:
        raise RuntimeError(
            f"Expected {args.expected_docs} docs after selection, got {len(records)} docs."
        )

    print(
        f"[INFO] input_docs={input_total} selected_docs={selected_total} loaded_docs={len(records)} "
        f"chunk_policy={args.chunk_policy} doc_batch_size={args.doc_batch_size} max_chunks={args.max_chunks}"
    )

    results = []
    for model_id in model_ids:
        print(f"[INFO] Uploading model={model_id} normalize={args.embedding_normalize}")
        result = _upload_for_model(
            model_id=model_id,
            records=records,
            cloud_url=args.qdrant_url,
            api_key=args.qdrant_api_key,
            collection_base=args.collection_base,
            embedding_batch_size=args.embedding_batch_size,
            upsert_batch_size=args.upsert_batch_size,
            force_recreate=args.force_recreate,
            normalize_embeddings=normalize_embeddings,
            run_smoke=not args.skip_smoke,
            doc_batch_size=args.doc_batch_size,
            chunk_policy=args.chunk_policy,
            max_chunks=args.max_chunks,
            min_provision_rate=args.min_provision_rate,
            min_structured_rate=args.min_structured_rate,
            min_citation_rate=args.min_citation_rate,
            min_non_cross_rate=args.min_non_cross_rate,
        )
        results.append(result)
        print(
            f"[DONE] model={result['model_id']} collection={result['collection_name']} "
            f"dim={result['dense_dim']} points={result['points_count']} status={result['status']} "
            f"indexed_chunks={result['indexed_chunks']} quality=(p={result['quality_provision']},s={result['quality_structured']},c={result['quality_citation']},x={result['quality_non_cross']}) "
            f"smoke={result['smoke_passed']}/{result['smoke_total']} hits={result['smoke_hits']}"
        )

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    collections = client.get_collections()
    print("[QDRANT] get_collections()")
    print(collections)


if __name__ == "__main__":
    main()
