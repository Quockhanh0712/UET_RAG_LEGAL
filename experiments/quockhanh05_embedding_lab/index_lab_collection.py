from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.quockhanh05_embedding_lab.config import (  # noqa: E402
    BM25_PATH,
    COLLECTION_NAME,
    DATA_PATH,
    DENSE_DIM,
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    HARD_CAP_TOKENS,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    NORMALIZE_EMBEDDINGS,
    OVERLAP_TOKENS,
    TARGET_CHUNK_TOKENS,
    apply_lab_environment,
    domain_counts,
    matched_domains,
)
from src.embeddings import (  # noqa: E402
    count_text_tokens,
    embed_query,
    embed_texts,
    get_embedding_dimension,
    get_model,
)
from src.legal_chunker_v2 import LegalChunkerV2  # noqa: E402
from src.qdrant_store import QdrantLegalStore, model_slug  # noqa: E402


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


def _token_count(text: str) -> int:
    return int(
        count_text_tokens(
            text,
            model_name=MODEL_ID,
            text_type="passage",
            add_prefix=True,
        )
    )


def _load_dataframe(path: Path, source_set: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    source_set = str(source_set or "").strip().lower()
    if source_set and "source_set" in df.columns:
        df = df[df["source_set"].astype(str).str.lower() == source_set]
    elif source_set and source_set != "primary":
        raise ValueError(f"Dataset has no source_set column for source_set={source_set!r}.")
    return df.reset_index(drop=True)


def _row_match_text(row: pd.Series) -> str:
    parts = [
        str(row.get("ten_van_ban") or ""),
        str(row.get("law_family") or ""),
        str(row.get("mo_ta") or ""),
    ]
    return " ".join(parts)


def _select_lab_records(df: pd.DataFrame, max_docs: int, max_docs_per_domain: int) -> List[Dict]:
    rows: List[tuple[int, List[str]]] = []
    for index, row in df.iterrows():
        domains = matched_domains(_row_match_text(row))
        if domains:
            rows.append((index, domains))

    counts = domain_counts(domains for _, domains in rows)
    print("[SELECT] matched rows by domain:")
    for domain in DOMAIN_ORDER:
        print(f"  - {domain}: {counts[domain]}")

    max_docs = max(0, int(max_docs))
    if max_docs_per_domain <= 0 and max_docs > 0:
        max_docs_per_domain = max(1, math.ceil(max_docs / len(DOMAIN_ORDER)))

    selected_indices: List[int] = []
    selected = set()
    domains_by_index = {index: domains for index, domains in rows}

    for domain in DOMAIN_ORDER:
        used_for_domain = 0
        for index, domains in rows:
            if domain not in domains or index in selected:
                continue
            selected.add(index)
            selected_indices.append(index)
            used_for_domain += 1
            if max_docs_per_domain > 0 and used_for_domain >= max_docs_per_domain:
                break
            if max_docs > 0 and len(selected_indices) >= max_docs:
                break
        if max_docs > 0 and len(selected_indices) >= max_docs:
            break

    if max_docs <= 0 or len(selected_indices) < max_docs:
        for index, _ in rows:
            if index in selected:
                continue
            selected.add(index)
            selected_indices.append(index)
            if max_docs > 0 and len(selected_indices) >= max_docs:
                break

    records: List[Dict] = []
    for index in selected_indices:
        record = df.loc[index].to_dict()
        record["encoder_profile"] = (
            f"{model_slug(MODEL_ID)}|norm=true|max_seq={MAX_SEQ_LENGTH}|target={TARGET_CHUNK_TOKENS}"
        )
        record["lab_domains"] = ",".join(domains_by_index.get(index, []))
        records.append(record)
    print(f"[SELECT] selected_docs={len(records)} max_docs={max_docs or 'all'}")
    return records


def _apply_c3_policy(chunks: Sequence[object]) -> List[object]:
    by_doc: Dict[str, List[object]] = {}
    for chunk in chunks:
        by_doc.setdefault(str(getattr(chunk, "doc_id", "") or "UNKNOWN"), []).append(chunk)

    selected: List[object] = []
    for doc_chunks in by_doc.values():
        provision_chunks = [
            chunk
            for chunk in doc_chunks
            if str(getattr(chunk, "chunk_type", "") or "") == "ProvisionChunk"
        ]
        selected.extend(provision_chunks or doc_chunks)
    return selected


def _split_text_by_token_budget(text: str, max_tokens: int) -> List[str]:
    words = str(text or "").split()
    if not words:
        return []

    parts: List[str] = []
    start = 0
    while start < len(words):
        low = start + 1
        high = len(words)
        best = start + 1
        while low <= high:
            mid = (low + high) // 2
            candidate = " ".join(words[start:mid])
            if _token_count(candidate) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best <= start:
            best = start + 1
        parts.append(" ".join(words[start:best]))
        start = best
    return [part for part in parts if part.strip()]


def _enforce_token_budget(chunks: Sequence[object], max_tokens: int) -> List[object]:
    final_chunks: List[object] = []
    split_count = 0

    for chunk in chunks:
        content = str(getattr(chunk, "content", "") or "")
        token_count = _token_count(content)
        if token_count <= max_tokens:
            setattr(chunk, "token_len_model", token_count)
            final_chunks.append(chunk)
            continue

        parts = _split_text_by_token_budget(content, max_tokens)
        split_count += 1
        for part_index, part in enumerate(parts, start=1):
            part_tokens = _token_count(part)
            if part_tokens > max_tokens:
                raise RuntimeError(
                    f"Unable to split chunk within max_seq_length={max_tokens}: "
                    f"chunk_id={getattr(chunk, 'chunk_id', '')} tokens={part_tokens}"
                )
            new_chunk = replace(chunk)
            new_chunk.content = part
            new_chunk.chunk_id = f"{getattr(chunk, 'chunk_id', '')}__labseg_{part_index}"
            new_chunk.token_len_est = len(part.split())
            new_chunk.token_len_model = part_tokens
            new_chunk.segment_level = "article_window"
            new_chunk.overflow_index = part_index
            new_chunk.overflow_total = len(parts)
            final_chunks.append(new_chunk)

    if split_count:
        print(f"[CHUNK] safety_split_overlong_chunks={split_count}")
    return final_chunks


def _build_chunks(records: Sequence[Dict], max_chunks: int) -> List[object]:
    chunker = LegalChunkerV2(
        encoder_model_name=MODEL_ID,
        encoder_max_len=MAX_SEQ_LENGTH,
        target_chunk_tokens=TARGET_CHUNK_TOKENS,
        hard_cap_tokens=HARD_CAP_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
    )

    chunks: List[object] = []
    for index, record in enumerate(records):
        chunks.extend(chunker.chunk_document(record, doc_index=index))
        if (index + 1) % 25 == 0:
            print(f"[CHUNK] docs={index + 1} raw_chunks={len(chunks)}")

    chunks = _apply_c3_policy(chunks)
    chunks = _enforce_token_budget(chunks, MAX_SEQ_LENGTH)
    if max_chunks > 0:
        chunks = chunks[:max_chunks]

    if not chunks:
        raise RuntimeError("No chunks generated for the lab collection.")

    max_observed = max(int(getattr(chunk, "token_len_model", 0) or 0) for chunk in chunks)
    print(f"[CHUNK] final_chunks={len(chunks)} max_token_len_model={max_observed}")
    if max_observed > MAX_SEQ_LENGTH:
        raise RuntimeError(f"Chunk token budget exceeded: {max_observed} > {MAX_SEQ_LENGTH}")
    return chunks


def _iter_batches(items: Sequence[object], batch_size: int) -> Iterable[Sequence[object]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _run_smoke(store: QdrantLegalStore) -> None:
    queries = [
        "toi trom cap tai san theo bo luat hinh su",
        "muc phat vuot den do giao thong duong bo",
        "xu phat xa thai gay o nhiem moi truong",
        "tham quyen cua canh sat bien viet nam",
        "quyen loi nguoi lao dong khi cham dut hop dong",
    ]
    passed = 0
    for query in queries:
        query_vector = embed_query(
            query,
            model_name=MODEL_ID,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )
        hits = store.hybrid_search(
            query=query,
            query_embedding=query_vector,
            top_k=5,
            filter_conditions={"is_active": True},
        )
        print(f"[SMOKE] hits={len(hits)} query={query}")
        if hits:
            passed += 1
    print(f"[SMOKE] passed={passed}/{len(queries)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a quick Qdrant lab collection for a 256-token embedding model.")
    parser.add_argument("--input-path", default=str(DATA_PATH))
    parser.add_argument("--source-set", default="primary")
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--max-docs-per-domain", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--qdrant-url", default="")
    parser.add_argument("--qdrant-api-key", default="")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--upsert-batch-size", type=int, default=64)
    parser.add_argument("--payload-profile", choices=("full", "lean"), default="full")
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    cloud_url, api_key = apply_lab_environment(
        qdrant_url=args.qdrant_url or None,
        qdrant_api_key=args.qdrant_api_key or None,
        require_qdrant=True,
    )
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    _set_sentence_transformer_max_length(MAX_SEQ_LENGTH)
    dense_dim = int(get_embedding_dimension(MODEL_ID))
    if dense_dim != DENSE_DIM:
        raise RuntimeError(f"Unexpected embedding dimension: expected {DENSE_DIM}, got {dense_dim}")

    df = _load_dataframe(Path(args.input_path), source_set=args.source_set)
    records = _select_lab_records(
        df,
        max_docs=args.max_docs,
        max_docs_per_domain=args.max_docs_per_domain,
    )
    chunks = _build_chunks(records, max_chunks=args.max_chunks)

    store = QdrantLegalStore(
        collection_name=COLLECTION_NAME,
        dense_dim=DENSE_DIM,
        use_cloud=True,
        cloud_url=cloud_url,
        api_key=api_key,
        bm25_path=str(BM25_PATH),
        payload_profile=args.payload_profile,
    )
    store.create_collection(force_recreate=bool(args.force_recreate))

    contents = [str(getattr(chunk, "content", "") or "") for chunk in chunks]
    store.bm25_encoder.fit(contents)
    store.bm25_encoder.save(str(BM25_PATH))
    print(f"[BM25] path={BM25_PATH}")

    indexed = 0
    for batch in _iter_batches(chunks, args.upsert_batch_size):
        texts = [str(getattr(chunk, "content", "") or "") for chunk in batch]
        vectors = embed_texts(
            texts,
            batch_size=args.embedding_batch_size,
            model_name=MODEL_ID,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            text_type="passage",
            add_prefix=True,
        )
        store.add_documents(
            chunks=list(batch),
            embeddings=vectors,
            batch_size=args.upsert_batch_size,
            fit_bm25=False,
        )
        indexed += len(batch)
        print(f"[INDEX] indexed_chunks={indexed}/{len(chunks)}")

    stats = store.get_stats()
    print(
        f"[DONE] collection={COLLECTION_NAME} points={stats.get('points_count')} "
        f"dim={stats.get('dense_dim')} status={stats.get('status')} elapsed_s={time.time() - start:.1f}"
    )
    print(f"[DONE] model={MODEL_ID} max_seq_length={MAX_SEQ_LENGTH}")

    if not args.skip_smoke:
        _run_smoke(store)


if __name__ == "__main__":
    main()
