"""
Upload legal chunks to Qdrant Cloud using selected embedding models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import embed_texts
from src.legal_chunker_v2 import LegalChunkerV2
from src.qdrant_store import QdrantLegalStore, collection_name_for_model


def _load_records(path: Path, limit_docs: int) -> List[Dict]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if limit_docs > 0:
        df = df.head(limit_docs)
    return df.to_dict("records")


def _build_chunks(records: List[Dict], max_chunks: int) -> List:
    chunker = LegalChunkerV2()
    chunks = []
    for idx, record in enumerate(records):
        chunks.extend(chunker.chunk_document(record, doc_index=idx))
        if max_chunks > 0 and len(chunks) >= max_chunks:
            break
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    return chunks


def _upload_for_model(
    model_id: str,
    chunks: List,
    cloud_url: str,
    api_key: str,
    collection_base: str,
    embedding_batch_size: int,
    upsert_batch_size: int,
    force_recreate: bool,
) -> Dict[str, str]:
    texts = [chunk.content for chunk in chunks]
    vectors = embed_texts(
        texts,
        batch_size=embedding_batch_size,
        model_name=model_id,
        normalize_embeddings=None,
        text_type="passage",
        add_prefix=True,
    )
    if not vectors:
        raise RuntimeError(f"No vectors generated for model={model_id}")
    dense_dim = len(vectors[0])
    collection_name = collection_name_for_model(collection_base, model_id, dense_dim)

    store = QdrantLegalStore(
        collection_name=collection_name,
        dense_dim=dense_dim,
        use_cloud=True,
        cloud_url=cloud_url,
        api_key=api_key,
    )
    store.create_collection(force_recreate=force_recreate)
    store.add_documents(chunks=chunks, embeddings=vectors, batch_size=upsert_batch_size)
    stats = store.get_stats()
    return {
        "model_id": model_id,
        "collection_name": collection_name,
        "dense_dim": str(dense_dim),
        "points_count": str(stats.get("points_count", "")),
        "status": str(stats.get("status", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload embeddings to Qdrant Cloud")
    parser.add_argument("--input-path", default="data/gold/vbpl_gold.parquet")
    parser.add_argument("--limit-docs", type=int, default=10)
    parser.add_argument("--max-chunks", type=int, default=500)
    parser.add_argument(
        "--models",
        default="AITeamVN/Vietnamese_Embedding,huyydangg/DEk21_hcmute_embedding",
        help="Comma-separated embedding model ids",
    )
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-api-key", required=True)
    parser.add_argument("--collection-base", default="legal_documents")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--upsert-batch-size", type=int, default=50)
    parser.add_argument("--force-recreate", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        raise ValueError("No models provided")

    records = _load_records(input_path, limit_docs=args.limit_docs)
    chunks = _build_chunks(records, max_chunks=args.max_chunks)
    if not chunks:
        raise RuntimeError("No chunks generated from input data")

    print(f"[INFO] records={len(records)} chunks={len(chunks)}")
    results = []
    for model_id in model_ids:
        print(f"[INFO] Uploading model={model_id}")
        result = _upload_for_model(
            model_id=model_id,
            chunks=chunks,
            cloud_url=args.qdrant_url,
            api_key=args.qdrant_api_key,
            collection_base=args.collection_base,
            embedding_batch_size=args.embedding_batch_size,
            upsert_batch_size=args.upsert_batch_size,
            force_recreate=args.force_recreate,
        )
        results.append(result)
        print(
            f"[DONE] model={result['model_id']} collection={result['collection_name']} "
            f"dim={result['dense_dim']} points={result['points_count']} status={result['status']}"
        )

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    collections = client.get_collections()
    print("[QDRANT] get_collections()")
    print(collections)


if __name__ == "__main__":
    main()
