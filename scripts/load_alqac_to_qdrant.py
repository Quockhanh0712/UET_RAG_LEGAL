"""
Load ALQAC.csv contexts to Qdrant for benchmark testing

This script loads the "context" column from ALQAC.csv as documents
into the legal_documents collection for benchmark testing.

Usage:
    python scripts/load_alqac_to_qdrant.py --csv ALQAC.csv
"""

import sys
import os
from pathlib import Path
import logging
import hashlib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "legal_documents"
DENSE_DIM = 768


def load_embedding_model():
    """Load embedding model"""
    import torch
    from sentence_transformers import SentenceTransformer
    
    model_name = "huyydangg/DEk21_hcmute_embedding"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"[EMBED] Loading model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    
    if device == "cuda":
        model.half()
        logger.info("[EMBED] Enabled FP16 precision")
    
    return model


def build_bm25_encoder(texts):
    """Build simple BM25 encoder"""
    from collections import Counter
    import math
    
    # Load Vietnamese tokenizer
    try:
        from pyvi import ViTokenizer
        tokenize = lambda text: ViTokenizer.tokenize(text.lower()).split()
        logger.info("[BM25] Using pyvi tokenizer")
    except:
        tokenize = lambda text: text.lower().split()
        logger.info("[BM25] Using simple split")
    
    # Build vocabulary and IDF
    vocab = {}
    doc_freq = Counter()
    doc_count = len(texts)
    
    logger.info(f"[BM25] Building vocabulary from {doc_count} documents...")
    
    for text in tqdm(texts, desc="Building BM25"):
        tokens = tokenize(text)
        unique = set(tokens)
        
        for token in unique:
            if token not in vocab:
                vocab[token] = len(vocab)
            doc_freq[token] += 1
    
    # Calculate IDF
    idf = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
    
    logger.info(f"[BM25] Vocabulary size: {len(vocab)}")
    
    def encode(text, k1=1.5, b=0.75, avg_len=100):
        """Encode text to sparse vector"""
        tokens = tokenize(text)
        doc_len = len(tokens)
        
        if doc_len == 0:
            return [], []
        
        tf = Counter(tokens)
        indices = []
        values = []
        
        for token, freq in tf.items():
            if token in vocab:
                idx = vocab[token]
                idf_val = idf.get(token, 0)
                
                # BM25 formula
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len / avg_len)
                score = idf_val * (numerator / denominator)
                
                if score > 0:
                    indices.append(idx)
                    values.append(float(score))
        
        return indices, values
    
    return encode


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="ALQAC.csv", help="Path to ALQAC.csv")
    args = parser.parse_args()
    
    csv_path = PROJECT_ROOT / args.csv
    
    # Load ALQAC.csv
    logger.info(f"[DATA] Loading {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Remove duplicates in context
    original_count = len(df)
    df = df.drop_duplicates(subset=['context'])
    logger.info(f"[DATA] Loaded {len(df)} unique contexts (from {original_count} total rows)")
    
    contexts = df['context'].tolist()
    
    # Connect to Qdrant
    logger.info(f"[QDRANT] Connecting to {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Recreate collection
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"[QDRANT] Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    logger.info(f"[QDRANT] Created collection: {COLLECTION_NAME}")
    
    # Load embedding model
    embed_model = load_embedding_model()
    
    # Build BM25 encoder
    bm25_encode = build_bm25_encoder(contexts)
    
    # Generate embeddings
    logger.info(f"[EMBED] Generating embeddings...")
    embeddings = []
    batch_size = 64
    
    for i in tqdm(range(0, len(contexts), batch_size), desc="Embedding"):
        batch = contexts[i:i + batch_size]
        batch_emb = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.extend(batch_emb.tolist())
    
    # Upload to Qdrant
    logger.info(f"[QDRANT] Uploading {len(contexts)} documents...")
    
    points = []
    for idx, (context, embedding) in enumerate(zip(contexts, embeddings)):
        # Generate sparse vector
        sparse_indices, sparse_values = bm25_encode(context)
        
        # Create point ID from context hash
        point_id = int(hashlib.md5(context.encode()).hexdigest()[:16], 16) % (10**15)
        
        point = PointStruct(
            id=point_id,
            vector={
                "dense": embedding
            },
            payload={
                "content": context,
                "source": "ALQAC",
                "idx": idx
            }
        )
        
        if sparse_indices:
            point.vector["sparse"] = SparseVector(
                indices=sparse_indices,
                values=sparse_values
            )
        
        points.append(point)
        
        # Upload in batches
        if len(points) >= 100:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
    
    # Upload remaining
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    # Verify
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ HOÀN TẤT! Data ALQAC đã sẵn sàng cho benchmark")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Collection: {COLLECTION_NAME}")
    logger.info(f"📊 Points: {info.points_count}")
    logger.info(f"📊 Unique contexts: {len(contexts)}")
    logger.info(f"{'='*60}")
    logger.info(f"\n🚀 Bây giờ có thể chạy benchmark:")
    logger.info(f"   python scripts/benchmark_alqac.py --csv ALQAC.csv --sample 50 --use-reranker")


if __name__ == "__main__":
    main()
