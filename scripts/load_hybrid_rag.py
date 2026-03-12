"""
Load final_chunks_v4_safe.csv to Qdrant with Hybrid Search (Dense + BM25)

Chiến lược: "Điều luật nguyên tử" (Atomic Articles)
- Mỗi dòng CSV = 1 điều luật hoàn chỉnh (không cần cắt thêm)
- Dense: huyydangg/DEk21_hcmute_embedding (768D)
- Sparse: BM25 (tự build)

Usage:
    python scripts/load_hybrid_rag.py --csv "final_chunks_v4_safe.csv" [--force-recreate]
"""

import os
import sys
import argparse
import logging
import ast
import json
from typing import List, Dict, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "legal_rag_hybrid"
DENSE_DIM = 768  # DEk21_hcmute_embedding dimension
BATCH_SIZE = 32
EMBEDDING_BATCH_SIZE = 64

# BM25 cache path
BM25_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "bm25_hybrid.json"


class SimpleBM25:
    """
    Simple BM25 encoder for Vietnamese legal text.
    Không cần thư viện ngoài nặng nề, tự build từ đầu.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avg_doc_len: float = 0
        self.doc_count: int = 0
        
        # Vietnamese tokenizer (optional)
        self._tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Lazy load Vietnamese tokenizer"""
        try:
            from pyvi import ViTokenizer
            self._tokenizer = ViTokenizer.tokenize
            logger.info("[BM25] Using pyvi Vietnamese tokenizer")
        except ImportError:
            logger.warning("[BM25] pyvi not found, using simple split")
            self._tokenizer = lambda x: x
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = self._tokenizer(text.lower())
        # Split by spaces, keep compound words from pyvi
        tokens = []
        for word in text.split():
            # Remove punctuation
            clean = ''.join(c for c in word if c.isalnum() or c == '_')
            if clean:
                tokens.append(clean)
        return tokens
    
    def fit(self, documents: List[str], show_progress: bool = True):
        """
        Build vocabulary and IDF from documents.
        
        Args:
            documents: List of document texts
            show_progress: Show progress bar
        """
        import math
        
        logger.info(f"[BM25] Fitting on {len(documents)} documents...")
        
        self.doc_count = len(documents)
        doc_freq: Dict[str, int] = {}
        total_len = 0
        
        iterator = tqdm(documents, desc="Building BM25") if show_progress else documents
        
        for doc in iterator:
            tokens = self.tokenize(doc)
            total_len += len(tokens)
            
            # Count unique tokens per document for Document Frequency
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
                
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 1
        
        # Calculate IDF for each token
        for token, df in doc_freq.items():
            # IDF with smoothing
            self.idf[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
        
        logger.info(f"[BM25] ✅ Vocabulary size: {len(self.vocab)}")
        logger.info(f"[BM25] ✅ Average doc length: {self.avg_doc_len:.1f}")
    
    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Encode text to sparse vector.
        
        Returns:
            (indices, values) - Sparse vector representation
        """
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        
        if doc_len == 0:
            return [], []
        
        # Count term frequency
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        indices = []
        values = []
        
        for token, freq in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf = self.idf.get(token, 0)
                
                # BM25 score formula
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score = idf * (numerator / denominator)
                
                if score > 0:
                    indices.append(idx)
                    values.append(float(score))
        
        return indices, values
    
    def save(self, path: str):
        """Save BM25 encoder to JSON file"""
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "avg_doc_len": self.avg_doc_len,
            "doc_count": self.doc_count,
            "k1": self.k1,
            "b": self.b
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"[BM25] Saved encoder to {path}")
    
    def load(self, path: str) -> bool:
        """Load BM25 encoder from JSON file"""
        if not os.path.exists(path):
            return False
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.avg_doc_len = data["avg_doc_len"]
        self.doc_count = data["doc_count"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        
        logger.info(f"[BM25] Loaded encoder with {len(self.vocab)} vocab from {path}")
        return True


def load_embedding_model():
    """Load Dense embedding model with GPU acceleration"""
    import torch
    from sentence_transformers import SentenceTransformer
    
    model_name = os.getenv("EMBEDDING_MODEL", "huyydangg/DEk21_hcmute_embedding")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"[EMBED] Loading model: {model_name}")
    logger.info(f"[EMBED] Device: {device}")
    
    model = SentenceTransformer(model_name, device=device)
    
    # Enable FP16 for faster inference on GPU
    if device == "cuda":
        model.half()
        logger.info(f"[EMBED] GPU: {torch.cuda.get_device_name(0)}")
        logger.info("[EMBED] Enabled FP16 precision")
    
    return model


def parse_metadata(meta_str: str) -> Dict:
    """Parse metadata string to dictionary"""
    if not meta_str or pd.isna(meta_str):
        return {}
    
    try:
        # Try ast.literal_eval first (handles Python dict format)
        return ast.literal_eval(meta_str)
    except:
        try:
            # Try JSON parse
            return json.loads(meta_str.replace("'", '"'))
        except:
            return {}


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV data"""
    logger.info(f"[DATA] Loading CSV: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Validate required columns
    required_cols = ['chunk_id', 'content', 'metadata']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Clean data
    df = df.dropna(subset=['content'])
    df['content'] = df['content'].astype(str)
    df['chunk_id'] = df['chunk_id'].astype(str)
    
    logger.info(f"[DATA] ✅ Loaded {len(df)} chunks")
    return df


def create_collection(client: QdrantClient, force_recreate: bool = False):
    """Create Qdrant collection with hybrid vectors"""
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if exists:
        if force_recreate:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"[QDRANT] Deleted existing collection: {COLLECTION_NAME}")
        else:
            logger.info(f"[QDRANT] Collection exists: {COLLECTION_NAME}")
            return False  # Not recreated
    
    # Create collection with Dense + Sparse vectors
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    
    # Create payload indexes for filtering
    indexed_fields = ["so_hieu", "dieu", "is_split"]
    for field in indexed_fields:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema="keyword"
            )
        except Exception:
            pass
    
    logger.info(f"[QDRANT] ✅ Created collection: {COLLECTION_NAME}")
    return True  # Newly created


def batch_embed(model, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Batch embed texts with progress bar"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
    
    return all_embeddings


def upload_to_qdrant(
    client: QdrantClient,
    df: pd.DataFrame,
    embeddings: List[List[float]],
    bm25: SimpleBM25,
    batch_size: int = BATCH_SIZE
):
    """Upload data to Qdrant with progress bar"""
    total = len(df)
    logger.info(f"[QDRANT] Uploading {total} documents...")
    
    for i in tqdm(range(0, total, batch_size), desc="Uploading"):
        batch_df = df.iloc[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        points = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            chunk_id = row['chunk_id']
            content = row['content']
            metadata = parse_metadata(row.get('metadata', '{}'))
            
            # Generate sparse vector
            sparse_indices, sparse_values = bm25.encode(content)
            
            # Use chunk_id as point ID (convert to int if needed)
            try:
                point_id = int(chunk_id)
            except ValueError:
                # Hash the chunk_id if not numeric
                import hashlib
                point_id = int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16)
            
            # Build payload
            payload = {
                "chunk_id": chunk_id,
                "content": content,
                **metadata  # Spread metadata fields
            }
            
            point = PointStruct(
                id=point_id,
                vector={
                    "dense": batch_embeddings[idx]
                },
                payload=payload
            )
            
            # Add sparse vector only if not empty
            if sparse_indices:
                point.vector["sparse"] = SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            
            points.append(point)
        
        # Upsert batch
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    
    logger.info(f"[QDRANT] ✅ Uploaded {total} documents")


def main():
    parser = argparse.ArgumentParser(description="Load legal data to Qdrant with Hybrid Search")
    parser.add_argument("--csv", type=str, default="final_chunks_v4_safe.csv", help="CSV file path")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreate collection")
    parser.add_argument("--skip-bm25-cache", action="store_true", help="Skip loading BM25 cache")
    args = parser.parse_args()
    
    # Resolve CSV path
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = PROJECT_ROOT / csv_path
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Connect to Qdrant
    logger.info(f"[QDRANT] Connecting to {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Test connection
    try:
        client.get_collections()
        logger.info("[QDRANT] ✅ Connected successfully")
    except Exception as e:
        logger.error(f"❌ Cannot connect to Qdrant: {e}")
        logger.error("Please ensure Qdrant is running: docker start qdrant")
        sys.exit(1)
    
    # Load CSV data
    df = load_csv_data(csv_path)
    contents = df['content'].tolist()
    
    # Create or check collection
    is_new_collection = create_collection(client, force_recreate=args.force_recreate)
    
    # Initialize BM25
    bm25 = SimpleBM25()
    
    # Try to load cached BM25
    if not args.skip_bm25_cache and not args.force_recreate and bm25.load(str(BM25_CACHE_PATH)):
        logger.info("[BM25] Using cached encoder")
    else:
        # Fit BM25 on all documents
        bm25.fit(contents)
        bm25.save(str(BM25_CACHE_PATH))
    
    # Load embedding model
    model = load_embedding_model()
    
    # Generate embeddings
    logger.info(f"[EMBED] Generating embeddings for {len(contents)} documents...")
    embeddings = batch_embed(model, contents, batch_size=EMBEDDING_BATCH_SIZE)
    
    # Upload to Qdrant
    upload_to_qdrant(client, df, embeddings, bm25)
    
    # Verify
    collection_info = client.get_collection(COLLECTION_NAME)
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 HOÀN TẤT! Dữ liệu đã sẵn sàng để truy vấn Hybrid.")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Collection: {COLLECTION_NAME}")
    logger.info(f"📊 Points count: {collection_info.points_count}")
    logger.info(f"📊 BM25 vocab size: {len(bm25.vocab)}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
