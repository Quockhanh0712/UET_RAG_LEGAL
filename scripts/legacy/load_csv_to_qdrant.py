"""
Load CSV Legal Data to Qdrant

Usage:
python scripts/legacy/load_csv_to_qdrant.py --csv path/to/law_clean_core_v1.csv
    
    # Resume from saved embeddings:
python scripts/legacy/load_csv_to_qdrant.py --csv path/to/law_clean_core_v1.csv --resume

This script:
1. Reads CSV with legal documents
2. Chunks documents using LegalChunkerV2
3. Generates embeddings (saves to disk for resume)
4. Indexes into Qdrant with dense + sparse vectors
"""

import os
import sys
import shutil
import pickle

# Disable TensorFlow - use PyTorch only
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
import logging
from pathlib import Path
from typing import List
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache directory for embeddings
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV file"""
    logger.info(f"Loading CSV: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Expected columns
    expected_cols = [
        'van_ban_id', 'ten_van_ban', 'loai_van_ban', 'co_quan',
        'chuong', 'ten_chuong', 'dieu_so', 'tieu_de_dieu',
        'noi_dung', 'clean_text'
    ]
    
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}")
    
    # Use clean_text if available, else noi_dung
    if 'clean_text' in df.columns:
        df['content'] = df['clean_text'].fillna(df.get('noi_dung', ''))
    else:
        df['content'] = df['noi_dung']
    
    # Remove empty content
    df = df[df['content'].notna() & (df['content'].str.len() > 10)]
    
    logger.info(f"Loaded {len(df)} rows")
    return df


def chunk_documents(df: pd.DataFrame) -> List:
    """Chunk documents using LegalChunkerV2"""
    from src.legal_chunker_v2 import LegalChunkerV2
    
    logger.info("Chunking documents...")
    chunker = LegalChunkerV2()
    
    chunks = chunker.process_dataframe(df)
    
    logger.info(f"Created {len(chunks)} chunks from {len(df)} articles")
    
    # Log stats
    sizes = [len(c.content) for c in chunks]
    logger.info(f"Chunk sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    
    return chunks


def save_to_cache(chunks: List, embeddings: List, cache_name: str = "full"):
    """Save chunks and embeddings to cache"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    chunks_path = CACHE_DIR / f"chunks_{cache_name}.pkl"
    embeddings_path = CACHE_DIR / f"embeddings_{cache_name}.npy"
    
    # Save chunks
    logger.info(f"Saving {len(chunks)} chunks to {chunks_path}")
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save embeddings as numpy array (float32 to save memory)
    logger.info(f"Saving {len(embeddings)} embeddings to {embeddings_path}")
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(embeddings_path, embeddings_array)
    
    logger.info(f"Cache saved: chunks={chunks_path.stat().st_size/1e6:.1f}MB, embeddings={embeddings_path.stat().st_size/1e6:.1f}MB")


def load_from_cache(cache_name: str = "full"):
    """Load chunks and embeddings from cache"""
    chunks_path = CACHE_DIR / f"chunks_{cache_name}.pkl"
    embeddings_path = CACHE_DIR / f"embeddings_{cache_name}.npy"
    
    if not chunks_path.exists() or not embeddings_path.exists():
        return None, None
    
    logger.info(f"Loading from cache...")
    
    # Load chunks
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from cache")
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded {len(embeddings)} embeddings from cache (shape: {embeddings.shape})")
    
    return chunks, embeddings.tolist()


def generate_embeddings(chunks: List, batch_size: int = 32) -> List[List[float]]:
    """Generate embeddings for chunks"""
    from src.legal_rag_pipeline import LegalEmbedding
    
    logger.info("Generating embeddings...")
    
    embedding_model = LegalEmbedding()
    embedding_model.load()
    
    contents = [c.content for c in chunks]
    
    embeddings = []
    for i in tqdm(range(0, len(contents), batch_size), desc="Embedding"):
        batch = contents[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch)
        embeddings.extend(batch_embeddings)
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings


def clear_qdrant_data(qdrant_path: str = "./data/qdrant_db"):
    """Completely clear Qdrant data folder"""
    if os.path.exists(qdrant_path):
        logger.info(f"Clearing Qdrant data at {qdrant_path}...")
        try:
            shutil.rmtree(qdrant_path)
            logger.info("Qdrant data cleared successfully")
        except Exception as e:
            logger.warning(f"Could not clear Qdrant data: {e}")
            # Try removing individual files
            for item in Path(qdrant_path).glob("*"):
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except:
                    pass


def index_to_qdrant(chunks: List, embeddings: List[List[float]], force_recreate: bool = False):
    """Index chunks and embeddings to Qdrant"""
    from src.qdrant_store import QdrantLegalStore
    
    logger.info("Indexing to Qdrant...")
    
    # Clear old data first if force_recreate
    if force_recreate:
        clear_qdrant_data()
    
    dense_dim = len(embeddings[0]) if embeddings else 768
    store = QdrantLegalStore(dense_dim=dense_dim)
    store.create_collection(force_recreate=force_recreate)
    
    store.add_documents(chunks, embeddings, batch_size=100)
    
    stats = store.get_stats()
    logger.info(f"Indexing complete. Stats: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Load CSV legal data to Qdrant")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--force-recreate", action="store_true", help="Recreate collection")
    parser.add_argument("--limit", type=int, help="Limit number of rows (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from cached embeddings")
    parser.add_argument("--cache-name", default="full", help="Cache name for resume")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing (only generate embeddings)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    chunks = None
    embeddings = None
    
    # Try to resume from cache
    if args.resume:
        chunks, embeddings = load_from_cache(args.cache_name)
        if chunks and embeddings:
            logger.info(f"Resuming with {len(chunks)} chunks and {len(embeddings)} embeddings from cache")
    
    if chunks is None:
        # 1. Load CSV
        df = load_csv(args.csv)
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} rows")
        
        # 2. Chunk documents
        chunks = chunk_documents(df)
        
        # 3. Generate embeddings
        embeddings = generate_embeddings(chunks, batch_size=args.batch_size)
        
        # 4. Save to cache immediately (in case of crash)
        cache_name = args.cache_name if not args.limit else f"test_{args.limit}"
        save_to_cache(chunks, embeddings, cache_name)
        logger.info(f"Embeddings cached as '{cache_name}' - can resume with --resume --cache-name {cache_name}")
    
    # 5. Index to Qdrant (skip if requested)
    if not args.skip_index:
        index_to_qdrant(chunks, embeddings, force_recreate=args.force_recreate)
    else:
        logger.info("Skipping indexing (--skip-index)")
    
    elapsed = time.time() - start_time
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("Done!")


if __name__ == "__main__":
    main()
