"""
Migrate data from local Qdrant to Qdrant Server (Docker)

This script copies all vectors and payloads from local SQLite storage
to the Qdrant server running in Docker.
"""

import os
import sys
import logging
import time
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
LOCAL_PATH = "./data/qdrant_db"
SERVER_HOST = "localhost"
SERVER_PORT = 6333
COLLECTION_NAME = "legal_documents"
BATCH_SIZE = 100
DENSE_DIM = 768


def migrate():
    """Migrate from local to server"""
    
    logger.info("="*60)
    logger.info("Migrate Local Qdrant -> Qdrant Server")
    logger.info("="*60)
    
    # Connect to local
    logger.info(f"Connecting to local: {LOCAL_PATH}")
    local_client = QdrantClient(path=LOCAL_PATH)
    
    # Connect to server
    logger.info(f"Connecting to server: {SERVER_HOST}:{SERVER_PORT}")
    server_client = QdrantClient(host=SERVER_HOST, port=SERVER_PORT)
    
    # Check local collection
    try:
        local_info = local_client.get_collection(COLLECTION_NAME)
        total_points = local_info.points_count
        logger.info(f"Local collection: {total_points:,} points")
    except Exception as e:
        logger.error(f"Local collection not found: {e}")
        return
    
    # Create collection on server
    collections = server_client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if exists:
        logger.info("Deleting existing server collection...")
        server_client.delete_collection(COLLECTION_NAME)
    
    logger.info("Creating collection on server...")
    server_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
                on_disk=True  # Store vectors on disk for memory efficiency
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        },
        # Optimizers for better search performance
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,  # Build index after 20k points
            memmap_threshold=20000     # Use memory mapping
        ),
        # HNSW config for faster search
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100,
            on_disk=True
        )
    )
    
    # Create payload indexes
    indexed_fields = [
        ("van_ban_id", "keyword"),
        ("loai_van_ban", "keyword"),
        ("co_quan", "keyword"),
        ("dieu_so", "keyword"),
        ("chuong", "keyword")
    ]
    
    for field_name, field_type in indexed_fields:
        try:
            server_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_type
            )
        except:
            pass
    
    logger.info("Collection created with HNSW index")
    
    # Migrate data in batches
    logger.info(f"Migrating {total_points:,} points in batches of {BATCH_SIZE}...")
    
    offset = None
    migrated = 0
    start_time = time.time()
    
    with tqdm(total=total_points, desc="Migrating") as pbar:
        while True:
            # Scroll through local data
            records, next_offset = local_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not records:
                break
            
            # Prepare points for server
            points = []
            for record in records:
                points.append(models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload
                ))
            
            # Upsert to server
            server_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
            migrated += len(records)
            pbar.update(len(records))
            
            offset = next_offset
            if offset is None:
                break
    
    elapsed = time.time() - start_time
    
    # Verify
    server_info = server_client.get_collection(COLLECTION_NAME)
    
    logger.info("="*60)
    logger.info("Migration Complete!")
    logger.info("="*60)
    logger.info(f"Migrated: {migrated:,} points")
    logger.info(f"Server count: {server_info.points_count:,}")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"Speed: {migrated/elapsed:.0f} points/sec")
    
    # Copy BM25 encoder
    bm25_src = os.path.join(LOCAL_PATH, "bm25_encoder.json")
    bm25_dst = "./data/qdrant_server/bm25_encoder.json"
    
    if os.path.exists(bm25_src):
        os.makedirs(os.path.dirname(bm25_dst), exist_ok=True)
        import shutil
        shutil.copy(bm25_src, bm25_dst)
        logger.info(f"Copied BM25 encoder to {bm25_dst}")
    
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Update .env: QDRANT_USE_SERVER=true")
    logger.info("2. Restart chatbot")


if __name__ == "__main__":
    migrate()
