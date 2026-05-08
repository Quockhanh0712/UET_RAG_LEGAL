"""
Copy data from legal_rag_hybrid to legal_documents collection

This script copies all points from one collection to another
to make data available for benchmark testing.
"""

import sys
from pathlib import Path
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
SOURCE_COLLECTION = "legal_rag_hybrid"
TARGET_COLLECTION = "legal_documents"
BATCH_SIZE = 100


def main():
    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Get source collection info
    source_info = client.get_collection(SOURCE_COLLECTION)
    logger.info(f"Source collection: {SOURCE_COLLECTION}")
    logger.info(f"Total points: {source_info.points_count}")
    
    # Create target collection with same config
    try:
        client.get_collection(TARGET_COLLECTION)
        logger.info(f"Target collection exists: {TARGET_COLLECTION}")
        
        # Ask user if want to delete
        response = input(f"Delete and recreate {TARGET_COLLECTION}? (y/n): ")
        if response.lower() == 'y':
            client.delete_collection(TARGET_COLLECTION)
            logger.info(f"Deleted collection: {TARGET_COLLECTION}")
        else:
            logger.info("Appending to existing collection...")
    except:
        pass
    
    # Create target collection
    try:
        client.create_collection(
            collection_name=TARGET_COLLECTION,
            vectors_config={
                "dense": VectorParams(size=768, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )
        logger.info(f"Created collection: {TARGET_COLLECTION}")
    except:
        logger.info(f"Collection already exists: {TARGET_COLLECTION}")
    
    # Copy data in batches
    logger.info(f"Copying data from {SOURCE_COLLECTION} to {TARGET_COLLECTION}...")
    
    offset = None
    total_copied = 0
    
    with tqdm(total=source_info.points_count, desc="Copying") as pbar:
        while True:
            # Scroll through source collection
            records, next_offset = client.scroll(
                collection_name=SOURCE_COLLECTION,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not records:
                break
            
            # Convert Record objects to PointStruct
            from qdrant_client.http.models import PointStruct
            points = []
            for record in records:
                point = PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload
                )
                points.append(point)
            
            # Upsert to target
            client.upsert(
                collection_name=TARGET_COLLECTION,
                points=points
            )
            
            total_copied += len(records)
            pbar.update(len(records))
            
            if next_offset is None:
                break
            
            offset = next_offset
    
    # Verify
    target_info = client.get_collection(TARGET_COLLECTION)
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Copy completed!")
    logger.info(f"{'='*60}")
    logger.info(f"Source: {SOURCE_COLLECTION} - {source_info.points_count} points")
    logger.info(f"Target: {TARGET_COLLECTION} - {target_info.points_count} points")
    logger.info(f"Copied: {total_copied} points")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
