"""
Debug script để kiểm tra tìm kiếm tài liệu người dùng
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from user_store import UserStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json

def main():
    print("=" * 80)
    print("DEBUG USER DOCUMENT SEARCH")
    print("=" * 80)
    
    # Initialize
    store = UserStore()
    store.ensure_collection()
    client = store._get_client()
    
    # 1. Check collection info
    print("\n[1] Kiểm tra Collection Info:")
    try:
        collection_info = client.get_collection(store.COLLECTION_NAME)
        print(f"   Collection: {store.COLLECTION_NAME}")
        print(f"   Points count: {collection_info.points_count}")
        print(f"   Vectors config: {collection_info.config.params.vectors}")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return
    
    # 2. Scroll through all points to see user_ids
    print("\n[2] Danh sách user_id và doc_id trong collection:")
    try:
        all_points = client.scroll(
            collection_name=store.COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False
        )[0]
        
        user_ids = set()
        doc_ids = set()
        user_doc_map = {}  # user_id -> list of doc_ids
        
        for point in all_points:
            user_id = point.payload.get("user_id", "N/A")
            doc_id = point.payload.get("doc_id", "N/A")
            filename = point.payload.get("filename", "N/A")
            
            user_ids.add(user_id)
            doc_ids.add(doc_id)
            
            if user_id not in user_doc_map:
                user_doc_map[user_id] = []
            if doc_id not in user_doc_map[user_id]:
                user_doc_map[user_id].append({
                    "doc_id": doc_id,
                    "filename": filename
                })
        
        print(f"   Total unique user_ids: {len(user_ids)}")
        print(f"   Total unique doc_ids: {len(doc_ids)}")
        print(f"\n   Chi tiết:")
        for user_id in sorted(user_ids):
            print(f"\n   📁 User ID: {user_id}")
            docs = user_doc_map[user_id]
            for doc in docs:
                print(f"      - Doc ID: {doc['doc_id'][:20]}... | Filename: {doc['filename']}")
                
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return
    
    # 3. Test search với một user_id cụ thể
    print("\n[3] Test Search:")
    if not user_ids:
        print("   ⚠️ Không có user_id nào trong collection!")
        return
    
    # Lấy user_id đầu tiên để test
    test_user_id = list(user_ids)[0]
    print(f"   Testing với user_id: {test_user_id}")
    
    # Sample queries để test
    test_queries = [
        "hợp đồng",
        "điều khoản",
        "quyền lợi",
        "trách nhiệm"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            results = store.search(
                query=query,
                user_id=test_user_id,
                top_k=3,
                use_hybrid=True
            )
            print(f"   ✅ Tìm thấy {len(results)} kết quả")
            for i, res in enumerate(results, 1):
                print(f"      {i}. Score: {res['score']:.4f} | Filename: {res.get('filename', 'N/A')}")
                print(f"         Content preview: {res['content'][:100]}...")
        except Exception as e:
            print(f"   ❌ Lỗi search: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Test với wrong user_id
    print("\n[4] Test với wrong user_id (should return 0 results):")
    wrong_user_id = "nonexistent_user_12345"
    print(f"   Testing với user_id: {wrong_user_id}")
    
    for query in test_queries[:1]:  # Just test 1 query
        print(f"\n   Query: '{query}'")
        try:
            results = store.search(
                query=query,
                user_id=wrong_user_id,
                top_k=3,
                use_hybrid=True
            )
            print(f"   ✅ Tìm thấy {len(results)} kết quả (expected 0)")
        except Exception as e:
            print(f"   ❌ Lỗi search: {e}")
    
    # 5. Check documents in memory
    print("\n[5] Documents in memory (_documents dict):")
    print(f"   Total documents: {len(store._documents)}")
    for doc_id, doc in list(store._documents.items())[:5]:  # Show first 5
        print(f"   - {doc_id[:20]}... | user: {doc.user_id} | status: {doc.status} | filename: {doc.filename}")
    
    print("\n" + "=" * 80)
    print("✅ DEBUG COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
