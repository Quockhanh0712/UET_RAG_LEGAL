# Backend Plan: Stage Shared Legal Pipeline After Chainlit

## Goal
- Nối `web_app/backend/main.py` vào shared legal pipeline sau khi Chainlit pass.
- Giữ nguyên `user_docs_private` flow hiện tại.
- Tách legal retrieval path cũ ra phía sau feature flag để rollback an toàn.

## Current Problem
- Backend đang tự embed/query legal collection bằng code cũ.
- Backend chưa dùng shared rewrite path, query-level RRF trace hoặc contract validation mới.
- Điều này tạo rủi ro lệch hành vi giữa Chainlit demo và API backend.

## Target Design
- Legal path trong backend đi qua adapter gọi `LegalRAGPipeline`.
- User-doc path vẫn giữ `user_store` hiện tại.
- `hybrid_search()` trong backend trở thành router:
  - legal only -> shared legal pipeline
  - user only -> existing user store
  - hybrid -> merge legal pipeline result với user-doc result theo policy hiện tại

## Feature Flag
- `BACKEND_USE_SHARED_LEGAL_PIPELINE=false` lúc đầu.
- Khi `false`, backend giữ legal retrieval path cũ.
- Khi `true`, `/api/search` và `/api/chat` legal path dùng adapter mới.
- Flag chỉ promote sau khi Chainlit smoke và backend parity pass.

## Response Mapping
- Backend response tối thiểu phải expose:
  - `sources`
  - `search_time`
  - `generation_time`
  - `rerank_time`
  - `reranker_used`
  - `rewrite_used`
  - `rewrite_status`
- Nếu API model hiện tại không có field tương ứng, adapter phải map vào `metadata` hoặc mở rộng response type có backward compatibility.

## Staged Rollout
- Stage 1:
  - tạo adapter và flag
  - chưa bật shared path
- Stage 2:
  - bật ở dev/smoke
  - chạy cùng query set đã dùng cho Chainlit
- Stage 3:
  - promote khi parity về collection/profile/citation pass

## Done Criteria
- Backend legal path dùng cùng collection/profile với Chainlit.
- Backend trả được citation và timing nhất quán.
- Rollback chỉ cần tắt flag, không sửa code hoặc đổi collection.
