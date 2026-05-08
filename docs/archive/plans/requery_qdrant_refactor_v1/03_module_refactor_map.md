# Module Refactor Map

## Goal
- Tách bootstrap và app-specific logic khỏi orchestration core.
- Giảm hardcode env trong entrypoints.
- Giữ refactor ở mức `Medium`, không đụng benchmark/indexing core.

## Proposed Modules

### `src/config/runtime_settings.py`
- Đọc env cho Chainlit, backend, smoke scripts.
- Chuẩn hóa default values.
- Validate kiểu dữ liệu và flag logic.

### `src/runtime/qdrant_contract.py`
- Resolve collection theo `QDRANT_COLLECTION` và `QDRANT_COLLECTION_BY_MODEL`.
- Health check cluster/collection.
- Kiểm tra collection status.
- Gắn BM25 manifest/contract validation vào một chỗ duy nhất.

### `src/services/query_rewrite_service.py`
- Bọc `LegalQueryRewriter` bằng policy bật/tắt.
- Chuẩn hóa telemetry:
  - latency
  - status
  - fallback reason
  - missing numbers
- Trả result type dùng chung cho Chainlit và backend.

### `src/factories/legal_pipeline_factory.py`
- Tạo singleton/lifecycle cho `LegalRAGPipeline`.
- Nhận runtime settings đã validate.
- Tránh mỗi entrypoint tự bootstrap pipeline theo cách riêng.

### `src/adapters/chainlit_adapter.py`
- Format answer, citations, timing, rewrite trace cho Chainlit UI.
- Gói fallback retrieval-only khi LLM lỗi.

### `src/adapters/backend_adapter.py`
- Map request/response giữa FastAPI models và pipeline output.
- Gắn compatibility layer cho schema cũ của backend.

## Existing Files That Stay
- `src/legal_query_rewriter.py`: low-level HTTP client + XML parser + fallback guards.
- `src/legal_rag_pipeline.py`: orchestration retrieval/rewrite/rerank core.
- `src/qdrant_store.py`: access layer tới Qdrant, không gánh app bootstrap.

## Dependency Direction
- `apps/*` và `web_app/backend/*` chỉ gọi `config -> runtime -> factory -> adapter`.
- `services` không import UI layer.
- `runtime` không biết Chainlit/FastAPI.
- `legal_rag_pipeline.py` không tự đọc env trực tiếp cho startup policy của app.

## Migration Sequence
1. Tạo shared settings + Qdrant contract layer.
2. Chuyển Chainlit sang dùng factory và adapter.
3. Chuyển smoke script sang cùng runtime path.
4. Tạo backend adapter và feature flag.
5. Xóa bootstrap/hardcode dư khi parity đã pass.

## Non-Goals
- Không tách lại toàn bộ benchmark harness.
- Không thay đổi Qdrant payload schema.
- Không thay đổi chunk generation hoặc indexing scripts trong vòng này.
