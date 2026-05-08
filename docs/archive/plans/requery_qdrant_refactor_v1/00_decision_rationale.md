# Decision Rationale: ReQuery + Qdrant Cloud + Shared Runtime

## Purpose
- Chốt logic ra quyết định cho vòng tích hợp `query rewrite + Qdrant Cloud mới` vào hệ thống.
- Khóa phạm vi để implementation sau đó không phải quyết định lại.
- Giữ retrieval core hiện tại, chỉ dọn kiến trúc runtime và adapter.

## Current State
- `src/legal_query_rewriter.py` đã có parser XML, fallback, dedup, number-preservation.
- `src/legal_rag_pipeline.py` đã có `rewrite -> per-subquery retrieval -> query-level RRF -> rerank`.
- `apps/chainlit_demo.py` đang là đường chạy ngắn nhất để kiểm tra end-to-end trên Qdrant cloud mới.
- `web_app/backend/main.py` vẫn dùng legal retrieval path cũ, chưa tái sử dụng shared pipeline.
- Collection phục vụ hiện tại là `legal_documents__truro7_vn_law_embedding__d768`.

## Research Framing Used
- `tension hunting`:
  - `demo speed vs production safety`
  - `shared pipeline reuse vs adapter stability`
- `constraint manipulation`:
  - không đổi chunking/indexing trong vòng này
  - không đưa graph vào scope
  - không ghi secret vào repo
- `adjacent possible`:
  - rewriter đã tồn tại, pipeline đã có RRF và rerank, nên vòng này chỉ cần chuẩn hóa runtime và adapter

## Decisions
- Lộ trình triển khai là `Chainlit-first`, sau đó mới nối backend.
- Rewriter là `feature-flagged`, không bật mặc định cho toàn hệ thống.
- Qdrant cloud mới là target duy nhất cho luồng demo/integration vòng này.
- `truro7/vn-law-embedding` là embedding profile chuẩn cho runtime plan này.
- Không redesign `LegalRAGPipeline`; chỉ tách bootstrap, config, contract validation và app adapters.
- Refactor depth là `Medium`: đủ để giảm hardcode/env scattering, chưa đụng sâu benchmark/indexing core.

## Why This Path
- Chainlit ít coupling hơn backend hiện tại, nên phù hợp làm rollout path đầu tiên.
- Shared runtime giúp Chainlit, smoke script và backend đọc cùng một contract Qdrant/model/rewrite.
- Adapter split giúp giữ `src/legal_rag_pipeline.py` ở vai trò orchestration thay vì trộn app-specific formatting.

## Non-Goals
- Không refactor graph pipeline.
- Không đổi chunking policy `Điều-first`.
- Không đổi collection schema hoặc reindex trong bộ plan này.
- Không gom `user_docs_private` vào shared legal retrieval path.

## Output Set
- `01_chainlit_requery_qdrant_plan.md`
- `02_backend_shared_pipeline_plan.md`
- `03_module_refactor_map.md`
- `04_env_contract_and_runtime_profiles.md`
- `05_test_acceptance_and_rollout.md`
- `06_documentation_update_matrix.md`
- `07_execution_backlog.md`
