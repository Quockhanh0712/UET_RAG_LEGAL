# Execution Backlog

## Phase A: Chainlit Integration

### A1. Shared runtime settings
- Owner: ML/RAG engineer
- Input: current env usage in `apps/chainlit_demo.py`, smoke script, pipeline defaults
- Output: unified runtime settings contract
- Done criteria: Chainlit và smoke script đọc cùng settings source
- Risk: default drift giữa app và script

### A2. Qdrant contract validation
- Owner: ML/RAG engineer
- Input: collection naming, model profile, BM25 manifest path
- Output: health + contract validation module
- Done criteria: startup fail-fast đúng các trường hợp hard fail
- Risk: collection `yellow` nhưng queryable cần policy rõ

### A3. Query rewrite service wrapper
- Owner: ML/RAG engineer
- Input: `LegalQueryRewriter`
- Output: service layer cho policy/timing/telemetry
- Done criteria: Chainlit có thể bật/tắt rewrite qua env mà không đổi pipeline core
- Risk: duplicate telemetry logic giữa app và pipeline

### A4. Chainlit adapter
- Owner: ML/RAG engineer
- Input: `LegalAnswer` hiện tại
- Output: formatter cho answer/citations/timing/rewrite trace
- Done criteria: UI và smoke artifact cùng nhìn một data shape
- Risk: fallback LLM/retrieval hiển thị không nhất quán

### A5. Chainlit smoke
- Owner: QA/demo owner
- Input: 10-query smoke set, cloud runtime mới
- Output: JSON/MD artifact
- Done criteria: pass acceptance gates Phase A
- Risk: cloud health hoặc reranker load instability

## Phase B: Shared Runtime and Refactor

### B1. Pipeline factory
- Owner: ML/RAG engineer
- Input: current bootstrap path trong Chainlit và scripts
- Output: singleton factory cho `LegalRAGPipeline`
- Done criteria: entrypoints không tự bootstrap pipeline bằng logic riêng
- Risk: duplicate singleton lifecycles

### B2. Smoke script migration
- Owner: ML/RAG engineer
- Input: current `scripts/smoke_chainlit_hybrid.py`
- Output: script dùng shared runtime + adapter contracts
- Done criteria: không còn reimplement bootstrap logic trong script
- Risk: artifact schema drift

### B3. Runtime docs sync
- Owner: docs owner hoặc ML/RAG engineer
- Input: Phase A/B implementation
- Output: runbook + research docs updated
- Done criteria: docs phản ánh đúng env contract và rollout state
- Risk: docs lag sau code

## Phase C: Backend Integration and Finalization

### C1. Backend adapter
- Owner: backend engineer
- Input: FastAPI request/response models, shared pipeline factory
- Output: adapter cho legal path
- Done criteria: backend legal retrieval gọi shared pipeline khi flag bật
- Risk: response compatibility với frontend hiện tại

### C2. Feature flag staging
- Owner: backend engineer
- Input: adapter mới, config layer
- Output: `BACKEND_USE_SHARED_LEGAL_PIPELINE`
- Done criteria: rollback bằng flag, không cần code revert
- Risk: fallback path cũ và mới trả shape khác nhau

### C3. Backend parity run
- Owner: QA/demo owner
- Input: cùng query set với Chainlit
- Output: parity note và decision log
- Done criteria: citation/timing/profile đạt gate
- Risk: legal path mới và merge logic user-doc path lệch nhau

### C4. Docs finalization
- Owner: docs owner
- Input: kết quả Phase C
- Output: doc matrix hoàn tất
- Done criteria: runbook, research README, progress log phản ánh đúng trạng thái mới
- Risk: promote xong nhưng docs còn mô tả flow cũ

## Completion Definition
- Chainlit chạy bằng shared runtime và optional rewrite path.
- Smoke artifacts chứng minh rewrite on/off hoạt động an toàn.
- Backend có integration path dưới feature flag.
- Tài liệu tracked đã cập nhật đủ để bàn giao cho vòng implementation tiếp theo.
