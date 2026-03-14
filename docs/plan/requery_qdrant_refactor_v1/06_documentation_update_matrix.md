# Documentation Update Matrix

## Goal
- Chốt tài liệu nào phải cập nhật khi code integration hoàn tất.
- Tách rõ update sau `Phase A`, `Phase B`, `Phase C`.

## Documents to Update

### `docs/CHAINLIT_HYBRID_DEMO_RUNBOOK.md`
- Phase A update.
- Nội dung cần thêm:
  - shared runtime loader
  - rewriter feature flag
  - collection contract validation
  - fail-fast conditions
  - smoke commands và artifact paths mới nếu thay đổi

### `data/gold/research/README.md`
- Phase B update.
- Nội dung cần thêm:
  - Qdrant cloud target đang dùng
  - current embedding/runtime profile
  - rewrite integration status
  - link tới artifacts smoke/benchmark liên quan

### `data/gold/research/embedding_graph_execution_progress_v3.md`
- Phase A, B, C đều phải append.
- Nội dung cần thêm:
  - run_id
  - config profile
  - rewrite on/off result
  - backend parity status
  - blocker và `next_48h`

## Optional Follow-Up Docs
- `docs/SYSTEM_DOCUMENTATION.md`
  - chỉ update sau khi backend shared path đã promote
- backend-specific API docs
  - chỉ update nếu response contract đổi field công khai

## Update Rules
- Không ghi secret hoặc API key vào tài liệu tracked.
- Runtime docs phải ghi rõ `env contract`, không nhúng giá trị nhạy cảm.
- Mọi thay đổi policy rollout phải được phản ánh ở runbook trước khi bật flag cho dev/smoke.
- Nếu artifact path đổi, runbook và research README phải update cùng lượt.

## Done Criteria
- Sau mỗi phase, có ít nhất một tài liệu tracked được cập nhật tương ứng.
- Không có tài liệu nào mô tả collection/profile cũ khi runtime đã chuyển sang shared path mới.
