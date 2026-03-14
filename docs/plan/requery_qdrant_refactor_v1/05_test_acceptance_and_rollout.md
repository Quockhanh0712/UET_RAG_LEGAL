# Test, Acceptance, and Rollout

## Test Matrix

### 1. Preflight
- Qdrant cloud reachable.
- `get_collections()` trả collection model-specific đúng.
- Collection target query được.
- Reranker model load được trên runtime đang dùng.

### 2. Chainlit Startup
- Startup resolve đúng collection `truro7`.
- Startup fail-fast nếu collection missing, `red`, hoặc contract mismatch.
- Startup không tự silent fallback sang collection khác.

### 3. Rewrite Tests
- `QUERY_REWRITER_ENABLED=false`: chỉ dùng query gốc.
- `QUERY_REWRITER_ENABLED=true`: trả `1..6` subqueries hoặc fallback về query gốc.
- Timeout hoặc API error không làm crash app.
- Mất số liệu định lượng phải fallback.

### 4. Retrieval Integration
- Hybrid dense+sparse hoạt động trên collection cloud mới.
- Query-level RRF chỉ chạy sau khi có nhiều subqueries.
- Cross-encoder rerank chạy sau shortlist union.
- Top results có `citation_full` non-empty.

### 5. Backend Parity
- Cùng query, Chainlit và backend shared path phải dùng cùng collection/model profile.
- Citation top results không lệch rõ rệt.
- Backend trả được `sources`, `timing`, `reranker_used`, `rewrite_used`.

## Acceptance Gates
- Smoke `10` query: `10/10` không crash.
- `>=9/10` query có top-1 `citation_full` non-empty.
- Rewrite complex set `20` query: parse/fallback hợp lệ `>=90%`.
- `rewrite=on` không được làm giảm `citation_valid_rate` so với `rewrite=off`.
- Backend chỉ sang stage bật cờ khi Chainlit gates đã pass.

## Rollout Order
1. Shared runtime settings + Qdrant contract layer.
2. Chainlit refactor sang shared runtime.
3. Smoke `rewrite=off`.
4. Smoke `rewrite=on` cho query phức tạp.
5. Shared runtime cho smoke scripts.
6. Backend adapter dưới feature flag.
7. Backend parity test.
8. Promote flag ở dev/smoke.

## Rollback Gates
- Collection mismatch giữa runtime và target model-space.
- BM25 manifest mismatch.
- Rewrite regression làm citation tệ đi.
- Backend parity fail.
- Cloud health không ổn định hoặc collection chuyển `red`.

## Artifacts Required
- Chainlit smoke JSON/MD.
- Rewrite A/B notes trên cùng query set.
- Decision log append cho mỗi lần đổi runtime profile.
- Progress log cập nhật `next_48h` sau mỗi test run chính.
