# Chainlit Integration Plan: ReQuery + Qdrant Cloud

## Goal
- Đưa `apps/chainlit_demo.py` thành entrypoint chuẩn cho demo legal retrieval trên Qdrant cloud mới.
- Dùng shared runtime loader thay cho hardcoded env bootstrap.
- Giữ `Direct Pipeline`, không đi qua backend cũ.

## Target Runtime
- Embedding model: `truro7/vn-law-embedding`
- Collection target: `legal_documents__truro7_vn_law_embedding__d768`
- Dense normalization: `true`
- Reranker: `AITeamVN/Vietnamese_Reranker`
- Graph context: `off` mặc định
- Rewriter: `off` mặc định, bật bằng env cho smoke/demo query phức tạp

## Runtime Flow
1. Load shared runtime settings.
2. Resolve Qdrant collection theo model-space.
3. Validate cloud connectivity, collection status và BM25 contract.
4. Lấy singleton pipeline từ factory.
5. Nếu `QUERY_REWRITER_ENABLED=true`, gọi rewrite service trước retrieval.
6. Chạy hybrid retrieval, query-level RRF và rerank như hiện tại.
7. Trả answer, citations, timing, rewrite trace.
8. Nếu LLM lỗi, fallback retrieval-only vẫn giữ citation và trace.

## Startup Validation
- Qdrant cloud reachable.
- Collection resolve đúng theo model profile.
- Collection status `green` hoặc `yellow` nhưng query được.
- Nếu `red`, fail-fast ngay startup.
- Nếu strict BM25 check bật, manifest phải khớp collection và embedding profile.

## Required Refactor in Chainlit Layer
- Bỏ hardcoded env scattering trong `apps/chainlit_demo.py`.
- Chuyển logic resolve collection, profile validation và pipeline bootstrap ra shared modules.
- Giữ UI contract hiện tại nhưng bổ sung trace rõ hơn cho rewrite:
  - `rewritten_queries`
  - `rewrite_status`
  - `fallback_reason`
  - `missing_numbers`
  - `per_subquery_hits`
  - `retrieval_union_count`

## Demo Behavior
- Default mode:
  - rewriter `off`
  - graph `off`
  - reranker `on`
- Smoke mode:
  - cho phép bật rewriter bằng env
  - ghi artifact JSON/MD cho từng query
- Failure behavior:
  - rewriter timeout hoặc parse lỗi không được làm hỏng phiên chat
  - collection mismatch phải dừng startup, không cố chạy trong trạng thái sai profile

## Acceptance
- Chainlit khởi động được với cloud profile mới.
- `10/10` query smoke không crash.
- `>=9/10` query có top-1 `citation_full` non-empty.
- Query rewrite `on` không làm giảm `citation_valid_rate` trên tập smoke so với `off`.
- UI và artifact đều nhìn thấy trace rewrite khi feature flag bật.
