# Chainlit Hybrid Smoke (10 queries)

- run_id: `run_20260312_override_1q`
- created_at: `2026-03-12T15:56:06.608107+00:00`
- model: `truro7/vn-law-embedding`
- collection: `legal_rag_vn`
- top_k: `5`
- use_reranker: `True`
- reranker_model: `AITeamVN/Vietnamese_Reranker`
- reranker_alpha: `1.0`
- use_graph: `False`
- require_active_only: `True`

## Summary

- query_total: `1`
- status_ok: `0`
- fallback_count: `1`
- hard_fail_count: `1`
- retrieval_p50_ms: `280.23`
- retrieval_p95_ms: `280.23`
- generation_p50_ms: `0.00`
- generation_p95_ms: `0.00`
- rerank_used_count: `0`
- rerank_p50_ms: `0.00`
- rerank_p95_ms: `0.00`

## Gate Checks

- at_least_3_results_each_query: `False`
- top1_citation_non_empty_gte_9: `False`
- llm_non_empty_or_fallback_gte_9: `False`
- reranker_used_when_enabled: `False`

### Q1. Cong an xa xu phat loi khong mang bang lai xe co dung khong?
- status: `error`
- retrieval_time_ms: `280.23`
- rerank_time_ms: `0.00`
- used_reranker: `False`
- generation_time_ms: `0.00`
- error: `Unexpected Response: 400 (Bad Request)
Raw response content:
b'{"status":{"error":"Bad request: Index required but not found for \\"is_active\\" of one of the following types: [bool]. Help: Create an index for this key or use a different filter."},"time":0.00038 ...' | fallback_error: Unexpected Response: 400 (Bad Request)
Raw response content:
b'{"status":{"error":"Bad request: Index required but not found for \\"is_active\\" of one of the following types: [bool]. Help: Create an index for this key or use a different filter."},"time":0.00004 ...'`
- results: `[]`
