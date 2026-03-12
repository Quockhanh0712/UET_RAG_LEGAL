# Chainlit Hybrid Smoke (10 queries)

- run_id: `run_20260312_override_1q_fix`
- created_at: `2026-03-12T15:59:03.969256+00:00`
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
- status_ok: `1`
- fallback_count: `0`
- hard_fail_count: `0`
- retrieval_p50_ms: `10142.41`
- retrieval_p95_ms: `10142.41`
- generation_p50_ms: `76066.27`
- generation_p95_ms: `76066.27`
- rerank_used_count: `1`
- rerank_p50_ms: `6923.63`
- rerank_p95_ms: `6923.63`

## Gate Checks

- at_least_3_results_each_query: `True`
- top1_citation_non_empty_gte_9: `False`
- llm_non_empty_or_fallback_gte_9: `False`
- reranker_used_when_enabled: `True`

### Q1. Cong an xa xu phat loi khong mang bang lai xe co dung khong?
- status: `ok`
- retrieval_time_ms: `10142.41`
- rerank_time_ms: `6923.63`
- used_reranker: `True`
- generation_time_ms: `76066.27`
- error: ``
- R1: score=`0.0001` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
