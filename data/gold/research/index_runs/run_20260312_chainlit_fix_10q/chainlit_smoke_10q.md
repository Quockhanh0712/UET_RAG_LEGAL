# Chainlit Hybrid Smoke (10 queries)

- run_id: `run_20260312_chainlit_fix_10q`
- created_at: `2026-03-12T16:16:15.748036+00:00`
- model: `truro7/vn-law-embedding`
- collection: `legal_rag_vn`
- top_k: `5`
- use_reranker: `True`
- reranker_model: `AITeamVN/Vietnamese_Reranker`
- reranker_alpha: `1.0`
- use_graph: `False`
- require_active_only: `True`

## Summary

- query_total: `10`
- status_ok: `10`
- fallback_count: `0`
- hard_fail_count: `0`
- retrieval_p50_ms: `4141.08`
- retrieval_p95_ms: `4491.71`
- generation_p50_ms: `78827.55`
- generation_p95_ms: `103825.93`
- rerank_used_count: `10`
- rerank_p50_ms: `2015.56`
- rerank_p95_ms: `2196.61`

## Gate Checks

- at_least_3_results_each_query: `True`
- top1_citation_non_empty_gte_9: `False`
- llm_non_empty_or_fallback_gte_9: `True`
- reranker_used_when_enabled: `True`

### Q1. Cong an xa xu phat loi khong mang bang lai xe co dung khong?
- status: `ok`
- retrieval_time_ms: `7269.75`
- rerank_time_ms: `5115.59`
- used_reranker: `True`
- generation_time_ms: `40129.78`
- error: ``
- R1: score=`0.0001` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 

### Q2. Phai thuc hien thao tac nap mau vao binh chua va xu ly mau so bo bang hoa chat nhu the nao?
- status: `ok`
- retrieval_time_ms: `4491.71`
- rerank_time_ms: `2158.21`
- used_reranker: `True`
- generation_time_ms: `80749.82`
- error: ``
- R1: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 

### Q3. Trinh tu danh gia chat luong hang nam doi voi Kiem soat vien duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `4079.39`
- rerank_time_ms: `1914.89`
- used_reranker: `True`
- generation_time_ms: `78720.22`
- error: ``
- R1: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 

### Q4. Tu tu chet sau khi tiem thuoc doc ma khong co than nhan nhan xac thi duoc ho tro mai tang nhu the nao?
- status: `ok`
- retrieval_time_ms: `4309.46`
- rerank_time_ms: `2100.30`
- used_reranker: `True`
- generation_time_ms: `91440.36`
- error: ``
- R1: score=`0.0002` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 

### Q5. Trach nhiem cua Bo Khoa hoc va Cong nghe ve quan ly va phat trien cong nghiep an ninh duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `4137.69`
- rerank_time_ms: `2017.10`
- used_reranker: `True`
- generation_time_ms: `103825.93`
- error: ``
- R1: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 

### Q6. Dap pha bien bao khu vuc bien gioi bi phat the nao?
- status: `ok`
- retrieval_time_ms: `4222.80`
- rerank_time_ms: `2014.03`
- used_reranker: `True`
- generation_time_ms: `76785.80`
- error: ``
- R1: score=`0.0001` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 

### Q7. Muc tieu phat trien khu cong nghiep ho tro duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `4066.58`
- rerank_time_ms: `1994.33`
- used_reranker: `True`
- generation_time_ms: `70365.43`
- error: ``
- R1: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 

### Q8. Thua phat lai duoc tong dat nhung giay to ho so tai lieu nao?
- status: `ok`
- retrieval_time_ms: `3946.32`
- rerank_time_ms: `1966.65`
- used_reranker: `True`
- generation_time_ms: `78934.88`
- error: ``
- R1: score=`0.0001` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 

### Q9. Muc phat khi su dung trai phep long duong do thi he pho de rua xe
- status: `ok`
- retrieval_time_ms: `4144.48`
- rerank_time_ms: `2196.61`
- used_reranker: `True`
- generation_time_ms: `107119.09`
- error: ``
- R1: score=`0.0005` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0001` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 

### Q10. Hai cong ty con chung cong ty me co duoc tham du goi thau cua nhau khong?
- status: `ok`
- retrieval_time_ms: `3940.92`
- rerank_time_ms: `1902.86`
- used_reranker: `True`
- generation_time_ms: `72548.18`
- error: ``
- R1: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R2: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
- R3: score=`0.0000` doc_id=`` path=`` type=``
  citation: 
