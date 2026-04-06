# Legal System Overview

Updated: `2026-03-16`

## 1. Muc tieu he thong

He thong nay la mot legal RAG demo cho tieng Viet, tap trung vao 2 bai toan:

- `Search Lab`: debug retrieval va citation
- `Chat Lab`: retrieval + answer generation co trace

Muc tieu hien tai:
- wiring phai on dinh
- rewrite phai co trace ro rang
- answer phai tranh hallucination va anti-loop
- citation phai duoc giu lai trong toan bo pipeline

## 2. Thanh phan chinh

### Frontend
- React app: `web_app/frontend`
- man hinh chinh: `web_app/frontend/src/components/LegalLabArea.jsx`

### Backend
- FastAPI app: `web_app/backend/main.py`
- shared runtime duoc dung khi:
  - `BACKEND_USE_SHARED_LEGAL_PIPELINE=true`
  - request `search_mode` nam trong `legal|user|hybrid`

### Shared legal runtime
- runtime settings: `src/config/runtime_settings.py`
- pipeline factory: `src/factories/legal_pipeline_factory.py`
- main pipeline: `src/legal_rag_pipeline.py`
- backend adapter: `src/adapters/backend_adapter.py`
- user-doc retrieval cho `user|hybrid` di qua active versioned collection cua `UserDocumentIndexer`
- user-doc retrieval mac dinh la user-wide theo `user_id`, khong khoa theo `session_id`

### Retrieval stack
- embeddings: `truro7/vn-law-embedding`
- vector store: Qdrant Cloud
- sparse: BM25
- merge: query-level RRF
- reranker: `AITeamVN/Vietnamese_Reranker`

### LLM stack
- default profile: `qwen3_5_9b_ollama_remote`
- endpoint: `http://222.255.250.24:11434`
- model: `qwen3.5:9b`

## 3. Collection va du lieu

Collection retrieval mac dinh:
- `legal_documents__truro7_vn_law_embedding__d768`

Nguon index:
- `data/gold/vbpl_gold.parquet`

Payload retrieval level hien tai:
- chu yeu la `Dieu` hoac `article_window`
- chunking production dang o che do `Dieu-first`

## 4. Luong xu ly tong quan

### Search Lab

```text
query
-> optional rewrite
-> hybrid retrieval
-> query-level RRF
-> reranker
-> source cards + rewrite trace
```

### Chat Lab

```text
query
-> optional rewrite
-> hybrid retrieval
-> query-level RRF
-> reranker
-> query_context + evidence_context
-> answer generation
-> sources + trace
```

## 5. Rewrite trong he thong hien tai

Rewrite hien tai la `Rewrite V2`:
- objective extraction
- question-unit synthesis
- quality gate
- original query luon duoc giu lai

Xem chi tiet tai:
- `docs/LEGAL_REWRITE_V2.md`

## 6. Answer generation hien tai

Answer duoc goi qua `RemoteLLMClient`.

Prompt answer da duoc siet lai de:
- `QUERY_CONTEXT` chi la metadata, khong phai can cu
- `EVIDENCE_CONTEXT` moi la can cu
- output theo schema:
  - `KET_LUAN`
  - `CAN_CU`
  - `PHAN_TICH_NGAN`
  - `CHUA_DU_CAN_CU`
- anti-loop trim footer va paragraph lap

## 7. Trace va debug

Backend payload hien tai tra ve:
- `rewrite_context`
- `rewrite_debug`
- `retrieval_query_set`
- `retrieval_query_plan`
- `query_facets`
- `per_subquery_hits`
- `retrieval_union_count`

Frontend hien duoc:
- original vs rewrite
- query facets
- conditions
- clause units
- named law hints
- question units
- retrieval query plan

## 8. Gioi han hien tai

He thong da on ve mat wiring, nhung van con 2 bottleneck:

1. Retrieval relevance
- top-1 van co the lech mien
- rewrite da tot hon, nhung retrieval/rerank chua du manh

2. Answer quality phu thuoc evidence
- neu evidence lech, answer se abstain hoac tra ve can cu yeu

## 9. File can biet

- `README.md`
- `docs/LEGAL_REWRITE_V2.md`
- `docs/LEGAL_PIPELINE_CURRENT_STATE.md`
- `src/legal_query_rewriter.py`
- `src/legal_rag_pipeline.py`
- `src/services/remote_llm_client.py`
- `src/adapters/backend_adapter.py`
- `web_app/backend/main.py`
