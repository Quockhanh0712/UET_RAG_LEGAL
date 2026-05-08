# Current State Audit

Updated: `2026-04-02`

## Scope duoc audit

- backend FastAPI trong `web_app/backend/main.py`
- shared legal runtime trong `src/`
- retrieval, rewrite, chunking, factory, runtime settings
- test coverage lien quan toi contract hien tai

## Problem-first framing

Van de can giai quyet khong phai la "chia file cho dep", ma la:

1. file lon dang tron qua nhieu responsibility
2. contract noi bo bi drift nhanh hon test
3. shared legal pipeline va legacy backend dang giao nhau trong mot entrypoint
4. RAG workflow online va ingestion workflow dang duoc model hoa chua ro boundary

Neu tiep tuc them feature tren trang thai nay:

- regression se kho trace
- test unit se phu thuoc implementation detail
- cach cache/runtime singleton hien tai se kho doi
- API layer se tiep tuc phong to

## Architectural tension analysis

### Tension 1: modularity vs deployment safety

- can tach module de code de test
- nhung khong duoc pha API contract `/api/chat`, `/api/search`, `/api/status`

Quyet dinh:

- refactor theo facade pattern
- giu lai import path cu trong mot giai doan chuyen tiep

### Tension 2: pure domain design vs pragmatic migration

- thiet ke sach nhat la chuyen sang package moi cho backend va legal_rag
- nhung migration mot lan rat de gay domino bug

Quyet dinh:

- moi task chi dich chuyen mot lop responsibility
- de `main.py`, `LegalRAGPipeline`, `QdrantLegalStore`, `LegalQueryRewriter` lam facade tam thoi

### Tension 3: high test isolation vs realistic regression coverage

- unit test nho de chay nhanh
- nhung backend nay co nhieu wiring/runtime logic

Quyet dinh:

- dong bang contract tests truoc
- sau do tach service va tao unit test doc lap cho service/module moi

## Hotspot inventory

| File | Approx lines | Trieu chung |
| --- | ---: | --- |
| `web_app/backend/main.py` | 2009 | tron schema, route, search/chat logic, upload, metrics, OCR, singleton bootstrap |
| `src/qdrant_store.py` | 1232 | tron config, client lifecycle, BM25 encoder, payload shaping, collection ops, search ops |
| `src/legal_query_rewriter.py` | 1182 | tron DTO, heuristic extraction, parsing, synthesis, quality gate, gateway |
| `src/legal_rag_pipeline.py` | 879 | tron result DTO, embedding wrapper, LLM wrapper, orchestration, retrieval, graph, context format |
| `src/legal_chunker_v2.py` | 860 | lon va co role ingestion/chunking policy rieng, chua tach khoi online workflow |
| `web_app/backend/main.py` tests | mixed | mot so test integration va unit van mang dau vet architecture cu |

## Responsibility mixing chi tiet

### `web_app/backend/main.py`

Dang gom it nhat 7 lop responsibility:

1. app bootstrap va lifespan
2. pydantic schema
3. runtime/profile factory
4. shared legal search/chat orchestration
5. legacy legal/user/hybrid retrieval path
6. document upload/history/metrics/OCR endpoints
7. singleton resource management

Ket qua:

- route handler kho unit test doc lap
- service logic va HTTP logic dang bi cot cung nhau
- dependency injection gan nhu khong co

### `src/legal_rag_pipeline.py`

Dang gom 5 lop responsibility:

1. DTO ket qua
2. embedding facade
3. answer LLM facade
4. retrieval orchestration
5. singleton accessor/toan cuc

Ket qua:

- pipeline khong con la "orchestrator mong"
- test search/query phai fake meta rat sau

### `src/qdrant_store.py`

Dang gom 4 lop responsibility:

1. runtime config va singleton
2. payload profile/building
3. sparse encoder BM25
4. Qdrant repository + search logic

Ket qua:

- doi BM25 hoac doi payload profile se dong vao file storage lon
- test kho phan tach data-shaping va io

### `src/legal_query_rewriter.py`

Dang gom 6 lop responsibility:

1. public DTOs
2. normalization utils
3. heuristic extraction
4. parse/merge extraction
5. query synthesis
6. quality gate va external LLM call

Ket qua:

- test theo behavior duoc, nhung kho test theo sub-module
- moi thay doi nho deu co nguy co pha pipeline rewrite

## Test state audit

### Test nao dang co gia tri cao

- `tests/unit/test_query_rewrite_service.py`
- `tests/unit/test_runtime_settings.py`
- `tests/unit/test_qdrant_store_runtime.py`
- `tests/unit/test_model_profile_registry.py`
- `tests/unit/test_legal_rag_pipeline_search.py`

### Test nao co dau vet architecture cu

- `tests/unit/test_core.py`
  - tham chieu `backend.core`
  - co skip logic do structure hien tai khong con phu hop
- `tests/integration/test_api.py`
  - mot so assert van nghi ve response fields/behavior cu
  - can review lai truoc khi xem la regression suite chinh

## Refactor constraints

Can giu on dinh:

- request/response contract cua:
  - `POST /api/search`
  - `POST /api/chat`
  - `GET /api/status`
  - `GET /api/settings/model-profiles`
- trace fields:
  - `rewrite_context`
  - `rewrite_debug`
  - `retrieval_query_set`
  - `retrieval_query_plan`
  - `per_subquery_hits`
  - `retrieval_union_count`
- runtime facade:
  - `get_legal_pipeline(...)`
  - `build_search_payload(...)`
  - `build_chat_payload(...)`
  - `LegalRAGPipeline`
  - `QdrantLegalStore`
  - `LegalQueryRewriter`

## Ket luan audit

Huong refactor dung la:

- tach theo boundary
- giu facade cu trong giai doan dau
- dong bang contract bang test truoc
- di tu ngoai vao trong:
  - API layer
  - application service
  - pipeline orchestrator
  - store/rewrite/chunking internals
