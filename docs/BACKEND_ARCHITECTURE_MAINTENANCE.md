# Backend Architecture and Maintenance Guide

Updated: `2026-04-03`

## 1) Muc tieu tai lieu

Tai lieu nay mo ta backend theo goc nhin van hanh:

- backend co nhung thanh phan nao
- du lieu chay qua cac thanh phan nhu the nao
- API contract hien tai
- nguyen tac giup code gon, de doc, de bao tri

## 2) Buc tranh tong the

Backend duoc to chuc theo 3 tang:

1. API layer (`web_app/backend/api/routes`, `web_app/backend/api/schemas.py`)
2. Application/service layer (`web_app/backend/services`, `src/services`, `src/adapters`)
3. Domain/runtime layer (`src/legal_rag`, `src/legal_rag_pipeline.py`, `src/qdrant_store.py`)

## 3) Cac thanh phan chinh

### 3.1 API composition

- App factory: `web_app/backend/app_factory.py`
  - dang ky middleware CORS
  - include cac router:
    - health
    - settings
    - legal_chat
    - legal_search
    - documents
    - metrics
    - ocr
- Schema contracts: `web_app/backend/api/schemas.py`
  - ChatRequest/ChatResponse
  - SearchRequest/SearchResponse
  - ScoreRequest/ScoreResponse
  - RerankerSettingsResponse
  - UploadResponse

### 3.2 Legacy bridge and orchestration entrypoint

- `web_app/backend/main.py`
  - luu vai tro orchestration runtime va persistence cho chat/search flow
  - cung cap cac handler ma routers goi vao
  - la diem hop nhat giua API layer va domain/runtime layer

### 3.3 Application services

- `web_app/backend/services/shared_legal_services.py`
  - `SharedLegalSearchService`
  - `SharedLegalChatService`
  - quan ly shared runtime cho `legal|user|hybrid`:
    - load runtime settings
    - xac thuc qdrant contract
    - tao rewrite service
    - goi legal pipeline search/query
    - merge legal results voi user-doc results khi `search_mode=hybrid`
    - normalize result/prompt/source contract cho legal corpus va user-uploaded legal chunks
    - build payload dung API schema

- `src/services/query_rewrite_service.py`
  - wrapper bat/tat rewrite
  - fallback safe neu rewrite gateway loi

- `src/services/model_profile_registry.py`
  - quan ly profile model theo id
  - tach biet profile public va profile enabled cho runtime

### 3.4 Runtime and factories

- `src/config/runtime_settings.py`
  - profile runtime
  - env locking va env override
  - apply env nhat quan truoc khi khoi tao pipeline

- `src/runtime/qdrant_contract.py`
  - resolve collection theo embedding model/dim
  - validate collection ton tai + queryable
  - validate BM25 contract

- `src/factories/legal_pipeline_factory.py`
  - cache pipeline theo runtime signature
  - khoi tao pipeline + configure runtime an toan

### 3.5 Domain pipeline (RAG)

- Public facade: `src/legal_rag_pipeline.py`
- Core orchestrator: `src/legal_rag/pipeline/orchestrator.py`
  - retrieval hybrid (dense + sparse)
  - rewrite query (optional)
  - rerank (optional)
  - graph expand (optional)
  - answer generation

### 3.6 Refactor modules (tach boundary de giam coupling)

- Qdrant store:
  - `src/legal_rag/stores/qdrant/client_factory.py`
  - `src/legal_rag/stores/qdrant/payload_profiles.py`
  - `src/legal_rag/stores/qdrant/bm25_encoder.py`
  - `src/legal_rag/stores/qdrant/repository.py`
  - facade compatibility: `src/qdrant_store.py`

- Query rewrite:
  - `src/legal_rag/rewrite/models.py`
  - `src/legal_rag/rewrite/gateway.py`
  - `src/legal_rag/rewrite/heuristics.py`
  - `src/legal_rag/rewrite/parser.py`
  - `src/legal_rag/rewrite/synthesis.py`
  - `src/legal_rag/rewrite/quality_gate.py`
  - facade compatibility: `src/legal_query_rewriter.py`

- Chunking:
  - `src/legal_rag/chunking/policy.py`
  - `src/legal_rag/chunking/metadata_builder.py`
  - `src/legal_rag/chunking/provision_chunker.py`
  - facade compatibility: `src/legal_chunker_v2.py`

## 4) Luong hoat dong chinh

### 4.1 /api/search

1. Router nhan `SearchRequest`
2. `main.search()` route vao shared service (neu shared runtime enabled cho `legal|user|hybrid`)
3. Shared service:
   - normalize query mode
   - load runtime settings
   - validate qdrant contract
   - tao rewrite service
   - `legal`: goi legal pipeline
   - `user`: goi user-doc retrieval qua `UserDocumentIndexer`
   - `hybrid`: goi legal pipeline + user-doc retrieval, roi merge/rerank tren cung contract
4. Adapter (`src/adapters/backend_adapter.py`) format ket qua -> `SearchResponse`

### 4.2 /api/chat

1. Router nhan `ChatRequest`
2. `main.chat()` route vao shared chat service
3. Shared chat service:
   - setup runtime va contract nhu search
   - goi `pipeline.query(...)`
   - neu LLM loi thi fallback retrieval-only
4. Adapter build `ChatResponse`
5. Persist history/metrics vao DB layer

### 4.3 /api/upload va /api/documents

1. Router documents goi handler trong `main.py`
2. `UserDocumentStore` thuc hien:
   - ensure collection
   - chunk file
   - embed dense + sparse
    - upsert points vao `user_docs_private`
3. danh sach tai lieu van la user-wide theo `user_id`
4. retrieval cho `search_mode=user|hybrid` mac dinh cung la user-wide theo `user_id`
5. `session_id` chi duoc giu nhu metadata cho lineage/history/status/artifact/debug, khong con la default retrieval filter

## 5) API map

### Health and status

- `GET /`
- `GET /api/status`

### Chat and search

- `POST /api/chat` -> `ChatRequest` / `ChatResponse`
- `POST /api/search` -> `SearchRequest` / `SearchResponse`

### Documents and history

- `POST /api/upload` (multipart: file, user_id, session_id) -> `UploadResponse`
- `GET /api/documents`
- `DELETE /api/documents/{doc_id}`
- `GET /api/history`
- `GET /api/sessions`

### Settings

- `GET /api/settings/model-profiles`
- `GET /api/settings/reranker`
- `PUT /api/settings/reranker`

### Metrics

- `POST /api/score` -> `ScoreRequest` / `ScoreResponse`
- `POST /api/preload-reranker`
- `GET /api/metrics/stats`
- `GET /api/metrics/history`
- `GET /api/metrics/all`
- `GET /api/metrics/export`

### OCR

- `POST /api/ocr/preview`
- `POST /api/ocr/extract`

## 6) Nguyen tac giu ma nguon gon, sach, de bao tri

### 6.1 Boundary rules

- Router chi lam parsing/validation HTTP
- Service layer dung de orchestration use-case
- Domain module khong phu thuoc FastAPI objects
- Facade file chi de compatibility va migration gate

### 6.2 Contract rules

- DTO public la single source of truth
- Moi thay doi contract phai di kem test contract
- Field trace (`rewrite_*`, `query_facets`, `rewrite_debug`) khong doi ten tuy tien

### 6.3 Runtime rules

- Runtime settings phai duoc apply nhat quan truoc init pipeline
- Qdrant contract phai validate truoc query production
- Collection naming phai deterministic theo embedding model/dim

### 6.4 Refactor rules

- Refactor theo RED -> GREEN -> REFACTOR
- Moi lan tach module phai de lai facade compatibility toi thieu 1 chu ky
- Tranh dublicate logic/DTO giua facade va module moi

### 6.5 Testing gate

Toi thieu 2 gate:

1. focused regression (theo roadmap task)
2. wide system check (`tests/unit` + `tests/integration`)

## 7) Checklist truoc khi merge

- [ ] API schemas khong doi contract ngoai y dinh
- [ ] Shared services va adapters van build dung response model
- [ ] Pipeline facade import path cu van dung
- [ ] Test focused regression pass
- [ ] Test wide system check pass
- [ ] Docs cap nhat neu boundary thay doi
