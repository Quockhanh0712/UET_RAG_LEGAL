# Target Architecture

Updated: `2026-04-02`

## Muc tieu architecture

- `main.py` chi con la composition root
- route handler khong chua business logic lon
- service layer goi shared pipeline va legacy path qua interface ro rang
- `LegalRAGPipeline` tro thanh facade/orchestrator mong
- qdrant/rewrite/chunking duoc tach thanh module co test doc lap

## Target package map

## Backend target

```text
web_app/backend/
  main.py                       # giu lai entrypoint, rat mong
  app_factory.py                # tao FastAPI app + register routers
  api/
    schemas.py                  # ChatRequest, SearchRequest, ...
    deps.py                     # dependency helpers
    routes/
      health.py
      legal_chat.py
      legal_search.py
      documents.py
      metrics.py
      settings.py
      ocr.py
  application/
    shared_legal_chat_service.py
    shared_legal_search_service.py
    legacy_search_service.py
    document_service.py
    metrics_service.py
    session_service.py
  infrastructure/
    qdrant_resources.py
    reranker_resources.py
    upload_pipeline.py
    persistence_gateway.py
  contracts/
    api_payloads.py             # optional dataclasses/value objects
```

## RAG runtime target

```text
src/legal_rag/
  __init__.py
  contracts/
    results.py                  # LegalAnswer, LegalSearchResult
    citations.py
  pipeline/
    orchestrator.py             # search/query flow
    context_builder.py
  retrieval/
    hybrid_retriever.py
    rerank_service.py
    graph_expander.py
  embeddings/
    encoder.py
  answering/
    llm_gateway.py
    prompt_builder.py
  rewrite/
    models.py
    gateway.py
    heuristics.py
    parser.py
    synthesis.py
    quality_gate.py
  stores/
    qdrant/
      client_factory.py
      payload_profiles.py
      bm25_encoder.py
      repository.py
  chunking/
    policy.py
    provision_chunker.py
    metadata_builder.py
```

## Transitional compatibility rule

Trong giai doan refactor, cac file cu van ton tai nhung chi dong vai tro facade:

- `src/legal_rag_pipeline.py`
- `src/qdrant_store.py`
- `src/legal_query_rewriter.py`
- `web_app/backend/main.py`

Facade rule:

- import tu module moi
- giu function/class public cu
- khong them logic moi vao facade

## Dependency rules

### Backend rules

- `api/routes/*` chi duoc goi `application/*` va `api/schemas.py`
- `application/*` khong import FastAPI
- `application/*` co the dung `src/*` hoac `src/legal_rag/*`
- `infrastructure/*` chua resource io, persistence, singleton

### RAG rules

- `pipeline/orchestrator.py` co the goi `retrieval/*`, `rewrite/*`, `answering/*`
- `retrieval/*` khong duoc biet FastAPI/request model
- `rewrite/*` khong duoc biet Qdrant
- `stores/qdrant/*` khong duoc chua query rewrite logic
- `chunking/*` la ingestion boundary, khong duoc tron vao online answer path

## Contract invariants

Bat buoc giu:

1. `build_search_payload()` va `build_chat_payload()` van tra ve field hien tai
2. `LegalRAGPipeline.search()` va `.query()` van con hoat dong voi call site cu
3. `get_legal_pipeline(settings)` van cache theo runtime key
4. `get_qdrant_store()` va `reset_qdrant_store()` van ton tai
5. `QueryRewriteService` van la gateway on dinh cho rewrite layer

## Rollout strategy

1. dong bang contract test
2. tach route va service layer
3. tach orchestrator va submodule noi bo
4. chuyen facade cu sang wrapper
5. remove dead paths chi sau khi contract test xanh
