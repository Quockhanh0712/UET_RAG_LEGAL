# Project Code Report

Generated: 2026-05-06

## 1. Executive Summary

Day la he thong Vietnamese Legal RAG, tap trung vao truy van phap ly tieng Viet voi trace ro rang cho retrieval, rewrite, rerank va answer generation.

Trang thai hien tai:

- Backend chinh la FastAPI trong `web_app/backend`.
- Frontend chinh la React/Vite trong `web_app/frontend`.
- Core runtime va RAG pipeline nam trong `src`.
- Shared legal path hien la duong mac dinh nen uu tien, bat bang `BACKEND_USE_SHARED_LEGAL_PIPELINE=true`.
- Retrieval chinh dung Qdrant Cloud, dense embedding `truro7/vn-law-embedding`, BM25 sparse, RRF merge va reranker `AITeamVN/Vietnamese_Reranker`.
- Rewrite V2 da co trace va quality gate; no khong phai paraphrase tu do.

Van de con lai khong phai wiring, ma la chat luong retrieval/ranking va viec giu contract on dinh khi tiep tuc refactor cac file lon.

## 2. Repo Map

Source of truth:

| Path | Vai tro |
|:---|:---|
| `README.md` | Onboarding tong quan, runtime mac dinh, lenh chay backend/frontend |
| `docs/LEGAL_SYSTEM_OVERVIEW.md` | Snapshot he thong va thanh phan chinh |
| `docs/LEGAL_PIPELINE_CURRENT_STATE.md` | Trang thai pipeline, data, chunking, model, trace |
| `docs/LEGAL_REWRITE_V2.md` | Quy tac rewrite V2 va contract trace |
| `docs/BACKEND_ARCHITECTURE_MAINTENANCE.md` | Backend boundary, API map, maintenance rules |
| `src/` | Core legal runtime, pipeline, rewrite, retrieval, services |
| `web_app/backend/` | FastAPI app, routers, schemas, user-doc handling, persistence |
| `web_app/frontend/` | React UI cho chat/search/legal lab/OCR/dashboard |
| `tests/` | Unit va integration tests |
| `scripts/ops/` | Scripts van hanh index/upload/patch Qdrant |
| `scripts/research/` | Benchmark/evaluation/exploration scripts |
| `scripts/migrations/` | Migration utilities |
| `scripts/legacy/` | Legacy loaders, khong phai duong primary |
| `docs/archive/` | Tai lieu lich su, plans, source snapshots |

Generated/local artifacts khong nen coi la source of truth:

- `.venv/`
- `.pytest_cache/`
- `web_app/frontend/dist/`
- `node_modules/`
- `data/uploads/`
- `__pycache__/`
- local log/cache files

## 3. Runtime Architecture

He thong duoc chia thanh 4 lop:

1. UI layer: React components trong `web_app/frontend/src`.
2. API layer: FastAPI app, routers va schemas trong `web_app/backend`.
3. Application/service layer: shared backend services, adapters, runtime settings, model registry.
4. Domain/runtime layer: legal RAG pipeline, rewrite, Qdrant store, chunking, embeddings, reranker.

Luong search/chat shared:

```text
Frontend request
-> FastAPI router/schema
-> web_app.backend.main handler
-> SharedLegalSearchService / SharedLegalChatService
-> RuntimeSettings + Qdrant contract validation
-> QueryRewriteService (optional)
-> LegalRAGPipeline
-> dense + sparse retrieval
-> query-level RRF merge
-> reranker
-> context builder
-> RemoteLLMClient answer (chat only)
-> backend adapter payload
-> frontend trace/cards/messages
```

## 4. Backend Map

Important backend files:

| File | Vai tro |
|:---|:---|
| `web_app/backend/main.py` | Legacy orchestration entrypoint, runtime globals, route handlers, persistence glue |
| `web_app/backend/app_factory.py` | CORS va router composition |
| `web_app/backend/api/schemas.py` | Public API DTO contracts |
| `web_app/backend/api/routes/*.py` | Thin routers goi legacy/shared handlers |
| `web_app/backend/services/shared_legal_services.py` | Shared orchestration cho `legal`, `user`, `hybrid` |
| `web_app/backend/document_ingestion_service.py` | Upload/ingestion workflow |
| `web_app/backend/user_store.py` | User document store wrapper |
| `web_app/backend/database.py` | SQLAlchemy persistence |
| `src/adapters/backend_adapter.py` | Convert pipeline results sang API payload |

Boundary rules:

- Router chi nen parse/validate HTTP va goi service/handler.
- `api/schemas.py` la contract public; doi field phai doi test.
- Domain code trong `src/legal_rag` khong nen import FastAPI objects.
- `web_app/backend/main.py` van la file lon va co nhieu legacy glue; khi sua nen tach theo huong co test, khong refactor lan rong.
- Shared service phai giu trace fields: `rewrite_context`, `rewrite_debug`, `retrieval_query_set`, `query_facets`, `per_subquery_hits`, `retrieval_union_count`.

## 5. Frontend Map

Important frontend files:

| File | Vai tro |
|:---|:---|
| `web_app/frontend/src/App.jsx` | App shell, state, API calls, routing view |
| `web_app/frontend/src/components/LegalLabArea.jsx` | Search Lab / Chat Lab shared legal UI |
| `web_app/frontend/src/components/ChatArea.jsx` | Legacy/general chat UI |
| `web_app/frontend/src/components/MessageBubble.jsx` | Answer/source/trace presentation |
| `web_app/frontend/src/components/Sidebar.jsx` | Session, docs, mode controls |
| `web_app/frontend/src/components/LegalOCR.jsx` | OCR UI |
| `web_app/frontend/src/components/QualityDashboard.jsx` | Metrics dashboard |
| `web_app/frontend/src/index.css` | Global styling |

Frontend state is mostly local component state plus `localStorage` for:

- `userId`
- `darkMode`
- `selectedModelProfileId`
- `queryMode`
- `labTab`
- `requireActiveOnly`
- `topK`
- `documentScope`
- `uploadMode`

Pitfall:

- UI model selector chi anh huong rewrite va answer LLM. No khong doi embedding retrieval hay Qdrant collection.
- Search Lab khong sinh answer; Chat Lab moi goi answer generation.
- `search_mode=user|hybrid` mac dinh user-wide theo `user_id`, khong khoa theo `session_id`.

## 6. Core RAG Runtime

Important runtime files:

| File | Vai tro |
|:---|:---|
| `src/legal_rag_pipeline.py` | Public facade va singleton pipeline getter |
| `src/legal_rag/pipeline/orchestrator.py` | Retrieval, rewrite integration, rerank, answer orchestration |
| `src/legal_rag/pipeline/context_builder.py` | Build rewrite/evidence context |
| `src/legal_rag/contracts/results.py` | Public result contracts |
| `src/factories/legal_pipeline_factory.py` | Runtime-aware pipeline cache/factory |
| `src/config/runtime_settings.py` | Runtime profiles and env override |
| `src/runtime/qdrant_contract.py` | Collection/embedding/BM25 contract validation |
| `src/services/query_rewrite_service.py` | Safe wrapper for rewrite on/off/fallback |
| `src/services/remote_llm_client.py` | Remote answer/rewrite provider client |
| `src/services/model_profile_registry.py` | Public model profile registry |
| `src/reranker.py` | Vietnamese reranker singleton path |

Qdrant modules:

| File | Vai tro |
|:---|:---|
| `src/legal_rag/stores/qdrant/client_factory.py` | Qdrant client setup |
| `src/legal_rag/stores/qdrant/repository.py` | Dense/sparse search and payload operations |
| `src/legal_rag/stores/qdrant/payload_profiles.py` | Payload shape/profile rules |
| `src/legal_rag/stores/qdrant/bm25_encoder.py` | Sparse encoder |
| `src/qdrant_store.py` | Compatibility facade |

Rewrite modules:

| File | Vai tro |
|:---|:---|
| `src/legal_query_rewriter.py` | Compatibility/public rewriter facade, still large |
| `src/legal_rag/rewrite/models.py` | Rewrite DTOs |
| `src/legal_rag/rewrite/gateway.py` | Provider gateway |
| `src/legal_rag/rewrite/parser.py` | Parse model output |
| `src/legal_rag/rewrite/synthesis.py` | Deterministic retrieval query synthesis |
| `src/legal_rag/rewrite/quality_gate.py` | Rewrite validation/fallback |
| `src/legal_rag/rewrite/heuristics.py` | Heuristic helpers |

Chunking/user docs:

| File | Vai tro |
|:---|:---|
| `src/legal_chunker_v2.py` | Compatibility legal chunker facade |
| `src/legal_rag/chunking/policy.py` | Chunking policy |
| `src/legal_rag/chunking/metadata_builder.py` | Metadata builder |
| `src/legal_rag/chunking/provision_chunker.py` | Provision chunker |
| `src/document_ingestion/*` | User document extraction/chunking/indexing/storage |

## 7. Data And Model Contracts

Current default shared runtime:

- model profile: `qwen3_5_9b_ollama_remote`
- model: `qwen3.5:9b`
- provider: `ollama_generate`
- endpoint: `http://222.255.250.24:11434`
- embedding: `truro7/vn-law-embedding`
- vector dim: `768`
- collection: `legal_documents__truro7_vn_law_embedding__d768`
- reranker: `AITeamVN/Vietnamese_Reranker`

Critical runtime rules:

- Set `BACKEND_USE_SHARED_LEGAL_PIPELINE=true` for shared legal path.
- Validate Qdrant contract before production query.
- Collection naming must stay deterministic by embedding model/dim.
- `require_active_only=true` filters inactive legal docs in shared legal path.
- `session_id` is lineage/history/debug metadata, not the default retrieval scope for user documents.

## 8. Test And Verification Strategy

Use focused tests by change type:

| Change type | Suggested check |
|:---|:---|
| Backend schemas/adapters | `pytest tests/unit/test_backend_adapter.py tests/unit/test_backend_application_services.py` |
| Shared legal service | `pytest tests/unit/test_backend_application_services.py tests/integration/test_backend_api_modularization.py` |
| Pipeline/orchestrator | `pytest tests/unit/test_legal_rag_pipeline_search.py tests/unit/test_legal_rag_pipeline_orchestrator_split.py` |
| Rewrite | `pytest tests/unit/test_legal_query_rewriter.py tests/unit/test_rewrite_modularization.py` |
| Qdrant contract/store | `pytest tests/unit/test_qdrant_contract.py tests/unit/test_qdrant_payload_profile.py tests/unit/test_qdrant_store_runtime.py` |
| Document ingestion | `pytest tests/unit/test_document_ingestion_chunking.py tests/unit/test_document_ingestion_indexer.py tests/integration/test_document_upload_async.py` |
| Repo hygiene/docs layout | `pytest tests/unit/test_repo_hygiene.py` |
| Frontend | `cd web_app/frontend && npm run build` |

Wide check before major merge:

```powershell
pytest tests/unit tests/integration
cd web_app\frontend
npm run build
```

Compile check from README:

```powershell
python -m py_compile src\legal_query_rewriter.py src\legal_rag_pipeline.py src\adapters\backend_adapter.py src\services\remote_llm_client.py web_app\backend\main.py
```

## 9. Hotspots And Risks

Large files that deserve extra care:

| File | Risk |
|:---|:---|
| `web_app/backend/main.py` | Legacy glue, persistence, route handlers, runtime globals in one file |
| `src/legal_query_rewriter.py` | Large compatibility facade plus rewrite behavior |
| `web_app/backend/services/shared_legal_services.py` | Central shared orchestration for legal/user/hybrid |
| `src/legal_chunker_v2.py` | Compatibility chunking path and parser edge cases |
| `src/legal_rag/stores/qdrant/repository.py` | Dense/sparse Qdrant behavior and payload contract |
| `src/legal_rag/pipeline/orchestrator.py` | Core retrieval/rewrite/rerank/answer sequencing |
| `web_app/frontend/src/App.jsx` | Many API calls and global UI state |
| `web_app/frontend/src/components/MessageBubble.jsx` | Trace/source rendering complexity |
| `web_app/frontend/src/components/LegalLabArea.jsx` | Shared legal UI controls and trace display |

Observed workflow friction from Antigravity artifacts:

- There are 18 conversation folders under `~/.gemini/antigravity/brain`.
- Several sessions have `task.md` + `implementation_plan.md` but no `walkthrough.md`; treat these as incomplete evidence, not proof of failure.
- Recent sessions cluster around dataset/retrieval/backend refactor work, matching the repo hotspots above.
- The project has many archived docs and moved scripts; new agents must distinguish live docs from `docs/archive`.

Primary technical risks:

1. Contract drift in API trace fields.
2. Confusing legacy path with shared legal path.
3. Accidentally changing retrieval scope from user-wide to session-only.
4. Updating UI model selector as if it controls embedding.
5. Refactoring compatibility facades before tests cover old import paths.
6. Running cloud/Qdrant-dependent tests without env access.
7. Committing local uploads, caches, secrets, or research artifacts.

## 10. Recommended Next Work

1. Keep reducing backend `main.py` by moving one bounded use-case at a time behind tested services.
2. Add/maintain contract tests for every public field returned by `/api/search` and `/api/chat`.
3. Build a small labeled retrieval benchmark around known Vietnamese legal questions.
4. Keep `docs/LEGAL_PIPELINE_CURRENT_STATE.md` updated when runtime, collection, model, or retrieval behavior changes.
5. Add frontend smoke coverage for Search Lab/Chat Lab trace rendering if UI work continues.
6. Keep archived docs under `docs/archive` and live docs short, explicit, and current.
