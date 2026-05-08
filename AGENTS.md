# AGENTS.md

This file is the working guide for coding agents in this repository.

## Start Here

Before changing code, read these files in order:

1. `README.md`
2. `docs/LEGAL_SYSTEM_OVERVIEW.md`
3. `docs/LEGAL_PIPELINE_CURRENT_STATE.md`
4. `docs/BACKEND_ARCHITECTURE_MAINTENANCE.md`
5. `docs/LEGAL_REWRITE_V2.md` if touching rewrite/query behavior
6. `docs/PROJECT_CODE_REPORT.md`

Check worktree state first:

```powershell
git status --short
```

The repo may contain user edits. Do not revert or delete unrelated changes.

## Project Summary

This is a Vietnamese Legal RAG system:

- core runtime: `src/`
- FastAPI backend: `web_app/backend/`
- React/Vite frontend: `web_app/frontend/`
- tests: `tests/`
- live docs: `docs/*.md`
- archived/historical docs: `docs/archive/`

The current primary path is the shared legal runtime:

```env
BACKEND_USE_SHARED_LEGAL_PIPELINE=true
```

With that flag enabled, `search_mode=legal|user|hybrid` goes through the shared backend orchestration.

## Architecture Boundaries

Respect these boundaries:

- Routers in `web_app/backend/api/routes` should stay thin.
- Public API DTOs live in `web_app/backend/api/schemas.py`.
- Shared backend orchestration lives in `web_app/backend/services/shared_legal_services.py`.
- Backend payload formatting lives in `src/adapters/backend_adapter.py`.
- Core RAG behavior lives in `src/legal_rag`.
- Compatibility facades such as `src/legal_rag_pipeline.py`, `src/qdrant_store.py`, `src/legal_query_rewriter.py`, and `src/legal_chunker_v2.py` should keep old import paths working unless the task explicitly removes them.
- Domain/runtime code in `src/legal_rag` should not depend on FastAPI objects.

## Runtime Rules

Do not confuse these concepts:

- UI model profile controls rewrite and answer generation.
- UI model profile does not change retrieval embedding or Qdrant collection.
- Active embedding is `truro7/vn-law-embedding`.
- Active shared collection is `legal_documents__truro7_vn_law_embedding__d768`.
- `search_mode=user|hybrid` uses user-wide retrieval by default via `user_id`.
- `session_id` is retained for upload lineage, history, status, artifacts, and debug metadata; it is not the default retrieval filter.
- `Rewrite + Combine` always keeps the original query in the retrieval query set.

Keep these trace fields stable unless changing the public contract deliberately:

- `rewrite_context`
- `rewrite_debug`
- `retrieval_query_set`
- `retrieval_query_plan`
- `query_facets`
- `per_subquery_hits`
- `retrieval_union_count`

## Backend Editing Rules

When touching backend code:

- Prefer small changes around the use-case being fixed.
- Avoid broad cleanup in `web_app/backend/main.py`.
- If changing request/response shape, update `web_app/backend/api/schemas.py`, adapter payloads, frontend consumers, and tests together.
- Validate Qdrant/runtime contract paths when changing collection, embedding, sparse search, or payload fields.
- Keep fallback behavior for provider/rewrite failures; endpoints should not crash when remote LLM calls fail.

High-risk backend files:

- `web_app/backend/main.py`
- `web_app/backend/services/shared_legal_services.py`
- `src/legal_rag/pipeline/orchestrator.py`
- `src/legal_rag/stores/qdrant/repository.py`
- `src/legal_query_rewriter.py`
- `src/legal_chunker_v2.py`

## Frontend Editing Rules

When touching frontend code:

- Main shell/state/API calls are in `web_app/frontend/src/App.jsx`.
- Shared legal UI is in `web_app/frontend/src/components/LegalLabArea.jsx`.
- Answer/source/trace rendering is in `web_app/frontend/src/components/MessageBubble.jsx`.
- Preserve localStorage keys unless migration is intentional.
- Do not hide trace data that backend returns for rewrite/retrieval debugging.
- Keep Search Lab and Chat Lab behavior distinct: Search Lab debugs retrieval, Chat Lab generates answers.

Frontend build check:

```powershell
cd web_app\frontend
npm run build
```

## Test Commands

Run focused checks by change type:

```powershell
pytest tests/unit/test_backend_adapter.py tests/unit/test_backend_application_services.py
pytest tests/unit/test_legal_rag_pipeline_search.py tests/unit/test_legal_rag_pipeline_orchestrator_split.py
pytest tests/unit/test_legal_query_rewriter.py tests/unit/test_rewrite_modularization.py
pytest tests/unit/test_document_ingestion_chunking.py tests/unit/test_document_ingestion_indexer.py
pytest tests/unit/test_repo_hygiene.py
```

Compile check:

```powershell
python -m py_compile src\legal_query_rewriter.py src\legal_rag_pipeline.py src\adapters\backend_adapter.py src\services\remote_llm_client.py web_app\backend\main.py
```

Wide check before major merge:

```powershell
pytest tests/unit tests/integration
cd web_app\frontend
npm run build
```

Some integration/cloud tests require valid Qdrant and model endpoint configuration. If env access is missing, report that explicitly instead of treating it as a code failure.

## Data And Git Hygiene

Never commit:

- `.env`
- secrets or API keys
- `.venv/`
- `node_modules/`
- `web_app/frontend/dist/`
- `.pytest_cache/`
- `__pycache__/`
- local upload artifacts under `data/uploads/`
- large generated research artifacts unless explicitly requested

Live docs belong in `docs/`. Historical plans, runbooks, and snapshots belong under `docs/archive/`.

## Common Pitfalls

- Assuming old Chainlit/demo paths are the current production path.
- Breaking compatibility facades while tests still import them.
- Changing retrieval filter semantics from user-wide to session-only.
- Renaming trace fields without updating backend adapter, frontend, and tests.
- Treating rewrite as free-form paraphrase instead of objective extraction plus deterministic query synthesis.
- Updating README/runtime docs without updating `docs/LEGAL_PIPELINE_CURRENT_STATE.md`.

## Documentation Rule

If a change affects architecture, runtime settings, public API fields, collection naming, embedding model, retrieval behavior, rewrite behavior, or frontend trace display, update the relevant live docs in the same change.
