# UET RAG Legal

Vietnamese legal RAG system with:
- shared legal pipeline
- hybrid retrieval (dense + sparse)
- query rewrite V2
- reranker
- FastAPI backend + React frontend

## 1. Current default runtime

The current shared legal demo defaults to:

- model profile: `qwen3_5_9b_ollama_remote`
- model: `qwen3.5:9b`
- provider: `ollama_generate`
- endpoint: `http://222.255.250.24:11434`
- embedding: `truro7/vn-law-embedding`
- collection: `legal_documents__truro7_vn_law_embedding__d768`
- reranker: `AITeamVN/Vietnamese_Reranker`

## 2. Main docs

- system overview: `docs/LEGAL_SYSTEM_OVERVIEW.md`
- rewrite design: `docs/LEGAL_REWRITE_V2.md`
- current pipeline state: `docs/LEGAL_PIPELINE_CURRENT_STATE.md`

## 3. Repo layout

- `src/`: core legal pipeline, rewrite, retrieval, reranker, runtime services
- `web_app/backend/`: FastAPI backend
- `web_app/frontend/`: React frontend
- `scripts/`: indexing and research scripts
- `docs/`: project docs
- `tests/`: tests

## 4. Prerequisites

- Python `3.10+`
- Node.js `18+`
- a valid Qdrant Cloud configuration
- access to the Ollama-compatible endpoint that serves `qwen3.5:9b`

## 5. Python setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional:

```powershell
pip install -r requirements-chainlit.txt
```

## 6. Frontend setup

```powershell
cd web_app\frontend
npm install
cd ..\..
```

## 7. Required environment variables

Create `.env` from your local setup. The exact values depend on your deployment, but the current shared legal path expects at least the following:

```env
BACKEND_USE_SHARED_LEGAL_PIPELINE=true

QDRANT_URL=https://<your-qdrant-endpoint>
QDRANT_API_KEY=<your-qdrant-key>
QDRANT_COLLECTION_BASE=legal_documents

OLLAMA_QWEN35_ENABLED=true
OLLAMA_QWEN35_URL=http://222.255.250.24:11434
OLLAMA_QWEN35_MODEL=qwen3.5:9b
OLLAMA_QWEN35_TIMEOUT_S=120

LEGAL_TGI_ENABLED=false
OLLAMA_QWEN3_ENABLED=false
```

Notes:
- `BACKEND_USE_SHARED_LEGAL_PIPELINE=true` is mandatory for the shared legal backend path.
- The shared legal endpoints only activate when the request uses `search_mode=legal`.
- If you send `search_mode=hybrid`, the backend falls back to the legacy route.

## 8. Run backend

Recommended:

```powershell
$env:BACKEND_USE_SHARED_LEGAL_PIPELINE='true'
.\.venv\Scripts\python.exe -m uvicorn web_app.backend.main:app --host 127.0.0.1 --port 8080
```

Alternative:

```powershell
python web_app\backend\main.py
```

Backend URL:
- `http://127.0.0.1:8080`

Useful endpoints:
- `GET /api/status`
- `GET /api/settings/model-profiles`
- `POST /api/search`
- `POST /api/chat`

## 9. Run frontend

```powershell
cd web_app\frontend
npm run dev
```

Frontend URL:
- `http://127.0.0.1:3000`

## 10. Shared legal API usage

### Search

```json
POST /api/search
{
  "query": "Cong ty A ... xa thai ...",
  "search_mode": "legal",
  "top_k": 8,
  "use_reranker": true,
  "model_profile_id": "qwen3_5_9b_ollama_remote",
  "query_mode": "rewrite_combine",
  "require_active_only": true
}
```

### Chat

```json
POST /api/chat
{
  "message": "Cong ty A ... xa thai ...",
  "search_mode": "legal",
  "top_k": 6,
  "use_reranker": true,
  "model_profile_id": "qwen3_5_9b_ollama_remote",
  "query_mode": "rewrite_combine",
  "response_mode": "rag",
  "require_active_only": true,
  "use_graph": false
}
```

## 11. Query mode

- `original`
  - no rewrite
  - retrieval uses the original query only

- `rewrite_combine`
  - Rewrite V2 extracts:
    - `main_act`
    - `surface_actors`
    - `conditions`
    - `named_law_hints`
    - `question_units`
  - code synthesizes retrieval subqueries deterministically
  - the original query is always kept in the retrieval set

## 12. What Rewrite V2 does

Rewrite V2 is not a free-form paraphraser.

It uses this flow:

```text
user query
-> objective extraction by qwen3.5:9b
-> question-unit decomposition
-> deterministic query synthesis
-> quality gate
-> retrieval query set
```

The main rules are:
- `main_act` must be the legal action, not a side fact
- no early legal judgment in rewrite
- named laws are weak hints only
- one main question unit maps to one retrieval query
- total retrieval queries are capped and always include the original query

## 13. Rewrite and answer traces

Search and chat responses now expose:
- `rewrite_context`
- `rewrite_debug`
- `retrieval_query_set`
- `retrieval_query_plan`
- `query_facets`
- `per_subquery_hits`
- `retrieval_union_count`

The frontend shows:
- original vs rewrite
- query facets
- conditions
- clause units
- named law hints
- question units
- retrieval query plan

## 14. Development checks

Python compile check:

```powershell
python -m py_compile src\legal_query_rewriter.py src\legal_rag_pipeline.py src\adapters\backend_adapter.py src\services\remote_llm_client.py web_app\backend\main.py
```

Frontend build check:

```powershell
cd web_app\frontend
npm run build
```

## 15. Known limitations

Current bottleneck is still retrieval quality.

Rewrite V2 is more stable and more interpretable than the old rewrite path, but it cannot fix:
- weak retrieval ranking
- wrong top-1 document
- irrelevant evidence returned by retrieval

If retrieval misses the right law, the answer stage may still abstain with `CHUA_DU_CAN_CU`.

## 16. Git workflow

Recommended branch flow:
- create a feature branch
- keep one task per commit group
- run compile/build checks before commit

Never commit:
- `.env`
- secrets
- `node_modules`
- `.venv`
- large research artifacts unless explicitly needed