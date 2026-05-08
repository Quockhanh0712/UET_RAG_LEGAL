# Quick Chat Demo Runbook

## Purpose

Use this app when you need a fast UI to test:

- hybrid retrieval
- citation quality
- query rewrite on/off
- reranker on/off

without the overhead of Chainlit.

## Runtime

- URL: `http://127.0.0.1:8010`
- App: [quick_chat_demo.py](D:\Source_gihub\codeX_project\apps\quick_chat_demo.py)
- Shared runtime:
  - [runtime_settings.py](D:\Source_gihub\codeX_project\src\config\runtime_settings.py)
  - [qdrant_contract.py](D:\Source_gihub\codeX_project\src\runtime\qdrant_contract.py)
  - [legal_pipeline_factory.py](D:\Source_gihub\codeX_project\src\factories\legal_pipeline_factory.py)

## Start

```powershell
.venv\Scripts\python.exe -m uvicorn apps.quick_chat_demo:app --host 127.0.0.1 --port 8010
```

## Modes

- `Search only`
  - Fastest mode.
  - Uses retrieval + optional rewrite + optional reranker.
  - Does not call the LLM.
  - Recommended for checking whether citations and top hits are correct.

- `Full QA`
  - Uses the same retrieval path, then calls the LLM.
  - Much slower.
  - Use only after retrieval looks reasonable.

## Suggested test flow

1. Start in `Search only`.
2. Keep `Use reranker = on`.
3. Test once with `Enable requery = off`.
4. Test again with `Enable requery = on` for complex multi-actor legal questions.
5. Only move to `Full QA` when citations/top documents look sane.

## Health check

```powershell
Invoke-RestMethod http://127.0.0.1:8010/health
```

Expected:

- `ok = true`
- collection = `legal_documents__truro7_vn_law_embedding__d768`
- collection_status = `green` or queryable `yellow`

## API smoke

```powershell
$body = @{
  message = "Cong an xa xu phat loi khong mang bang lai xe co dung khong?"
  mode = "search"
  top_k = 5
  use_reranker = $true
  use_graph = $false
  require_active_only = $true
  query_rewriter_enabled = $false
  query_rewriter_force = $false
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8010/api/chat `
  -ContentType "application/json" `
  -Body $body
```

## Current note

- The app path is healthy and fast enough for retrieval demo.
- Retrieval relevance is still not where it needs to be on some legal queries.
- Therefore use this UI to debug retrieval/citation first, not as a quality sign-off tool yet.
