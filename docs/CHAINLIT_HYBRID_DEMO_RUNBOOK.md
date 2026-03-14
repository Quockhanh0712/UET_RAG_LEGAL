# Chainlit Hybrid Demo Runbook (Qdrant Cloud AWS, Shared Runtime)

## 1) Setup

Use project venv:

```powershell
.venv\Scripts\Activate.ps1
```

Install demo dependencies:

```powershell
.venv\Scripts\pip.exe install -r requirements-chainlit.txt
```

Verify dependency health:

```powershell
.venv\Scripts\python.exe -c "import chainlit; print(chainlit.__version__)"
.venv\Scripts\pip.exe check
```

## 2) Runtime Env (do not commit secrets)

Set runtime env for this demo profile only (PowerShell session):

```powershell
$env:QDRANT_USE_CLOUD="true"
$env:QDRANT_CLOUD_URL="https://718fe745-9ec6-416e-ab74-88304640f37d.us-east-1-1.aws.cloud.qdrant.io:6333"
$env:QDRANT_API_KEY="<YOUR_QDRANT_API_KEY>"
$env:QDRANT_COLLECTION="legal_documents"
$env:QDRANT_COLLECTION_BY_MODEL="true"

$env:EMBEDDING_MODEL="truro7/vn-law-embedding"
$env:EMBEDDING_NORMALIZE="true"
$env:EMBEDDING_DEVICE="cuda"
$env:USE_FP16="true"

$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="qwen3:8b"
$env:OLLAMA_URL="http://222.255.250.24:11434"

$env:CHAINLIT_TOP_K="5"
$env:CHAINLIT_USE_RERANKER="true"
$env:CHAINLIT_USE_GRAPH="false"
$env:CHAINLIT_REQUIRE_ACTIVE_ONLY="true"

# Query rewrite is feature-flagged. Keep off by default, enable when testing
# complex multi-actor / multi-liability legal questions.
$env:QUERY_REWRITER_ENABLED="false"
$env:QUERY_REWRITER_URL="http://222.255.250.24:8080/v1/completions"
$env:QUERY_REWRITER_TIMEOUT_S="20"
$env:QUERY_REWRITER_MAX_SUBQUERIES="6"
$env:QUERY_REWRITER_FORCE="false"
# Optional fallback if your current cloud cluster does not have
# legal_documents__truro7_vn_law_embedding__d768:
# $env:CHAINLIT_COLLECTION_OVERRIDE="legal_rag_vn"

# Match backend reranker profile (web_app/backend/main.py)
$env:RERANKER_MODEL="AITeamVN/Vietnamese_Reranker"
$env:RERANKER_ALPHA="1.0"
$env:RERANKER_MAX_LENGTH="2304"
$env:RERANKER_BATCH_SIZE="8"
```

Optional BM25 contract checks:

```powershell
$env:CHAINLIT_BM25_PATH="data/qdrant_db/bm25_encoder.json"
# optional strict check:
# $env:CHAINLIT_BM25_MANIFEST="data/gold/research/index_runs/<run_id>/bm25_manifest.json"
```

## 3) Preflight Qdrant

```powershell
.venv\Scripts\python.exe -c "from qdrant_client import QdrantClient;import os; c=QdrantClient(url=os.environ['QDRANT_CLOUD_URL'],api_key=os.environ['QDRANT_API_KEY']); print([x.name for x in c.get_collections().collections])"
```

Target serving collection for this profile:

`legal_documents__truro7_vn_law_embedding__d768`

## 4) Run Chainlit Demo

```powershell
.venv\Scripts\chainlit run apps/chainlit_demo.py
```

Open the local Chainlit URL in terminal output (default `http://localhost:8000`).

Note:
- Avoid `-w` during demo. Auto-reload can trigger noisy Engine.IO polling errors (`Too many packets in payload`) while app is reloading.
- This app now uses the shared runtime loader + Qdrant contract validation before chat is available.
- If startup fails, fix the contract error first instead of bypassing it in UI code.

## 5) Run Smoke Test (10 Queries)

```powershell
.venv\Scripts\python.exe scripts/smoke_chainlit_hybrid.py
```

Expected artifacts:

- `data/gold/research/index_runs/<run_id>/chainlit_smoke_10q.json`
- `data/gold/research/index_runs/<run_id>/chainlit_smoke_10q.md`

## 6) Pass Criteria

- 10/10 queries no hard crash.
- Each query returns at least 3 results.
- Top-1 `citation_full` non-empty for >= 9/10 queries.
- LLM answer non-empty or fallback retrieval present for >= 9/10 queries.
- Retrieval p95 stable with no cloud timeout spikes.

If BM25 artifact mismatch is detected (when manifest check enabled), stop run and fix mapping before rerun.
If the serving collection is `red` or returns `503`, stop demo and fix Qdrant storage/optimizer state before retrying smoke.
