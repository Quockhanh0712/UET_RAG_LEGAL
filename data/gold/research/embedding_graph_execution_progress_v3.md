# Embedding + Graph Execution Progress (v3.1)

Updated: `2026-03-12`

## Objective
- Execute research plan v3.1 for legal retrieval with priority on citation precision.
- Complete embedding benchmark matrix, chunking compliance, graph auditability, and cloud rollout readiness.

## Locked Decisions
- Data scope: `gold_primary + gold_archive`.
- Extraction: deterministic/rule-based for chunking/graph in V1.
- Retrieval target profile: `G2 + C3` vs `B3`.
- Cloud direction: `Qdrant Cloud + Neo4j Aura`.
- Embedding policy: self-host, permissive OSS only.

## Current Status Snapshot
- Chunking quality (latest):
  - provision_id_valid: `100.00%`
  - structured_chunk_rate: `99.82%`
  - citation_full_rate: `100.00%`
  - non_cross_dieu_rate: `100.00%` (gate reached)
- Graph relation audit:
  - sample_size: `162`
  - judged_samples: `0`
  - precision_core: `0.00%` (manual labeling pending)
- Benchmark matrix:
  - existing statuses observed: `ok`, `graph-pending`, `pending-chunk-index`
  - latest smoke run before model switch: `run_20260312_030426` => `B0..B3=ok`, `G1..G3=graph-pending`
  - latest A/B smoke run on cloud: `run_20260312_033429` (`max_queries=3`) => both `AITeamVN` and `DEk21` with `B0..B3=ok`, `G1..G3=graph-pending`
- Cloud upload status:
  - `legal_documents__aiteamvn_vietnamese_embedding__d1024`: `80` points
  - `legal_documents__huyydangg_dek21_hcmute_embedding__d768`: `80` points

## Daily Log
| date | owner | task_id | input | output_artifact | metric_delta | blocker | next_action |
|---|---|---|---|---|---|---|---|
| 2026-03-12 | codex | A1 | plan v3.1 | `src/embeddings.py` | n/a | none | finish multi-model benchmark wiring |
| 2026-03-12 | codex | A2 | plan v3.1 | `src/qdrant_store.py` | n/a | none | validate collection-per-model workflow |
| 2026-03-12 | codex | A3 | plan v3.1 | `scripts/run_research_benchmark_v2.py` | output contract extended | infra local not fully ready | run smoke benchmark |
| 2026-03-12 | codex | A4 | plan v3.1 | `src/legal_chunker_v2.py` + `scripts/evaluate_chunking_quality_v2.py` | metadata expanded | none | rerun chunking quality |
| 2026-03-12 | codex | A5 | plan v3.1 | docs + research artifacts | docs aligned | none | run compile + smoke checks |
| 2026-03-12 | codex | A6 | smoke benchmark | `data/gold/research/benchmark_results.csv` | output contract migrated + new rows | qdrant package issue in non-venv env | rerun in `.venv` |
| 2026-03-12 | codex | A7 | chunk quality rerun | `data/gold/research/chunk_quality_report.md` | non_cross_dieu 99.60% -> 100.00% | none | move to graph/infra blockers |
| 2026-03-12 | codex | A8 | cloud upload test | `scripts/upload_qdrant_cloud_embeddings.py` + Qdrant Cloud collections | uploaded 2 embedding collections | first attempt timeout on large batch | reduce batch and retry |
| 2026-03-12 | codex | A9 | qdrant infra compatibility | `src/qdrant_store.py` | legacy client fallback enabled | none | continue benchmark on cloud |
| 2026-03-12 | codex | A10 | qdrant snippet verification | cloud `get_collections()` output | pass | none | proceed to larger upload/index |
| 2026-03-12 | codex | A11 | cloud stats fallback | `src/qdrant_store.py` + upload script rerun | points_count/status returned correctly | none | scale upload volume |
| 2026-03-12 | codex | A12 | embedding primary switch plan | `src/embeddings.py` + `.env` + `src/legal_rag_pipeline.py` | default model -> `AITeamVN`, batch -> `8`, normalize default true, model-aware collection routing | none | run A/B smoke on cloud |
| 2026-03-12 | codex | A13 | A/B smoke benchmark | `data/gold/research/benchmark_results.csv` (`run_20260312_033429`) | both models `B0..B3=ok` | Neo4j unavailable -> graph pending | continue Sprint C graph audit |

## Experiment Runs
- Completed smoke run:
  - `python scripts/run_research_benchmark_v2.py --max-queries 3 --embedding-models "AITeamVN/Vietnamese_Embedding,huyydangg/DEk21_hcmute_embedding" --normalize-options true --collection-by-model`
  - run id: `run_20260312_033429`

## Paper Survey Ledger
- Embedding references in active shortlist:
  - `AITeamVN/Vietnamese_Embedding`
  - `huyydangg/DEk21_hcmute_embedding`
  - `AITeamVN/Vietnamese_Embedding_v2`
  - `BAAI/bge-m3`
  - `intfloat/multilingual-e5-large-instruct`
  - `Alibaba-NLP/gte-multilingual-base`

## Risk / Blocker
- Graph precision gate blocked until 200 relation samples are manually labeled.
- Graph variants remain `graph-pending` until Neo4j is connected.
- Multi-dimension collections require strict model-specific indexing before fair comparisons.
- Cloud upsert can timeout for large batches; use smaller batch size or higher timeout.

## Next 48h
1. Increase cloud upload from sample (`80 chunks`) to target corpus per model.
2. Run extended A/B benchmark (`max_queries=60`) for stronger signal.
3. Complete Neo4j connectivity and convert `graph-pending` to measured `G1..G3`.
4. Start manual labeling for 200 graph relation samples.
| 2026-03-12 | codex | A14 | hybrid production run `run_20260312_063309` | `scripts/run_hybrid_index_production.py` | pilot completed for 2 models, winner=`truro7/vn-law-embedding` | none | full run/cutover follow summary `data\gold\research\index_runs\run_20260312_063309\run_summary.json` |

## Update 2026-03-12 (A15)
- Root cause fixed for benchmark `infra-pending`:
  - benchmark process was defaulting to local Qdrant due early import + missing cloud args propagation.
  - patch applied in:
    - `scripts/run_research_benchmark_v2.py`: add `--qdrant-url/--qdrant-api-key`, configure env before importing pipeline.
    - `scripts/run_hybrid_index_production.py`: pass cloud args to benchmark subprocess and force offline cache (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- Validation (cloud smoke):
  - `run_20260312_065413` (`DEk21`, `max_queries=1`): `B0..B3=ok`.
  - `run_20260312_065519` (`truro7`, `max_queries=1`): `B0..B3=ok`.
- Current blocker:
  - quality metrics still empty in smoke runs because sample query lacks labels (`notes: 1/1 queries missing labels`).
- Next action:
  - run full labeled benchmark (`max_queries=60`) in same profile to produce gate metrics for promotion/rollback decision.

## Update 2026-03-12 (A16)
- Full upload to Qdrant Cloud completed for production collection:
  - model: `truro7/vn-law-embedding`
  - collection: `legal_documents__truro7_vn_law_embedding__d768`
  - scope: `gold_primary=1255`, chunk policy `C3`, normalize `true`
  - indexed chunks: `155,938`
  - quality gate during upload: `provision_id=100%`, `structured=99.90%`, `citation_full=100%`, `non_cross_dieu=100%`
  - smoke hybrid query: `3/3` pass (`hits=15`)
- Post-upload verification:
  - collection exists on cloud and `points_count=155,938`
  - collection status: `green`

## Update 2026-03-12 (A17)
- Manual query smoke (10 Vietnamese legal queries) executed on cloud:
  - endpoint: `qdrant cloud europe-west3`
  - model: `truro7/vn-law-embedding`
  - collection: `legal_documents__truro7_vn_law_embedding__d768`
  - retrieval profile: `hybrid dense=0.6 + sparse=0.4`, `normalize=true`, filter `is_active=true`, `top_k=5`
- Artifacts:
  - `data/gold/research/index_runs/run_20260312_063309/manual_query_test_truro7_cloud_basic.json`
  - `data/gold/research/index_runs/run_20260312_063309/manual_query_test_truro7_cloud_basic.md`
- Quick signal:
  - Most intent-specific queries returned relevant legal families at top-1/top-2 (`giao thông`, `thi hành án`, `đấu thầu`, `công nghiệp an ninh`).
  - Repeated off-topic hit `07-TT/LBTL-ĐC` still appears in multiple queries and should be handled in next ranking cleanup pass.

## Update 2026-03-12 (A18)
- Chainlit direct-pipeline demo integration completed:
  - new files:
    - `apps/chainlit_demo.py`
    - `scripts/smoke_chainlit_hybrid.py`
    - `docs/CHAINLIT_HYBRID_DEMO_RUNBOOK.md`
    - `requirements-chainlit.txt`
  - runtime lock used in demo/smoke:
    - model: `truro7/vn-law-embedding`
    - collection: `legal_documents__truro7_vn_law_embedding__d768`
    - retrieval: `dense=0.6`, `sparse=0.4`, `normalize=true`, `top_k=5`, `reranker=true`, `graph=false`
- Dependency validation:
  - `chainlit==2.10.0` installed in `.venv`
  - `pip check` passed (`No broken requirements found`)
- Smoke run executed (10 queries, non-UI):
  - run id: `run_20260312_150920`
  - artifact:
    - `data/gold/research/index_runs/run_20260312_150920/chainlit_smoke_10q.json`
    - `data/gold/research/index_runs/run_20260312_150920/chainlit_smoke_10q.md`
  - result summary:
    - `status_ok=10/10`, `fallback_count=0`, `hard_fail_count=0`
    - `retrieval_p50_ms=3389.72`, `retrieval_p95_ms=3738.75`
    - `generation_p50_ms=18919.63`, `generation_p95_ms=29184.62`
    - gates: `>=3 results/query=true`, `top1 citation non-empty >=9/10=true`, `LLM non-empty or fallback >=9/10=true`
- Current note:
  - Retrieval stability is good, but relevance drift from repeated off-topic hits still exists on some queries; next pass should tune rerank/metadata constraints without changing scope.

## Update 2026-03-12 (A19)
- Chainlit startup hotfix applied:
  - issue: `ModuleNotFoundError: No module named 'src'` when running `chainlit run apps/chainlit_demo.py`.
  - fix: add `PROJECT_ROOT` to `sys.path` in `apps/chainlit_demo.py` before importing pipeline modules.
- Validation:
  - chainlit run command now holds process until manual timeout (no immediate import crash), indicating app boot path is healthy.

## Update 2026-03-12 (A20)
- Demo retrieval path aligned with backend profile (`web_app/backend/main.py`) for RRF + reranking:
  - reranker model locked to `AITeamVN/Vietnamese_Reranker`
  - rerank alpha locked to `1.0`
  - smoke/contracts now expose `used_reranker` + `rerank_time_ms` for each query
- Patch scope:
  - `src/reranker.py`: default model and max_length/batch_size env defaults updated
  - `src/legal_rag_pipeline.py`: reranker init and alpha now driven by env (`RERANKER_MODEL`, `RERANKER_ALPHA`)
  - `apps/chainlit_demo.py`: lock demo runtime profile, include reranker metrics in UI output
  - `scripts/smoke_chainlit_hybrid.py`: add reranker metrics + gate `reranker_used_when_enabled`
- Cloud smoke validation:
  - run id: `run_20260312_153902`
  - artifact:
    - `data/gold/research/index_runs/run_20260312_153902/chainlit_smoke_10q.json`
    - `data/gold/research/index_runs/run_20260312_153902/chainlit_smoke_10q.md`
  - summary:
    - `status_ok=10/10`, `fallback_count=0`, `hard_fail_count=0`
    - `rerank_used_count=10`, `gate_reranker_used=true`
    - `retrieval_p95_ms=3635.99`, `rerank_p95_ms=1619.71`, `generation_p95_ms=38074.63`

## Update 2026-03-12 (A21)
- Startup blocker triaged from live demo run:
  - error: `Collection not found on cloud: legal_documents__truro7_vn_law_embedding__d768`
  - root cause: active `.env` cloud endpoint (`us-east`) currently contains collections `user_docs_private`, `legal_rag_vn` only.
- Applied compatibility patch:
  - add optional `CHAINLIT_COLLECTION_OVERRIDE` in demo/smoke to allow explicit collection selection when target cluster differs.
  - improve startup error text to include `url` + list of available collections + exact remediation.
- Validation:
  - without override: clear diagnostic error now shown.
  - with `CHAINLIT_COLLECTION_OVERRIDE=legal_rag_vn`: pipeline initialization succeeds.

## Update 2026-03-12 (A22)
- Legacy collection compatibility patch for strict-mode filter:
  - issue observed on `legal_rag_vn`: `400 Bad request` due missing payload index for `is_active`.
  - fix applied: when `is_active` filter fails, automatically retry retrieval without that filter.
- Validation:
  - smoke run `run_20260312_override_1q_fix` on `CHAINLIT_COLLECTION_OVERRIDE=legal_rag_vn` now completes query execution without hard retrieval crash.

## Update 2026-03-12 (A23)
- Cloud endpoint mismatch resolved for demo default path:
  - old endpoint in `.env` (`us-east`) did not contain `legal_documents__truro7_vn_law_embedding__d768`.
  - switched demo runtime endpoint to cluster containing indexed production collections (`europe-west3`).
- Verification:
  - `get_collections()` now returns:
    - `legal_documents__truro7_vn_law_embedding__d768`
    - `legal_documents__huyydangg_dek21_hcmute_embedding__d768`
    - corresponding pilot shadow collections.
  - direct pipeline init without override succeeds and resolves collection `legal_documents__truro7_vn_law_embedding__d768`.

## Update 2026-03-12 (A24)
- Full smoke re-run completed on target `truro7` collection with RRF + reranker enabled:
  - run id: `run_20260312_chainlit_fix_truro10q`
  - artifacts:
    - `data/gold/research/index_runs/run_20260312_chainlit_fix_truro10q/chainlit_smoke_10q.json`
    - `data/gold/research/index_runs/run_20260312_chainlit_fix_truro10q/chainlit_smoke_10q.md`
  - summary:
    - `status_ok=10/10`, `fallback_count=0`, `hard_fail_count=0`
    - `retrieval_p50_ms=3103.82`, `retrieval_p95_ms=3597.29`
    - `rerank_p50_ms=1265.77`, `rerank_p95_ms=1719.36`, `rerank_used_count=10`
    - gates: `gate_min3_results=true`, `gate_top1_citation=true`, `gate_llm_or_fallback=true`, `gate_reranker_used=true`
- Demo runbook stabilization:
  - default command changed to run Chainlit without hot reload (`-w`) to avoid Engine.IO polling noise during live demo.
