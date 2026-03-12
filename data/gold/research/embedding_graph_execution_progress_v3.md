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
