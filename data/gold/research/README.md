# Research Workspace (v3.1)

This folder contains implementation artifacts for:
- legal chunking quality and metadata compliance
- hybrid + graph benchmark matrix across multiple embedding models
- graph relation auditability and decision tracking

## Core Artifacts
- `evidence_matrix.csv`
- `graph_schema_spec.md`
- `chunking_spec.md`
- `experiment_plan.yaml`
- `research_survey_report.md`
- `decision_log.md`
- `query_set_v1.csv` (60 queries)
- `eval_pairs_template.csv` (300 query-doc labeling rows)
- `benchmark_results.csv` (extended benchmark contract output)
- `embedding_graph_execution_progress_v3.md` (daily execution progress log)
- `infra_readiness_checklist_v3.md` (Sprint A readiness tracker)
- `updated_research_plan_v3_1.md` (implementation status view by sprint)

## Commands

Bootstrap/rebuild research artifacts:

```powershell
python scripts/bootstrap_research_v2_artifacts.py
```

Evaluate chunking gates:

```powershell
python scripts/evaluate_chunking_quality_v2.py --input-path data/gold/vbpl_gold.parquet
```

Prepare/evaluate graph relation labels:

```powershell
python scripts/evaluate_graph_relations_v2.py --input-path data/gold/vbpl_gold.parquet --sample-size 200
```

Run benchmark matrix for one model (current collection):

```powershell
python scripts/run_research_benchmark_v2.py `
  --queries-path data/gold/research/query_set_v1.csv `
  --labels-path data/gold/research/eval_pairs_template.csv `
  --embedding-models AITeamVN/Vietnamese_Embedding `
  --normalize-options true `
  --collection-by-model
```

Run benchmark matrix for shortlist with model-specific collections:

```powershell
python scripts/run_research_benchmark_v2.py `
  --queries-path data/gold/research/query_set_v1.csv `
  --labels-path data/gold/research/eval_pairs_template.csv `
  --embedding-models "AITeamVN/Vietnamese_Embedding,huyydangg/DEk21_hcmute_embedding,AITeamVN/Vietnamese_Embedding_v2,BAAI/bge-m3,intfloat/multilingual-e5-large-instruct,Alibaba-NLP/gte-multilingual-base" `
  --normalize-options both `
  --collection-by-model
```

Upload sample chunks to Qdrant Cloud for selected embeddings:

```powershell
python scripts/upload_qdrant_cloud_embeddings.py `
  --input-path data/gold/vbpl_gold.parquet `
  --limit-docs 3 `
  --max-chunks 80 `
  --models "AITeamVN/Vietnamese_Embedding,huyydangg/DEk21_hcmute_embedding" `
  --qdrant-url "<QDRANT_CLOUD_URL>" `
  --qdrant-api-key "<QDRANT_API_KEY>" `
  --embedding-batch-size 8 `
  --force-recreate
```

## Benchmark Output Contract (`benchmark_results.csv`)
- `run_id`
- `variant_id`
- `embedding_model_id`
- `embedding_dim`
- `normalize_embeddings`
- `status`
- `num_queries`
- `Recall@20`
- `nDCG@10`
- `MRR@10`
- `citation_valid_rate`
- `latency_p95_ms`
- `encode_p95_ms`
- `notes`
- `created_at`

## Status Semantics (`status`)
- `ok`: run completed with metrics
- `graph-pending`: Neo4j unavailable for graph variants
- `infra-pending`: retrieval backend unavailable
- `pending-chunk-index`: chunk variant requires dedicated re-indexed collection
