# Updated Research Plan v3.1 (Implementation Status)

Updated: `2026-03-12`

## Sprint Status
| sprint | objective | status | notes |
|---|---|---|---|
| Sprint A | Infra baseline + readiness | in_progress | qdrant cloud/local connectivity pass + sample upload done; Neo4j still pending |
| Sprint B | Embedding benchmark matrix | in_progress | multi-model benchmark script implemented, pending infra package/index |
| Sprint C | Chunking + graph optimization | in_progress | chunking gate reached (`non_cross_dieu=100%` by boundary metric); graph labeling still pending |
| Sprint D | Cloud rollout | pending | waiting local gate + credentials |
| Sprint E | Research artifact tracking | active | progress file and checklist created |

## What Has Been Implemented
- Embedding runtime now supports:
  - model override per run,
  - normalize option (`auto/true/false`),
  - E5 `query:`/`passage:` prefix handling,
  - dynamic embedding dimension query.
- Qdrant integration now supports:
  - dynamic `dense_dim`,
  - collection naming by model/dim (`legal_documents__{model_slug}__d{dim}`),
  - payload index extensions (`token_len_est`, `encoder_profile`).
- Chunking schema now includes:
  - `token_len_est`,
  - `encoder_profile`.
- Benchmark script now supports:
  - multi-embedding matrix runs,
  - extended output contract fields:
    - `embedding_model_id`,
    - `embedding_dim`,
    - `normalize_embeddings`,
    - `encode_p95_ms`,
  - legacy CSV auto-migration to new schema.
- Research artifacts/docs updated:
  - `README.md`,
  - `decision_log.md`,
  - `experiment_plan.yaml`,
  - `chunking_spec.md`,
  - progress tracker and infra checklist.

## Current Blockers
1. Neo4j local/cloud chưa kết nối nên `G1..G3` vẫn `graph-pending`.
2. Cloud upload batch lớn có thể timeout, cần batch policy ổn định cho full run.
3. Graph relation precision gate blocked until manual labels are filled.

## Immediate Next Actions
1. Expand cloud indexing from sample to full target set for 2 embeddings.
2. Validate Neo4j connectivity and rerun `G1..G3`.
3. Run shortlist embedding matrix with collection-by-model.
4. Start manual labeling for graph relation sample (`>=200` judged rows).
