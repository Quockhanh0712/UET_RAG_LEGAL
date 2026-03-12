# Decision Log

Updated: `2026-03-12 16:45:00`

## Locked Defaults
- Data scope: `gold_primary + gold_archive` only.
- Priority: citation precision first.
- Extraction policy: deterministic/rule-based only (no LLM extraction for chunking/graph).
- Graph rollout: document-level first, provision-level in phase 2 after gate pass.
- Fallback policy: if Neo4j unavailable, run baseline/hybrid variants and mark graph variants `graph-pending`.
- Cloud direction: managed (`Qdrant Cloud + Neo4j Aura`) after local gate pass.
- Embedding deployment: self-host SentenceTransformers/TEI, permissive OSS license only.

## Active Recommendation
- Retrieval profile: `B3` baseline + `G2 + C3` target.
- Graph mode for legal-impact queries: `legal_impact`.
- Chunking mode target: `ProvisionChunk + ClauseChunk` with strict citation metadata.
- Embedding primary: `AITeamVN/Vietnamese_Embedding` (d1024).
- Embedding fallback: `huyydangg/DEk21_hcmute_embedding` (d768).
- Runtime embedding defaults: `EMBEDDING_DEVICE=cuda`, `EMBEDDING_BATCH_SIZE=8`, `EMBEDDING_NORMALIZE=true`.
- A/B smoke benchmark status: `run_20260312_033429` -> both models pass `B0..B3` on cloud.
- Embedding shortlist includes:
  - `huyydangg/DEk21_hcmute_embedding`
  - `AITeamVN/Vietnamese_Embedding`
  - `AITeamVN/Vietnamese_Embedding_v2`
  - `BAAI/bge-m3`
  - `intfloat/multilingual-e5-large-instruct`
  - `Alibaba-NLP/gte-multilingual-base`
- Qdrant collection naming: `legal_documents__{model_slug}__d{dim}`.

## Promotion Conditions
- `Recall@20` +5% or `nDCG@10` +3% over `B3`.
- `citation_valid_rate >= 95%`.
- `latency_p95_ms` increase <= 35%.
