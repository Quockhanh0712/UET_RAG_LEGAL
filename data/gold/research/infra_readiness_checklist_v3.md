# Infra Readiness Checklist (Sprint A)

Updated: `2026-03-12`

## Local / Dev
| item | expected | current | status | note |
|---|---|---|---|---|
| Python package `qdrant_client` | installed | installed | done | upgraded and validated in runtime |
| Qdrant endpoint | reachable (`localhost:6333` or local path) | reachable | done | local client init and collection create OK |
| Neo4j endpoint | reachable | not verified | pending | currently graph variants marked pending |
| Embedding runtime | sentence-transformers importable | pass | done | text-only stub applied for torchvision/TF blockers |
| Benchmark script contract | extended columns written | pass | done | `benchmark_results.csv` migrated to v3 contract |
| A/B smoke benchmark (`AITeamVN` vs `DEk21`) | `B0..B3` run without infra error | pass | done | `run_20260312_033429` |

## Cloud (Managed)
| item | expected | current | status | note |
|---|---|---|---|---|
| Qdrant Cloud URL/API key | available in env | available and tested | done | uploaded 2 collections and listed via `get_collections()` |
| Neo4j Aura URI/user/password | available in env | not configured | pending | required for Sprint D |
| Collection naming policy | `legal_documents__{model_slug}__d{dim}` | implemented in code | done | apply when indexing per model |
| Default runtime routing | auto route by `model+dim` | enabled | done | `QDRANT_COLLECTION_BY_MODEL=true` |
| Rollback profile | env flag to local | local fallback available | in_progress | promote to deployment runbook in Sprint D |

## Cloud Collections (Current)
| collection_name | embedding_model | dim | points_count | note |
|---|---|---:|---:|---|
| `legal_documents__aiteamvn_vietnamese_embedding__d1024` | `AITeamVN/Vietnamese_Embedding` | 1024 | 80 | sample upload |
| `legal_documents__huyydangg_dek21_hcmute_embedding__d768` | `huyydangg/DEk21_hcmute_embedding` | 768 | 80 | sample upload |
