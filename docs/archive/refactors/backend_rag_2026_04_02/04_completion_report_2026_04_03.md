# Completion Report - 8/8 Tasks

Updated: `2026-04-03`

## Executive summary

Refactor roadmap in this folder is completed end-to-end:

- task 01: done
- task 02: done
- task 03: done
- task 04: done
- task 05: done
- task 06: done
- task 07: done
- task 08: done

Public facade contracts were kept stable while internal modules were split by responsibility.

## Scope delivered in this session

### Task 05 (qdrant store modularization)

- Added modular qdrant package:
  - `src/legal_rag/stores/qdrant/client_factory.py`
  - `src/legal_rag/stores/qdrant/payload_profiles.py`
  - `src/legal_rag/stores/qdrant/bm25_encoder.py`
  - `src/legal_rag/stores/qdrant/repository.py`
- Kept compatibility facade in `src/qdrant_store.py`
- Added/updated tests:
  - `tests/unit/test_qdrant_bm25_encoder.py`
  - `tests/unit/test_qdrant_filter_builder.py`

### Task 06 (query rewrite modularization)

- Added rewrite package:
  - `src/legal_rag/rewrite/models.py`
  - `src/legal_rag/rewrite/gateway.py`
  - `src/legal_rag/rewrite/heuristics.py`
  - `src/legal_rag/rewrite/parser.py`
  - `src/legal_rag/rewrite/synthesis.py`
  - `src/legal_rag/rewrite/quality_gate.py`
- `src/legal_query_rewriter.py` now orchestrates via these modules (facade contract preserved)
- Added/updated tests:
  - `tests/unit/test_rewrite_modularization.py`
  - `tests/unit/test_legal_query_rewriter.py`

### Task 07 (chunking and ingestion boundaries)

- Added chunking package:
  - `src/legal_rag/chunking/policy.py`
  - `src/legal_rag/chunking/provision_chunker.py`
  - `src/legal_rag/chunking/metadata_builder.py`
- `src/legal_chunker_v2.py` now delegates policy/window/id logic to chunking modules
- Added tests:
  - `tests/unit/test_chunking_modularization.py`

### Task 08 (runtime wiring and cleanup)

- Regression gates executed and fixed issues found in wide system check:
  - Added test dependency: `python-multipart==0.0.20`
  - `web_app/backend/user_store.py` updated for qdrant client compatibility (no hard dependency on `models.Modifier`)
  - `web_app/backend/user_store.py` collection creation now idempotent on "already exists"
  - `src/services/model_profile_registry.py`: `list_model_profiles` default adjusted for testable preset visibility
  - `web_app/backend/main.py`: explicit `enabled_only=True` on API responses to preserve runtime behavior

## Validation log

### Focused refactor regression suite

Command:

```bash
pytest -q tests/unit/test_qdrant_payload_profile.py tests/unit/test_qdrant_store_runtime.py tests/unit/test_qdrant_contract.py tests/unit/test_qdrant_bm25_encoder.py tests/unit/test_qdrant_filter_builder.py tests/unit/test_legal_query_rewriter.py tests/unit/test_query_rewrite_service.py tests/unit/test_rewrite_modularization.py tests/unit/test_legal_chunker_v2.py tests/unit/test_chunking_modularization.py tests/integration/test_backend_api_modularization.py tests/unit/test_backend_application_services.py tests/unit/test_legal_rag_pipeline_orchestrator_split.py tests/unit/test_legal_rag_pipeline_search.py tests/unit/test_runtime_settings.py tests/unit/test_core.py
```

Result:

- `44 passed, 20 skipped`

### Wide system check

Command:

```bash
pytest -q tests/unit tests/integration
```

Final result:

- `78 passed, 20 skipped`

## Notes

- Remaining output warnings are non-blocking for this refactor scope:
  - pydantic protected namespace warnings (`model_*` fields)
  - sqlalchemy 2.0 deprecation warning (`declarative_base`)
  - swig deprecation warnings from tokenizer/native deps
