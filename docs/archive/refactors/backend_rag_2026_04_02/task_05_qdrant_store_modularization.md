# Task 05 - Qdrant Store Modularization

## Muc tieu

Tach `src/qdrant_store.py` thanh cac module doc lap: client factory, payload profiles, BM25 encoder, repository/search.

## Scope

Trong scope:

- payload shaping
- BM25 encoder
- qdrant client/runtime signature
- repository methods cho hybrid search

Ngoai scope:

- khong doi external Qdrant collection contract
- khong doi retrieval payload field names

## Proposed split

- `src/legal_rag/stores/qdrant/client_factory.py`
- `src/legal_rag/stores/qdrant/payload_profiles.py`
- `src/legal_rag/stores/qdrant/bm25_encoder.py`
- `src/legal_rag/stores/qdrant/repository.py`

Facade compatibility:

- `src/qdrant_store.py` van expose:
  - `SearchResult`
  - `BM25Encoder`
  - `QdrantLegalStore`
  - `get_qdrant_store()`
  - `reset_qdrant_store()`

## RED

Them/cap nhat tests cho:

1. runtime signature cache recreation
2. cloud/local client init
3. payload lean/full profile
4. BM25 save/load/encode behavior
5. metadata filter builder neu chua co test

Recommended tests:

- `test_build_qdrant_payload_lean_profile_preserves_required_fields`
- `test_qdrant_store_build_filter_for_active_only`
- `test_bm25_encoder_roundtrip`
- `test_get_qdrant_store_recreates_on_signature_change`

## GREEN

Implementation steps:

1. move payload helpers sang module rieng
2. move BM25Encoder sang module rieng
3. move client init/runtime signature sang client factory
4. move repository/search methods sang repository class
5. giu `QdrantLegalStore` la facade gom cac thanh phan tren

## REFACTOR

- dat interface ro giua payload policy va repository
- remove env reading lap lai
- nhom cac method collection ops vs search ops

## Exit criteria

- co test doc lap cho payload va BM25
- facade cu van hoat dong
- runtime cache behavior khong doi

## Dependency

- Task 01
- Task 04
