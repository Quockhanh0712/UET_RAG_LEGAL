# Env Contract and Runtime Profiles

## Shared Env Contract

### Qdrant
- `QDRANT_USE_CLOUD`
- `QDRANT_CLOUD_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `QDRANT_COLLECTION_BY_MODEL`

### Embedding Runtime
- `EMBEDDING_MODEL`
- `EMBEDDING_NORMALIZE`
- `EMBEDDING_DEVICE`
- `USE_FP16`

### Reranker
- `RERANKER_MODEL`
- `RERANKER_ALPHA`
- `RERANKER_MAX_LENGTH`
- `RERANKER_BATCH_SIZE`

### Rewriter
- `QUERY_REWRITER_ENABLED`
- `QUERY_REWRITER_URL`
- `QUERY_REWRITER_TIMEOUT_S`
- `QUERY_REWRITER_MAX_SUBQUERIES`
- `QUERY_REWRITER_FORCE`

### App Flags
- `CHAINLIT_TOP_K`
- `CHAINLIT_USE_RERANKER`
- `CHAINLIT_USE_GRAPH`
- `CHAINLIT_REQUIRE_ACTIVE_ONLY`
- `BACKEND_USE_SHARED_LEGAL_PIPELINE`

### Optional Strict Validation
- `CHAINLIT_BM25_PATH`
- `CHAINLIT_BM25_MANIFEST`
- `QDRANT_CONTRACT_STRICT`

## Runtime Resolution Rules
- Base collection là `QDRANT_COLLECTION`.
- Nếu `QDRANT_COLLECTION_BY_MODEL=true`, runtime phải resolve sang collection model-specific.
- Model production profile của vòng này là `truro7/vn-law-embedding`.
- Resolved collection target là `legal_documents__truro7_vn_law_embedding__d768`.

## Default Profiles

### Chainlit Demo
- `QUERY_REWRITER_ENABLED=false`
- `CHAINLIT_USE_RERANKER=true`
- `CHAINLIT_USE_GRAPH=false`
- `BACKEND_USE_SHARED_LEGAL_PIPELINE` không áp dụng

### Chainlit Smoke for Complex Queries
- `QUERY_REWRITER_ENABLED=true`
- `QUERY_REWRITER_FORCE=false`
- strict contract check có thể bật

### Backend Dev/Smoke
- `BACKEND_USE_SHARED_LEGAL_PIPELINE=false` ban đầu
- `QUERY_REWRITER_ENABLED=false` cho đến khi parity pass

## Required Runtime Types
- `RuntimeSettings`
  - `qdrant_url`
  - `qdrant_api_key`
  - `collection_base`
  - `collection_resolved`
  - `embedding_model`
  - `normalize_embeddings`
  - `device`
  - `use_fp16`
  - `reranker_model`
  - `rewrite_enabled`
  - `rewrite_url`
  - `rewrite_timeout_s`
- `QdrantContractStatus`
  - `collection_name`
  - `collection_status`
  - `embedding_model_id`
  - `vector_dim`
  - `bm25_manifest_id`
  - `contract_ok`
  - `contract_error`

## Validation Rules
- Thiếu `QDRANT_CLOUD_URL` hoặc `QDRANT_API_KEY` khi cloud mode bật là hard fail.
- Collection không resolve được là hard fail.
- Collection `red` là hard fail.
- Collection `yellow` chỉ được chấp nhận nếu query smoke pass.
- BM25 manifest mismatch là hard fail khi strict mode bật.
