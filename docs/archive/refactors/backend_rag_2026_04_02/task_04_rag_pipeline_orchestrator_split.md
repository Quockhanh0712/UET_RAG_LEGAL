# Task 04 - RAG Pipeline Orchestrator Split

## Muc tieu

Lam `LegalRAGPipeline` thanh facade/orchestrator mong, khong om ca embedding, llm, retrieval formatting, singleton detail trong mot file lon.

## Scope

Trong scope:

- tach result DTOs
- tach context builder
- tach retrieval orchestration
- tach answering gateway/prompt builder khoi pipeline facade

Ngoai scope:

- chua tach qdrant store internals
- chua tach query rewrite internals

## Proposed split

- `src/legal_rag/contracts/results.py`
- `src/legal_rag/pipeline/orchestrator.py`
- `src/legal_rag/pipeline/context_builder.py`
- `src/legal_rag/embeddings/encoder.py`
- `src/legal_rag/answering/llm_gateway.py`
- `src/legal_rag/answering/prompt_builder.py`

`src/legal_rag_pipeline.py` se tam thoi chi:

- import facade
- expose `LegalAnswer`, `LegalSearchResult`, `LegalRAGPipeline`, `get_legal_rag_pipeline()`

## RED

Them/cap nhat tests cho:

1. `LegalRAGPipeline.search()` van tra ve object contract cu
2. `LegalRAGPipeline.query()` van dung trace fields
3. context builder tao citation/context dung format da khoa
4. prompt builder giu rule `QUERY_CONTEXT` vs `EVIDENCE_CONTEXT`

Recommended tests:

- `test_search_returns_contract_after_orchestrator_split`
- `test_query_uses_evidence_context_not_query_context_as_ground_truth`
- `test_context_builder_returns_citations_and_context_text`

## GREEN

Implementation steps:

1. move DTO sang module contracts
2. move `_format_context()` sang context builder
3. move `LegalLLM` sang answering gateway
4. move orchestration logic cua `search/query/_retrieve` sang orchestrator
5. giu facade methods o file cu

## REFACTOR

- reduce direct env access trong orchestrator
- inject dependencies ro hon
- xoa duplicate helper giua facade va module moi

## Exit criteria

- `src/legal_rag_pipeline.py` giam manh kich thuoc va role
- tests contract search/query van xanh
- route/service layer khong can biet module split ben trong

## Dependency

- Task 01
- Task 03
