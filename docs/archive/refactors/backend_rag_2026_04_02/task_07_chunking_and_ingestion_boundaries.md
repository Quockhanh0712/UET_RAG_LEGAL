# Task 07 - Chunking And Ingestion Boundaries

## Muc tieu

Tach `src/legal_chunker_v2.py` va phan ingestion policy khoi online RAG workflow de:

- online retrieval khong phu thuoc implementation detail chunking
- chunking co test/doc rieng cho indexing

## Scope

Trong scope:

- chunk policy
- provision/article window split logic
- metadata assembly lien quan ingestion
- contract giua chunker va qdrant payload

Ngoai scope:

- khong doi collection schema neu khong co migration plan ro
- khong doi retrieval online contract trong cung task

## Proposed split

- `src/legal_rag/chunking/policy.py`
- `src/legal_rag/chunking/provision_chunker.py`
- `src/legal_rag/chunking/metadata_builder.py`

## RED

Them/cap nhat tests cho:

1. article-first boundary rules
2. article_window overflow behavior
3. `citation_full`, `provision_id`, `legal_path` luon duoc tao
4. chunk khong cross-article neu policy khoa nhu hien tai

Recommended tests:

- `test_chunker_keeps_article_boundary_when_possible`
- `test_chunker_generates_article_window_for_overflow`
- `test_chunk_metadata_has_citation_full_and_provision_id`

## GREEN

Implementation steps:

1. move policy constants/rules sang module policy
2. move split logic sang chunker module
3. move metadata builder sang module rieng
4. giu `legal_chunker_v2.py` la facade tam thoi neu can

## REFACTOR

- document online-vs-ingestion boundary
- remove duplicated metadata shaping logic neu ton tai

## Exit criteria

- chunking policy co test ro
- ingestion code va online retrieval khong tron logic voi nhau

## Dependency

- Task 01
