# Ke Hoach MD Hoan Chinh: TDD cho Data Hybrid Search + Legal Graph (Standard + Prod Gate)

## 1. Executive Summary
- Muc tieu: tao 1 file master theo doi day du chien luoc, TDD, tien do, du lieu, va tai nguyen cho 2 task:
  - Task A: data pipeline cho `chunking + hybrid search`.
  - Task B: data pipeline cho `legal graph`.
- Profile da khoa:
  - Resource profile: `Standard`.
  - TDD level: `Prod Gate`.
- Data scope co dinh: `gold_primary + gold_archive`.
- Priority thu tu: `citation precision > recall > latency`.
- V1 policy: chi dung deterministic/rule-based cho chunking/graph, khong dung LLM extraction.

## 2. Current Baseline Snapshot (dated 2026-03-12)
### 2.1 Dataset Snapshot
- `gold_primary = 1255`
- `gold_archive = 15165`

### 2.2 Chunking Quality Snapshot
- `provision_id = 100%`
- `structured = 99.82%`
- `citation_full = 100%`
- `non_cross_dieu = 100%`

### 2.3 Graph Audit Snapshot
- `sample_size = 162`
- `judged = 0`
- Trang thai: chua du co so de tinh precision relation core.

### 2.4 Benchmark Snapshot
- Run gan nhat: `run_20260312_033429`
- Trang thai:
  - `B0..B3 = ok`
  - `G* = graph-pending`
  - `C* = pending-chunk-index`
- Cloud collections hien co:
  - `legal_documents__aiteamvn_vietnamese_embedding__d1024`
  - `legal_documents__huyydangg_dek21_hcmute_embedding__d768`

## 3. Task A - Hybrid Search Data Plan + TDD
### 3.1 Muc tieu
- On dinh hoa chunking + metadata contract cho retrieval.
- Chot production profile: `AITeamVN + normalize=true + C3`.
- Dam bao index/query khong tron vector space khac dimension.

### 3.2 RED (viet test fail truoc)
- Unit tests:
  - legal boundary parsing dung marker `Chuong/Dieu/Khoan/Diem`.
  - khong cross-Dieu split.
  - mandatory metadata fill cho moi chunk.
- Integration tests:
  - upload/query Qdrant tren 2 collection `d1024` va `d768`.
  - metadata filters hoat dong: `is_active`, `law_family`, `doc_type`.
  - rerank pipeline smoke pass.
- Benchmark fail condition ban dau:
  - chua dat gate `citation_valid_rate` va metric improvement so voi `B3`.

### 3.3 GREEN (implement toi thieu de pass)
- Enforce chunk invariants:
  - khong cat ngang Dieu.
  - fallback split chi trong cung Dieu.
  - `citation_full` bat buoc 100%.
- Enforce collection naming theo model-dim:
  - `legal_documents__{model_slug}__d{dim}`.
- Enforce default retrieval profile:
  - `embedding_model = AITeamVN/Vietnamese_Embedding`
  - `normalize_embeddings = true`
  - `chunk policy = C3`

### 3.4 REFACTOR (khong doi hanh vi)
- Toi uu encode latency (batch/device profile cho RTX 4050 6GB).
- Toi uu upsert/query batch size tren Qdrant Cloud.
- Toi uu p95 latency retrieval nhung giu nguyen output contract.

### 3.5 Deliverables Task A
- Chunk dataset dung contract metadata.
- Qdrant collections tách theo model-dim.
- Benchmark rows day du contract cho A/B model.

## 4. Task B - Legal Graph Data Plan + TDD
### 4.1 Muc tieu
- Hoan thien graph document-level V1 phuc vu expansion sau hybrid shortlist.
- Dam bao edge evidence day du, dedup edge, va guard hop=1.

### 4.2 RED (viet test fail truoc)
- Unit tests:
  - detect relation `AMENDS/REPLACES/ABROGATES/CITES`.
  - mandatory evidence fields khong rong.
- Integration tests:
  - import Neo4j schema document-level.
  - graph expansion chi chay sau shortlist.
  - guard `hop=1` bat buoc.
  - duplicate edge `(source,target,type)=0`.
- Eval fail condition ban dau:
  - `judged=0` nen precision relation chua do duoc.

### 4.3 GREEN (implement toi thieu de pass)
- Rule-based extractor cho relation core + metadata edges.
- Neo4j import pipeline doc-level.
- Expansion mode:
  - `local`
  - `legal_impact`
  - `temporal`
- Temporal/as-of filter ap dung o retrieval layer.

### 4.4 REFACTOR (khong doi hanh vi)
- Chuan hoa `confidence_rule` naming.
- Chuan hoa dedup va update semantics (`updated_at`).
- Toi uu query pattern graph expansion de kiem soat p95 overhead.

### 4.5 Deliverables Task B
- Graph dataset document-level co edge evidence complete.
- Relation eval summary co precision tren tap judged.
- Expansion pipeline co mode switch + guard hop.

## 5. Shared Data Contracts & Interfaces
### 5.1 Chunk Payload Contract (Task A)
- `chunk_type`
- `legal_path`
- `segment_level`
- `fallback_reason`
- `provision_id`
- `citation_full`
- `is_active`
- `status_label`
- `law_family`
- `token_len_est`
- `encoder_profile`

### 5.2 Graph Edge Contract (Task B)
- `source_doc_id`
- `target_doc_id`
- `relation_type`
- `evidence_text`
- `evidence_source`
- `confidence_rule`
- `snapshot_date`
- `updated_at`

### 5.3 Benchmark Output Contract
- `run_id`
- `variant_id`
- `embedding_model_id`
- `embedding_dim`
- `normalize_embeddings`
- `Recall@20`
- `nDCG@10`
- `MRR@10`
- `citation_valid_rate`
- `latency_p95_ms`
- `encode_p95_ms`
- `status`
- `notes`

### 5.4 Progress Row Contract (trong file MD)
- `date`
- `task`
- `stage`
- `dataset_version`
- `queries_used`
- `key_metrics`
- `decision`
- `risks`
- `owner`
- `eta`

## 6. Resource Plan (People/Infra/Compute/Storage)
### 6.1 People
- `1 ML Engineer`
  - So huu benchmark, embedding profile, retrieval metrics.
- `1 Data Engineer`
  - So huu chunk pipeline, index jobs, Qdrant/Neo4j data flow.
- `0.5 Legal Annotator`
  - So huu labeling relation set 200 mau va citation validity support.

### 6.2 Infra
- Local compute: `RTX 4050 6GB`, CUDA-enabled env.
- Vector DB: `Qdrant Cloud`.
- Graph DB: `Neo4j Aura`.
- Fallback: local/dev mode cho benchmark baseline khi graph chua san sang.

### 6.3 Compute Profile (default)
- `EMBEDDING_DEVICE=cuda`
- `USE_FP16=true`
- `EMBEDDING_BATCH_SIZE=8`
- `normalize_embeddings=true`

### 6.4 Storage & Dataset
- Nguon data duy nhat: `gold_primary + gold_archive`.
- Qdrant collections tách theo `model+dim`.
- Graph snapshot theo ngay de audit (`snapshot_date`).

## 7. Execution Timeline (day-by-day)
### Day 1 (2026-03-12)
- Chot default embedding profile production.
- Verify cloud collections + smoke benchmark A/B.
- Update readiness + progress artifacts.

### Day 2 (2026-03-13)
- RED tests cho Task A + Task B.
- Kiem tra test fail dung expectation.
- Freeze test data slices cho reproducibility.

### Day 3 (2026-03-14)
- GREEN Task A: pass chunk invariants, metadata contract, model-dim routing.
- Chay benchmark `B0..B3` tren 2 models.

### Day 4 (2026-03-15)
- GREEN Task B: relation extractor + graph import doc-level.
- Bat expansion `hop=1` va chay integration tests graph.

### Day 5 (2026-03-16)
- Labeling batch 1 cho relation eval (>=100 mau).
- Chay precision tạm thời va log blockers.

### Day 6 (2026-03-17)
- Labeling batch 2 den du 200 mau.
- Chay full graph quality gate va latency overhead gate.

### Day 7 (2026-03-18)
- REFACTOR Task A/B theo ket qua p95 va precision.
- Chot Go/No-Go report cho rollout cloud profile.

## 8. Risk Register + Mitigation
| risk_id | risk | impact | mitigation | owner | status |
|---|---|---|---|---|---|
| R1 | Neo4j Aura chua san sang | graph variants pending | dung `graph-pending` policy + tiep tuc baseline hybrid | Data Engineer | open |
| R2 | Labeling 200 mau cham | tre precision gate | uu tien relation core, chia 2 batch 100+100 | Legal Annotator | open |
| R3 | Cloud upsert timeout | tre index benchmark | giam batch, tang timeout, retry theo chunk batch | Data Engineer | open |
| R4 | Metric uplift khong dat gate | khong promote config | A/B lai normalize/profile, xem fallback flow | ML Engineer | open |
| R5 | Loang vector space khi tron dim | retrieval suy giam | bat buoc collection tách `model+dim` | ML Engineer | controlled |

## 9. Progress Update Protocol
### 9.1 Nguyen tac cap nhat
- Append-only, khong sua lich su cu.
- Bat buoc cap nhat khi:
  - xong benchmark run,
  - doi decision ky thuat,
  - phat sinh blocker.
- Cuoi moi ngay lam viec phai co dong `next_48h`.

### 9.2 Daily Progress Table
| date | task | stage | dataset_version | queries_used | key_metrics | decision | risks | owner | eta |
|---|---|---|---|---|---|---|---|---|---|
| 2026-03-12 | Hybrid Data | GREEN-smoke | gold_primary+gold_archive@current | 3 | `run_20260312_033429: B0..B3=ok` | keep `AITeamVN` primary, `DEk21` fallback | R3 | ML Engineer | 2026-03-13 |
| 2026-03-12 | Legal Graph | RED-prep | gold_primary+gold_archive@current | 0 | `sample_size=162, judged=0` | can not claim precision yet | R1,R2 | Data Engineer | 2026-03-15 |

### 9.3 Progress Tracking Fields (bat buoc)
- `date`, `owner`, `task_id`, `run_id`, `data_scope`, `metrics_delta`, `blocker`, `next_48h`.

## 10. Acceptance Gates (Go/No-Go)
### 10.1 Task A Gates (Hybrid Data)
- Unit gates:
  - legal boundary parsing pass.
  - `non_cross_dieu = 100%`.
  - mandatory metadata fields filled.
- Integration gates:
  - query pass tren ca 2 collection `d1024` va `d768`.
  - metadata filters + rerank flow pass.
- Benchmark gate:
  - so voi `B3`, dat `+5% Recall@20` hoac `+3% nDCG@10`.
  - `citation_valid_rate >= 95%`.

### 10.2 Task B Gates (Legal Graph)
- Unit gates:
  - detect `AMENDS/REPLACES/ABROGATES/CITES` dung.
  - evidence fields non-empty.
- Integration gates:
  - import doc-level pass.
  - expansion post-shortlist, `hop=1`.
  - duplicate edge `(source,target,type)=0`.
- Quality gate:
  - precision relation core tren 200 mau judged `>= 0.85`.
  - expansion latency overhead `<= 35%`.

### 10.3 Cross-task Smoke Gates
- GPU encode pass cho `AITeamVN` va `DEk21`.
- Qdrant `get_collections()` thay 2 collection tach dimension.
- Neu Neo4j chua san: bat buoc ghi `graph-pending` trong progress/decision log.

## Assumptions & Defaults (Locked)
- Data scope co dinh: `gold_primary + gold_archive`.
- Khong dung LLM extraction cho chunking/graph o V1.
- Priority: `citation precision > recall > latency`.
- Resource profile: `Standard`.
- Cap nhat tien do theo ngay lam viec, khong bo trong ngay co thay doi ky thuat.
