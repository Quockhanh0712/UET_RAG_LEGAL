### Title
Research sâu cho GraphRAG + Chunking Hybrid Search trên corpus pháp luật đã lọc mềm (`VietTung04/vbpl-vn`)

### Summary
- Mục tiêu: thiết kế kiến trúc GraphRAG và chunking cho legal retrieval theo hướng tăng recall nhưng vẫn kiểm soát precision/citation.
- Dữ liệu đầu vào cố định tại thời điểm `2026-03-11`: `gold_primary=1255`, `gold_archive=15165`, `law_count(primary)=290`, `law_ratio=23.11%`, `recent_ratio=44.46%`.
- Cách làm: survey có trọng số bằng chứng (peer-reviewed > shared-task papers > preprint docs), sau đó chốt blueprint kỹ thuật + ma trận thực nghiệm + tiêu chí go/no-go.

### Key Changes / Research Workstreams
- **Workstream 1: Evidence Survey có kiểm chứng nguồn**
- Áp dụng workflow từ skill `ml-paper-writing` (citation verification) + `brainstorming-research-ideas` (framing theo tension recall/precision).
- Tạo `evidence matrix` với cột cố định: `paper_id`, `venue`, `task`, `dataset`, `retrieval_stack`, `graph_role`, `chunking_role`, `metrics`, `key_finding`, `risk`.
- Phân hạng nguồn:
- Tier A (bắt buộc): GraphRAG gốc, COLIEE task docs/proceedings, ACL Anthology VLSP 2025 DRiLL, EACL 2023 SAR graph retriever.
- Tier B (tham khảo): preprints legal GraphRAG/temporal legal KG.
- Tier C (implementation docs): Microsoft GraphRAG dataflow/inputs.
- Kết quả survey phải trả lời dứt điểm 5 câu hỏi:
- Khi nào graph giúp hơn hybrid thuần?
- Quan hệ pháp lý nào có lợi nhất cho retrieval?
- Chunk granularity nào tối ưu cho luật (Điều/Khoản)?
- Multi-stage (retrieve→rerank→filter) cải thiện ở đâu?
- Cơ chế temporal/versioning nào bắt buộc trong miền luật?

- **Workstream 2: Blueprint GraphRAG cho miền luật (deterministic, không LLM extraction)**
- Đồ thị 2 lớp:
- `Structure graph`: Document → Chuong → Dieu → Khoan → Diem.
- `Legal-relation graph`: `AMENDS`, `REPLACES`, `ABROGATES`, `CITES`, `ISSUED_BY`, `SIGNED_BY`, `BELONGS_TO_FAMILY`, `TYPE_SUPERIOR_TO`.
- Temporal model:
- `effective_from`, `effective_to`, `status_label`, `snapshot_date`.
- Mỗi edge quan hệ sửa đổi/bãi bỏ có `evidence_text`, `evidence_source`, `confidence_rule`.
- Truy vấn graph theo 3 mode:
- `local` (1-hop theo Điều/Khoản),
- `legal-impact` (quan hệ sửa đổi/thay thế/bãi bỏ),
- `temporal` (as-of date).
- Graph chỉ expand sau khi có shortlist hybrid để tránh bùng nổ chi phí.

- **Workstream 3: Chunking strategy cho Hybrid Search**
- Chuẩn chunk 3 loại:
- `ProvisionChunk` (Điều-level, ưu tiên dense): giữ tiêu đề Điều + ngữ cảnh Chương.
- `ClauseChunk` (Khoản/Điểm-level, ưu tiên sparse): giữ path đầy đủ `Dieu/Khoan/Diem`.
- `ContextHeader` prepend: `citation_base`, `doc_type`, `law_family`, `effective status`.
- Quy tắc cứng:
- Không cắt ngang Điều.
- Nếu Điều dài: tách Khoản rồi Điểm; fallback hard-split chỉ trong cùng Điều.
- Mọi chunk phải có `citation_full` và `provision_id`.
- Indexing policy:
- Dense index: ưu tiên `ProvisionChunk`.
- Sparse BM25 index: cả `ProvisionChunk` + `ClauseChunk`.
- Metadata filter mặc định: `is_active=true`, cho phép mở `include_archive`.
- Rerank stage: cross-encoder trên top-100 hybrid + top-graph-expanded.

- **Workstream 4: Experiment matrix (bắt buộc có ablation)**
- Baseline:
- `B0`: BM25 only.
- `B1`: Dense only.
- `B2`: Hybrid BM25 + Dense.
- `B3`: Hybrid + Reranker.
- Graph variants:
- `G1`: B3 + graph local expansion.
- `G2`: B3 + legal-impact expansion.
- `G3`: B3 + temporal filtering.
- Chunking ablation:
- `C1`: fixed-size chunk.
- `C2`: Điều-only.
- `C3`: Điều + Khoản/Điểm (đề xuất chính).
- `C4`: C3 + metadata prepend.
- Metrics:
- Retrieval: `Recall@20`, `nDCG@10`, `MRR@10`, `Macro-F2` (theo DRiLL style).
- Citation quality: `%answer có citation hợp lệ`, `%citation đúng as-of-date`.
- Graph utility: `%query cần legal-impact được cải thiện Recall@20`.
- Acceptance gate:
- So với `B3`, `G2+C3` phải đạt `+5% Recall@20` hoặc `+3% nDCG@10`.
- Citation validity >= `95%`.
- Latency p95 tăng không quá `35%`.

### Public Interfaces / Deliverables
- Thêm bộ artifact nghiên cứu (không thay ingestion logic hiện tại):
- `data/gold/research/evidence_matrix.csv`
- `data/gold/research/graph_schema_spec.md`
- `data/gold/research/chunking_spec.md`
- `data/gold/research/experiment_plan.yaml`
- `data/gold/research/research_survey_report.md`
- Chuẩn output report:
- 1 trang executive summary (khuyến nghị chọn config chính thức).
- 1 bảng quyết định cuối cùng: `recommended_profile`, `recommended_graph_mode`, `recommended_chunking_variant`.

### Test Plan
- Dataset validation:
- Không duplicate `doc_id`/`content_hash`.
- Tỷ lệ `LAW+PHAP_LENH` và `recent_ratio` được report riêng cho `primary` và `archive`.
- Chunking validation:
- 100% chunk có `provision_id`, `citation_full`, không cross-Điều.
- Graph validation:
- Precision rule-based cho `AMENDS/REPLACES/ABROGATES` trên tập gán nhãn tay (>=200 mẫu) phải >= `0.85`.
- Retrieval benchmark:
- Bộ 30 query chuẩn (luật phổ biến + câu hỏi sửa đổi/bãi bỏ + câu hỏi temporal).
- So sánh toàn bộ B0/B1/B2/B3/G1/G2/G3 và C1/C2/C3/C4.

### Assumptions (locked)
- Nguồn chính vẫn là `VietTung04/vbpl-vn`; web sources chỉ verify/fill metadata, chưa auto-merge vào `primary`.
- Không dùng LLM cho extraction graph/chunking; chỉ rule/regex/deterministic parsing.
- Mặc định profile nghiên cứu là `balanced`; chỉ promote sang production khi pass acceptance gate.
- Các nguồn nền tảng dùng làm evidence chính:
- GraphRAG: https://arxiv.org/abs/2404.16130
- GraphRAG docs: https://microsoft.github.io/graphrag//index/default_dataflow/
- GraphRAG inputs: https://microsoft.github.io/graphrag/index/inputs/
- COLIEE Task 3: https://coliee.org/tasks/task3
- COLIEE 2023 overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC11026282/
- BM25 legal baseline: https://arxiv.org/abs/2105.05686
- Statute retrieval + graph structure: https://aclanthology.org/2023.eacl-main.203/
- Vietnamese legal hybrid retrieval (DRiLL): https://aclanthology.org/2025.vlsp-1.17.pdf
- Vietnamese two-stage retrieval+rERank (DRiLL): https://aclanthology.org/2025.vlsp-1.20.pdf
- Structure-aware legal chunking: https://aclanthology.org/2025.justnlp-main.19/
