# Qdrant Data and Chunking Guide

## Mục tiêu
- Mô tả chính xác dữ liệu legal hiện đã được index lên Qdrant cloud mới.
- Giải thích vì sao hệ thống tạo ra số lượng chunk hiện tại.
- Chốt quy trình tái tạo lại toàn bộ chunks và upload mà không phải đọc ngược nhiều file code.

## 1. Snapshot dữ liệu hiện đã lên Qdrant

### Nguồn dữ liệu
- Dataset đầu vào: `data/gold/vbpl_gold.parquet`
- Scope đã dùng cho lần upload hiện tại:
  - `source_set=primary`
  - `loaded_docs=1255`
- Embedding profile dùng để index:
  - model: `truro7/vn-law-embedding`
  - normalize: `true`
  - device: `cuda`
  - fp16: `true`

### Collection hiện tại
- Qdrant collection: `legal_documents__truro7_vn_law_embedding__d768`
- Dense vector dim: `768`
- Sparse vector: BM25 encoder build từ chính tập chunk đã chọn để index
- Payload profile: `lean`
- Lần upload full gần nhất ghi nhận:
  - `indexed_chunks=57551`
  - `points_count=57551`
  - collection status lúc kết thúc run: `yellow`
  - optimizer status: `ok`
  - smoke trong lúc upload: `3/3`, `hits=15`

### Ý nghĩa của `57551` points
- Mỗi point trên Qdrant tương ứng với `1 retrieval chunk`.
- Số này thấp hơn collection cũ `155,938` vì pipeline hiện tại đã đổi sang:
  - chunk theo `Điều-first`
  - chỉ index `ProvisionChunk`
  - bỏ clause-level retrieval ra khỏi production path

## 2. Dữ liệu mỗi point trên Qdrant gồm gì

### Vectors
- `dense`: embedding của `truro7/vn-law-embedding`
- `sparse`: BM25-style sparse vector từ `content`

### Payload chính
- `chunk_id`
- `content`
- `doc_id`
- `so_ky_hieu`
- `van_ban_id`
- `ten_van_ban`
- `loai_van_ban`
- `doc_type`
- `doc_type_rank`
- `co_quan_ban_hanh`
- `co_quan`
- `chuc_danh`
- `nguoi_ky`
- `law_family`
- `ngay_ban_hanh`
- `ngay_co_hieu_luc`
- `ngay_het_hieu_luc`
- `is_active`
- `status_label`
- `chuong`
- `ten_chuong`
- `dieu_so`
- `tieu_de_dieu`
- `legal_path`
- `chunk_type`
- `segment_level`
- `fallback_reason`
- `provision_id`
- `citation_base`
- `citation_full`
- `token_len_est`
- `token_len_model`
- `overflow_index`
- `overflow_total`
- `window_char_start`
- `window_char_end`
- `clause_spans`
- `encoder_profile`

### Ví dụ payload đã clean
```json
{
  "chunk_id": "100/2015/QH13_D77_C81_472fd2f2",
  "doc_id": "100/2015/QH13",
  "ten_van_ban": "BỘ LUẬT HÌNH SỰ",
  "law_family": "BỘ LUẬT HÌNH SỰ",
  "dieu_so": "77",
  "tieu_de_dieu": "Phạt tiền",
  "legal_path": "Dieu 77",
  "chunk_type": "ProvisionChunk",
  "segment_level": "article",
  "provision_id": "100/2015/QH13_D77",
  "citation_full": "Bộ luật 100/2015/QH13 - BỘ LUẬT HÌNH SỰ | Dieu 77",
  "clause_spans": ["Khoan 1-2"],
  "encoder_profile": "truro7_vn_law_embedding|norm=true"
}
```

## 3. Chunking hiện tại tạo chunks như thế nào

### Nguyên tắc chính
- Primary boundary là `Điều`.
- Nếu một `Điều` còn nằm trong budget tokenizer của embedding model, tạo đúng `1 chunk`.
- Nếu một `Điều` vượt budget, chỉ tách bên trong chính `Điều` đó thành các `article_window`.
- `Khoản` và `Điểm` không còn là retrieval boundary; chúng chỉ còn giữ ở `clause_spans` để trace citation.

### Budget hiện tại
- `encoder_max_len = 512`
- `target_chunk_tokens = 420`
- `hard_cap_tokens = 448`
- `overlap_tokens = 64`

### Các bước tạo chunk
1. Chuẩn hóa row văn bản bằng `normalize_document_row()`
   - clean `doc_id`
   - clean `ten_van_ban`
   - sinh `law_family`
   - sinh `citation_base`
2. Chuẩn hóa legal text
   - bỏ khoảng trắng thừa
   - thống nhất line breaks
   - bảo vệ numeric dot trong các số kiểu `100.000`, `1.500.000.000`
3. Ổn định marker cấu trúc
   - đưa `Chương`
   - `Điều`
   - marker inline
   về line boundary để regex parse ổn định
4. Parse `Điều`
   - dùng `DIEU_PATTERN`
   - tách `dieu_so`, `tieu_de_dieu`, `article_body`
5. Tính chapter context
   - map `Điều` vào `chuong`, `ten_chuong`
6. Tách overflow nếu cần
   - tokenize bằng tokenizer thật của model embedding
   - nếu `Điều` dài, split thành `article_window`
   - mỗi window giữ lại header `Dieu X. <title>`
   - giữa các window có `overlap_tokens=64`
7. Trích `clause_spans`
   - detect các `Khoản` hợp lệ trong body
   - nén thành range như `Khoan 1-2`, `Khoan 1-4`
8. Sinh metadata chunk
   - `chunk_id`
   - `provision_id`
   - `citation_full`
   - `segment_level`
   - `overflow_index/overflow_total`
   - `token_len_model`
   - `encoder_profile`

## 4. Vì sao không còn `ClauseChunk`

### Lý do đổi policy
- Clause-level chunking trước đây làm mất ngữ cảnh và tạo lỗi parse kiểu `Khoan 000`.
- Các số tiền như `100.000` hoặc `1.500.000.000` dễ bị regex hiểu sai là clause marker.
- Retrieval theo `Điều` giúp giữ đủ ngữ cảnh pháp lý và ổn định hơn cho citation.

### Chính sách production hiện tại
- `chunk_policy=c3`
- Với article-first chunking, `C3` chỉ giữ:
  - `ProvisionChunk`
- Nếu document parse kém và không có `Điều`, hệ thống vẫn có thể sinh fallback document chunks để không mất dữ liệu hoàn toàn.

## 5. Tại sao chất lượng upload hiện là như vậy

### Metrics từ lần upload full hiện tại
- `provision_id_valid_rate=100.00%`
- `structured_chunk_rate=96.16%`
- `citation_full_rate=100.00%`
- `non_cross_dieu_rate=100.00%`

### Diễn giải
- `100% provision_id`: mọi chunk retrieval chính đều có định danh pháp lý hợp lệ.
- `96.16% structured`: phần lớn chunk là structured legal chunks; phần còn lại chủ yếu là fallback document windows cho văn bản parse không sạch hoàn toàn.
- `100% citation_full`: tất cả chunk đều có citation đầy đủ để hiển thị/truy vết.
- `100% non_cross_dieu`: không có chunk nào cắt ngang nhiều `Điều`.

## 6. Qdrant tạo collection và points ra sao

### Collection shape
- Named vectors:
  - `dense`
  - `sparse`
- Distance cho dense:
  - `COSINE`
- `on_disk_payload=true`
- Payload indexes được tạo cho các field filter/search quan trọng:
  - `doc_id`
  - `doc_type`
  - `status_label`
  - `law_family`
  - `legal_path`
  - `is_active`
  - `dieu_so`
  - `token_len_model`
  - `encoder_profile`
  - và các field core khác

### Point identity
- `point_id` trên Qdrant được sinh từ hash của `chunk_id`
- Nghĩa là rerun cùng chunk sẽ overwrite point cũ thay vì nhân đôi điểm

## 7. Quy trình tạo và upload toàn bộ chunks

### Bước 1: load records
- Script: `scripts/upload_qdrant_cloud_embeddings.py`
- Hàm: `_load_records()`
- Load từ parquet, lọc `source_set=primary`, lấy `1255` docs

### Bước 2: build chunks cho từng doc batch
- Hàm: `_build_chunks_for_batch()`
- Mỗi batch:
  - khởi tạo `LegalChunkerV2`
  - gọi `chunk_document()` cho từng văn bản
  - áp `chunk_policy=c3`

### Bước 3: fit BM25 toàn cục
- Phase 1 của script:
  - stream qua các doc batches
  - lấy toàn bộ `content` của chunks đã chọn
  - `partial_fit()` vào `BM25Encoder`
  - sau khi xong mới `save()` encoder

### Bước 4: embed + upsert
- Phase 2 của script:
  - gắn `encoder_profile`
  - gọi `embed_texts()` trên GPU
  - build dense+sparse vectors
  - build lean payload
  - `add_documents()` vào Qdrant theo `upsert_batch_size`

### Bước 5: smoke trong lúc upload
- Sau upload, script chạy một smoke hybrid nhỏ
- Mục tiêu:
  - xác nhận collection query được
  - xác nhận dense+sparse pipeline không lỗi

## 8. Command tái tạo đầy đủ

```powershell
.venv\Scripts\python.exe scripts/upload_qdrant_cloud_embeddings.py `
  --input-path data/gold/vbpl_gold.parquet `
  --source-set primary `
  --expected-docs 1255 `
  --doc-batch-size 20 `
  --chunk-policy c3 `
  --models "truro7/vn-law-embedding" `
  --embedding-normalize true `
  --qdrant-url "<QDRANT_CLOUD_URL>" `
  --qdrant-api-key "<QDRANT_API_KEY>" `
  --collection-base legal_documents `
  --payload-profile lean `
  --embedding-batch-size 32 `
  --upsert-batch-size 250 `
  --force-recreate
```

### Điều kiện trước khi chạy
- `.venv` có `torch`, `sentence-transformers`, `qdrant-client`, `pyvi`
- GPU khả dụng nếu muốn encode nhanh
- Qdrant cluster còn đủ dung lượng
- Không để nhầm collection giữa model khác dimension/profile khác

## 9. File code liên quan
- Chunking:
  - `src/legal_chunker_v2.py`
- Upload/index:
  - `scripts/upload_qdrant_cloud_embeddings.py`
- Qdrant payload + collection:
  - `src/qdrant_store.py`
- Title/citation normalization:
  - `src/legal_data_processing.py`

## 10. Caveats hiện tại
- Collection sau upload full được ghi nhận là `yellow`, chưa phải `green`.
- `structured_chunk_rate` chưa đạt tuyệt đối vì vẫn còn một phần fallback chunks cho các văn bản khó parse.
- Collection hiện tại là article-first retrieval index; nếu cần clause-level reasoning thì phải dùng metadata `clause_spans` hoặc phase graph sau này, không quay lại clause retrieval trực tiếp.
