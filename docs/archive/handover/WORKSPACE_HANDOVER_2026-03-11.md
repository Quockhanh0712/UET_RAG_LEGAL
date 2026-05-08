# Workspace Handover Report (2026-03-11)

## 1) Project Snapshot

- Project: Vietnamese Legal Hybrid RAG + Graph-ready pipeline
- Main corpus source: `VietTung04/vbpl-vn` (Hugging Face)
- Current status: data-first pipeline is implemented and producing `gold_primary` + `gold_archive`
- Build date reference used in this workspace: `2026-03-11`

## 2) What Is Implemented

### 2.1 Data pipeline (document-level)

- Loader from Hugging Face in `scripts/load_vbpl_vn_dual_index.py`
- Deterministic normalization in `src/legal_data_processing.py`
  - `doc_id` normalization
  - date parsing (`ngay_ban_hanh`, `ngay_co_hieu_luc`, `ngay_het_hieu_luc`)
  - `is_active`, `status_label`, `expires_in_days`
  - `content_hash` for incremental updates
- Soft filtering and ranking in `src/legal_gold_filter.py`
  - profile-based: `strict`, `balanced`, `broad`
  - soft target for law count with law backfill
  - two outputs: `primary` and `archive`
- Data-only one-click runner: `scripts/run_gold_data_first.ps1`

### 2.2 Retrieval and indexing layer

- Qdrant hybrid storage in `src/qdrant_store.py`
  - dense + sparse (BM25-like) vectors
  - payload indexes for legal metadata
  - filtering by fields (including `is_active`)
- Chunking in `src/legal_chunker_v2.py`
  - intended legal-aware hierarchy: `Chuong -> Dieu -> Khoan -> Diem`
  - fallback hard split if legal boundaries are not detected
- Query pipeline in `src/legal_rag_pipeline.py`
  - hybrid retrieval (dense + sparse)
  - optional graph expansion
  - optional reranker

### 2.3 Graph layer (rule-based, no LLM extraction)

- Relation extraction from `mo_ta` in `src/legal_graph_extractor.py`
  - `AMENDS`, `REPLACES`, `ABROGATES`
  - regex doc-reference linking from text
- Neo4j persistence in `src/neo4j_store.py`
  - nodes: `Document`, `Provision`, `DocType`, `Agency`, `Signer`, `LawFamily`
  - edges include: `HAS_PROVISION`, `ISSUED_BY`, `SIGNED_BY`, `OF_TYPE`, `BELONGS_TO_FAMILY`, `TYPE_SUPERIOR_TO`, and amendment edges

### 2.4 Mining/reporting

- Dataset report generator: `scripts/report_gold_dataset.py`
- Produced reports in `data/gold/reports/`

## 3) Current Data Artifacts (already generated)

- Primary dataset: `data/gold/vbpl_gold.parquet`
- Archive dataset: `data/gold/vbpl_gold_archive.parquet`
- Web source seeds: `data/gold/web_sources_seed.json`
- Reports folder: `data/gold/reports/`

## 4) Current Metrics (from generated reports + runtime checks)

### 4.1 Gold set size and quality

- `gold_primary`: `1255` docs
- `gold_archive`: `15165` docs
- Quality score: `92.0` (grade `A`)
- Law count in primary (`LUAT + PHAP_LENH`): `290`
- Law ratio in primary: `23.11%`
- Recent ratio (issue year >= 2015): `44.46%` (warn in current policy)
- Active ratio in primary: `100%`
- Duplicate `doc_id`: `0`
- Duplicate `content_hash`: `0`

### 4.2 Primary doc-type distribution

- `THONG_TU`: `520` (41.43%)
- `NGHI_DINH`: `445` (35.46%)
- `LUAT`: `262` (20.88%)
- `PHAP_LENH`: `28` (2.23%)

### 4.3 Seed law-family coverage

- Covered in primary: `16/17`
- Missing seed family: `Luat Xuat nhap canh`

### 4.4 Graph extraction readiness (rule-based check on primary)

- Total extracted legal-impact relations: `107`
- By type:
  - `AMENDS`: `103`
  - `ABROGATES`: `4`
  - `REPLACES`: `0`
- Source docs with at least one relation: `84`
- Unique target docs referenced: `102`

## 5) Important Technical Findings (critical for next workspace)

### 5.1 Chunking quality risk (high priority)

- Current legal-aware parser depends on line-anchored markers (`^Dieu ...`).
- In current normalized text, legal markers are often inline (not at line start).
- Result on `gold_primary`:
  - Docs with line-anchored `Dieu`: `2/1255`
  - Docs with `Dieu` anywhere in text: `1212/1255`
  - Produced chunks: `21909` total
  - Structured chunks with detected `dieu_so`: only `10`
- Practical impact:
  - Most docs are chunked by fallback hard-split, not by legal units.
  - Citation precision and graph/provision linkage quality are reduced.

### 5.2 Metadata normalization risk (medium)

- `law_family` still contains noisy long strings in multiple records.
- Some records include OCR-like artifacts in `ten_van_ban`/`law_family`.
- This affects seed matching quality and family-level graph grouping.

### 5.3 Infra runtime status (local machine at check time)

- Qdrant client initialized but server unreachable:
  - error: `[WinError 10061] No connection could be made because the target machine actively refused it`
- Neo4j graph store is disabled (missing credentials and/or service not running).

## 6) Runbook (re-run in new workspace)

### 6.1 Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r web_app/backend/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 6.2 Build gold datasets only (recommended first step)

```powershell
python scripts/load_vbpl_vn_dual_index.py `
  --dataset VietTung04/vbpl-vn `
  --data-only `
  --profile balanced `
  --gold-output data/gold/vbpl_gold.parquet
```

### 6.3 Generate mining report

```powershell
python scripts/report_gold_dataset.py `
  --primary-path data/gold/vbpl_gold.parquet `
  --archive-path data/gold/vbpl_gold_archive.parquet `
  --output-dir data/gold/reports
```

### 6.4 Optional full indexing (after infra is up)

```powershell
python scripts/load_vbpl_vn_dual_index.py `
  --dataset VietTung04/vbpl-vn `
  --gold-output data/gold/vbpl_gold.parquet `
  --qdrant-collection gold_legal `
  --force-recreate
```

## 7) Files To Copy/Keep For Migration

- Core pipeline:
  - `scripts/load_vbpl_vn_dual_index.py`
  - `scripts/run_gold_data_first.ps1`
  - `scripts/report_gold_dataset.py`
  - `src/legal_data_processing.py`
  - `src/legal_gold_filter.py`
  - `src/legal_chunker_v2.py`
  - `src/legal_graph_extractor.py`
  - `src/qdrant_store.py`
  - `src/neo4j_store.py`
  - `src/legal_rag_pipeline.py`
- Data artifacts:
  - `data/gold/vbpl_gold.parquet`
  - `data/gold/vbpl_gold_archive.parquet`
  - `data/gold/web_sources_seed.json`
  - `data/gold/reports/*`

## 8) Recommended Next Actions (priority order)

1. Fix legal-structure preprocessing before chunking:
   - normalize inline markers into reliable boundaries for `Chuong/Dieu/Khoan/Diem`
   - keep deterministic rules (no LLM)
2. Rebuild chunk set and validate:
   - `%chunks with provision_id + citation_full`
   - `%chunks parsed by legal structure vs fallback`
3. Improve `law_family` cleanup:
   - reduce noisy long family strings and OCR artifacts
   - canonical family mapping for graph joins
4. Bring up infra and smoke test:
   - start Qdrant + Neo4j
   - run dual indexing on a 200-doc sample first
5. Start retrieval benchmark matrix:
   - baseline hybrid vs graph-expanded variants
   - track recall/citation validity/latency gates

## 9) Operational Notes

- Pipeline is deterministic and currently avoids LLM-based extraction for filtering/chunking/graph relation extraction.
- `primary` is intended as default retrieval corpus; `archive` is long-tail/history.
- `--abort-on-low-laws` is optional; default behavior is warning + continue.

