# Chunking Spec (V1)

## Objective
- Improve citation precision by deterministic legal structure parsing.
- Keep fallback deterministic and constrained to avoid cross-article leakage.

## Preprocessing
- Stabilize inline markers into line boundaries before regex parsing:
  - `Chuong/Chương`
  - `Dieu/Điều`
  - `Khoan/Khoản`
  - `Diem/Điểm`

## Chunk Types
- `ProvisionChunk`: article-level (`Dieu`).
- `ClauseChunk`: clause/point-level (`Khoan/Diem`).
- `FallbackChunk`: parser fallback when structure is missing or hard split is required.

## Mandatory Chunk Metadata
- `chunk_type`
- `legal_path`
- `segment_level`
- `fallback_reason`
- `token_len_est`
- `encoder_profile`
- `provision_id`
- `citation_full`
- `is_active`
- `status_label`
- `law_family`

## Invariants
- Do not split across `Dieu`.
- If long content requires split, split only within same provision.
- `citation_full` must always be populated.
