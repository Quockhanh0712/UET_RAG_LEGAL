# Graph Schema Spec (V1)

## Scope
- Default mode: document-level graph for legal impact retrieval.
- Upgrade target: provision-level graph after quality/latency gate passes.

## Node Types
- `Document`: canonical legal document (`doc_id` as unique key).
- `DocType`, `Agency`, `Signer`, `LawFamily`: metadata nodes.
- `Provision` remains available for phase-2 expansion.

## Edge Types
- Core legal-impact edges: `AMENDS`, `REPLACES`, `ABROGATES`, `CITES`.
- Metadata edges: `ISSUED_BY`, `SIGNED_BY`, `BELONGS_TO_FAMILY`, `OF_TYPE`, `TYPE_SUPERIOR_TO`.

## Mandatory Edge Properties (AMENDS/REPLACES/ABROGATES/CITES)
- `evidence_text`
- `evidence_source` (default: `mo_ta`)
- `confidence_rule`
- `snapshot_date`
- `updated_at`

## Expansion Rules
- Graph expansion happens after hybrid shortlist.
- V1 cap: `hops=1`.
- Modes:
  - `local`: includes `CITES + legal-impact edges` and same-type expansion.
  - `legal_impact`: only `AMENDS/REPLACES/ABROGATES`.
  - `temporal`: legal-impact edges with active/as-of filtering in retrieval layer.
