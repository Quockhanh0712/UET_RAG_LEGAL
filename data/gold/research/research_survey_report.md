# Research Survey Report (v2)

Generated: `2026-03-11 23:10:07`

## Skills Applied
- `brainstorming-research-ideas`: tension hunting, adjacent possible, two-sentence test.
- `creative-thinking-for-research`: constraint manipulation, inversion/negation, reformulation.

## Hypotheses (12)

| ID | Framework | Hypothesis |
|---|---|---|
| H1 | Tension hunting | Citation precision and recall can both improve if graph expansion is gated by intent. |
| H2 | Tension hunting | Latency increase from graph expansion can be bounded by shortlist-first design. |
| H3 | Tension hunting | Deterministic parsing can match LLM extraction reliability for legal chunk boundaries. |
| H4 | Constraint manipulation | Forcing no cross-Dieu split increases citation validity. |
| H5 | Constraint manipulation | Restricting graph hops to 1 preserves relevance and controls noise. |
| H6 | Constraint manipulation | Document-level graph first reduces infra risk versus immediate provision-level graph. |
| H7 | Problem reformulation | Reframe retrieval from "find text" to "find legally citable provision path". |
| H8 | Problem reformulation | Treat temporal queries as filtering problem first, generation problem second. |
| H9 | Problem reformulation | Use chunk type as retrieval control signal instead of only text similarity. |
| H10 | Adjacent possible | Existing qdrant hybrid + reranker stack can evaluate B0..G3 without new infra. |
| H11 | Adjacent possible | Existing gold corpus quality is sufficient for v1 benchmark if labels are curated. |
| H12 | Adjacent possible | Edge metadata (`evidence/confidence/snapshot`) enables auditable legal-impact graph. |

## Candidate Options (Top 4)

| Option | Description | Citation Precision Impact | Feasibility | Infra Cost | Decision |
|---|---|---:|---:|---:|---|
| O1 | Hybrid + reranker + intent-gated graph + structure chunking | High | High | Medium | **Primary** |
| O2 | Hybrid + reranker + structure chunking, no graph | Medium-High | Very High | Low | Backup |
| O3 | Dense-only + graph expansion | Medium | Medium | Medium | Rejected |
| O4 | Sparse-only + legal-impact graph | Medium | High | Low-Med | Rejected |

## Two-Sentence Test
- Problem: Vietnamese legal retrieval struggles to maintain both recall and citation validity, especially for amendment and temporal questions where document relationships matter.
- Insight: We improve this by deterministic structure-aware chunking plus intent-gated document-level graph expansion after hybrid shortlist, which works because it constrains noisy expansion while preserving legal-impact signals.

## Risk Table

| Risk | Severity | Mitigation |
|---|---|---|
| Inline marker noise breaks parser | High | Add marker stabilization preprocessing and fallback reason tracking |
| Graph expansion adds irrelevant docs | High | Intent-gated modes + hop=1 + shortlist-first |
| Neo4j unavailable in dev | Medium | Mark graph variants as `graph-pending`, still run B0..B3 |
| Labeling bottleneck for 300 pairs | Medium | Use template with split dev/test and prioritize amendment/temporal first |
| Latency regression after rerank+graph | Medium | Track p95 in benchmark gate and enforce <=35% increase |
