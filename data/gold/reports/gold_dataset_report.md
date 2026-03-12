# Gold Dataset Mining Report

- Generated at: `2026-03-11 21:49:56`
- Primary rows: `1255`
- Archive rows: `15165`

## Quality Summary

- Quality grade: **A**
- Quality score: **92.0/100**
- Law count (primary): `290`
- Law ratio (primary): `23.11%`
- Recent ratio (primary): `44.46%`
- Active ratio (primary): `100.00%`

## Seed Coverage

- Covered seed families in primary: `16/17`
- Uncovered seed families: `1`
- Uncovered preview: Luat Xuat nhap canh

## Doc Type Distribution

### Primary (top 10)

| doc_type   |   count |     ratio |
|:-----------|--------:|----------:|
| THONG_TU   |     520 | 0.414343  |
| NGHI_DINH  |     445 | 0.354582  |
| LUAT       |     262 | 0.208765  |
| PHAP_LENH  |      28 | 0.0223108 |

### Archive (top 10)

| doc_type   |   count |      ratio |
|:-----------|--------:|-----------:|
| THONG_TU   |   11351 | 0.7485     |
| NGHI_DINH  |    3543 | 0.23363    |
| PHAP_LENH  |     137 | 0.00903396 |
| LUAT       |     134 | 0.00883614 |

## Text Length

- Primary chars p50/p90: `25528/98302`
- Archive chars p50/p90: `10893/40968`

## Quality Checks

| check_name               | value                   | target                   | status   |
|:-------------------------|:------------------------|:-------------------------|:---------|
| law_count_or_ratio       | count=290, ratio=0.2311 | count>=40 OR ratio>=0.25 | PASS     |
| recent_ratio             | 0.4446                  | >=0.55 (year>=2015)      | WARN     |
| active_ratio             | 1.0000                  | >=0.95                   | PASS     |
| duplicate_doc_id         | 0                       | ==0                      | PASS     |
| duplicate_content_hash   | 0                       | ==0                      | PASS     |
| missing_issue_date_ratio | 0.0000                  | <=0.05                   | PASS     |

## Notes

- `gold_primary` is retrieval default corpus.
- `gold_archive` is long-tail corpus for history/graph expansion.