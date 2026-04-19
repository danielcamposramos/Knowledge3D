# D3 FINAL REPORT

Reproduction command: `bash scripts/ingestion/d3/run.sh --recover-only`

## Scope

- D3 stage root: `/K3D/GitHub/Knowledge3D/scripts/ingestion/staging/D3_dedup`
- Duplicate digest: `/K3D/GitHub/Knowledge3D/TEMP/D3_duplicate_cluster_digest_04.19.2026.jsonl`

## B6.1 + B6 Results

- Alias target rewrites: 204
- Rows touched by alias rewrites: 38
- Reciprocal edges inserted: 273
- Merged rows after recovery: 277613
- Duplicate digest entries: 100

## Post-B6 Gates

- `duplicate_row_count`: 51221
- `missing_matryoshka`: 0
- `raw_payload`: 0
- `unidirectional_site_count`: 547
- `missing_target`: 510

## Raw Payload Drift

- D2 `raw_payload` rows: 1995
- B3 `success`: 1648
- B3 `queued`: 333
- B3 processed total: 1981
- Drift (`processed_total - d2_raw_count`): -14

## Violation Counts

- `duplicate_content`: 51221
- `missing_target`: 510
- `unidirectional_ref`: 547

## Artifact Hashes

- `B6_edge_rewrites.jsonl` `022b4128c9332413145700f00208afdbbb954662ab2be274cc242e3cbe564e45`
- `B6_edge_reconciliation.jsonl` `ea81416ae145c8ad55d9289629604d11b0d097ba68f3a7be514e090471ade187`
- `duplicate_cluster_digest.jsonl` `b97ea513fd75fee47d6c58d4b08b8012ac2268513c0efd44af8c7b68a9b7dee0`
- `merged_stars.jsonl` `e80afddd17bd9e8a16a5910708435dde0690d61857e189ddb0e9bf75dd2a5625`
- `re_audit_d3/galaxy_census.jsonl` `3d8063794b4e1fb8c797c695226aae5d13d1157dff32e9edaf413df3eef603d1`
- `re_audit_d3/violations.jsonl` `fcb70c00ac7a420797be76e6eb30932498b968b40e0b78a1f9d9d57c13da0955`
- `re_audit_d3/RE_AUDIT_REPORT.md` `cfdf2fdc54b60d474a533d0058ea8566dd7175e86ae93177aa90e7a754d03d8c`
