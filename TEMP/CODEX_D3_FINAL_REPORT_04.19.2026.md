# D3 FINAL REPORT

Reproduction command: `bash scripts/ingestion/d3/run.sh --recover-only`

## Scope

- D3 stage root: `/K3D/GitHub/Knowledge3D/scripts/ingestion/staging/D3_dedup`
- Duplicate digest: `/K3D/GitHub/Knowledge3D/TEMP/D3_duplicate_cluster_digest_04.19.2026.jsonl`

## B6.1 + B6 Results

- Alias target rewrites: 30
- Rows touched by alias rewrites: 7
- Reciprocal edges inserted: 24488
- Merged rows after recovery: 277716
- Duplicate digest entries: 100

## Post-B6 Gates

- `duplicate_row_count`: 51111
- `missing_matryoshka`: 251
- `raw_payload`: 65
- `unidirectional_site_count`: 927
- `missing_target`: 890

## Raw Payload Drift

- D2 `raw_payload` rows: 1995
- B3 `success`: 1648
- B3 `queued`: 333
- B3 processed total: 1981
- Drift (`processed_total - d2_raw_count`): -14

## Violation Counts

- `ad_hoc_id`: 35
- `duplicate_content`: 51111
- `missing_matryoshka`: 251
- `missing_target`: 890
- `raw_payload`: 65
- `unidirectional_ref`: 927

## Artifact Hashes

- `B6_edge_rewrites.jsonl` `76ae1ad2ee995414d24b00a1388615c9989b1d349bf9cd501f8e7b4128566aa4`
- `B6_edge_reconciliation.jsonl` `aa1070705dc465575692a78af75db501d6d6a95ba4f699408a50eb0101763b52`
- `duplicate_cluster_digest.jsonl` `dfe1406b3e5511a0b052aae87180c5d21647c454b94d366a2ad374286ce17fa0`
- `merged_stars.jsonl` `01358754a16a26950b68e3766def9b6e0a524b389d33eed314f4d18a7243317a`
- `re_audit_d3/galaxy_census.jsonl` `50f8b2f18c6c0022db52535ab6370ad8d303f00a8624f299a3647afa2cd42bd1`
- `re_audit_d3/violations.jsonl` `bc6b73cba128f678cbd211d4505ccacd56df87ca21c20b17620fb4d976d2f290`
- `re_audit_d3/RE_AUDIT_REPORT.md` `519d911e5d57faab3606b7c6de9ab7d1ff2cf904a5c79f7ac3df00be49ccf373`
