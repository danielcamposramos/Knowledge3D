# CODEX D1 Audit Report — 2026-04-18

Reproduction command: `bash scripts/ingestion/audit/run.sh`

## Scope

- Live storage root: `/K3D/Knowledge3D.local/galaxies`
- Census artifact: `/K3D/GitHub/Knowledge3D/scripts/ingestion/staging/D1_audit/galaxy_census.jsonl`
- Violations artifact: `/K3D/GitHub/Knowledge3D/scripts/ingestion/staging/D1_audit/violations.jsonl`
- Files audited: 22
- Total rows audited: 464334

## Headline Counts

- Canonical IDs: 303952
- Missing IDs: 339
- Ad-hoc IDs: 160043
- Procedural rows: 462339
- Raw/non-procedural rows: 1995
- Matryoshka-present Word/Character rows: 0
- Matryoshka-missing Word/Character rows: 70678
- Duplicate content groups: 249905
- Rows participating in duplicate groups: 367275
- Symlink edges scanned: 18368415
- Unidirectional symlink sites: 18368305

## Violation Counts

- `ad_hoc_id`: 160043
- `duplicate_content`: 367275
- `missing_id`: 339
- `missing_matryoshka`: 70678
- `missing_target`: 18365046
- `raw_payload`: 1995
- `unidirectional_ref`: 18368305

## Galaxy Census

| Galaxy | Entries | Canonical | Missing ID | Ad-hoc ID | Procedural | Raw | Matryoshka Present | Matryoshka Missing | Duplicate Groups | Duplicate Rows | Symlink Edges | Unidirectional |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3DObjects | 1796 | 0 | 0 | 1796 | 1796 | 0 | 0 | 0 | 47 | 584 | 0 | 0 |
| Audio | 3251 | 0 | 0 | 3251 | 3251 | 0 | 0 | 0 | 2 | 2890 | 0 | 0 |
| Book_BiologyAtlas | 16 | 0 | 0 | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 702 | 702 |
| Book_LanguageFoundations | 17 | 0 | 0 | 17 | 17 | 0 | 0 | 0 | 0 | 0 | 807 | 807 |
| Book_MathematicsPrimer | 17 | 0 | 0 | 17 | 17 | 0 | 0 | 0 | 0 | 0 | 723 | 723 |
| Book_PhysicsHandbook | 18 | 0 | 0 | 18 | 18 | 0 | 0 | 0 | 0 | 0 | 828 | 828 |
| Book_ToolManual | 17 | 0 | 0 | 17 | 17 | 0 | 0 | 0 | 0 | 0 | 899 | 899 |
| Character | 2608 | 2152 | 0 | 456 | 2607 | 1 | 0 | 2608 | 805 | 1608 | 0 | 0 |
| Drawing | 1360 | 0 | 204 | 1156 | 1146 | 214 | 0 | 0 | 55 | 467 | 0 | 0 |
| Grammar | 103039 | 7 | 118 | 102914 | 102918 | 121 | 0 | 0 | 386 | 89513 | 468 | 413 |
| Language | 116779 | 116220 | 0 | 559 | 116220 | 559 | 0 | 0 | 116193 | 116220 | 9100084 | 9100084 |
| Math | 37732 | 6 | 15 | 37711 | 37577 | 155 | 0 | 0 | 5119 | 13992 | 28096 | 28041 |
| Meta | 2 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| Number | 1001 | 0 | 0 | 1001 | 1001 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Reality | 10874 | 0 | 2 | 10872 | 10602 | 272 | 0 | 0 | 100 | 1940 | 18500 | 18500 |
| Tool | 77 | 0 | 0 | 77 | 52 | 25 | 0 | 0 | 0 | 0 | 100 | 100 |
| Word | 68070 | 68069 | 0 | 1 | 67422 | 648 | 0 | 68070 | 11003 | 23837 | 120 | 120 |
| game_mechanics | 125 | 0 | 0 | 125 | 125 | 0 | 0 | 0 | 0 | 0 | 996 | 996 |
| meaning_layer_stars | 117497 | 117497 | 0 | 0 | 117497 | 0 | 0 | 0 | 116193 | 116220 | 9214388 | 9214388 |
| proceduralized_gsm8k_train_10 | 10 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | 1 | 1 | 825 | 825 |
| proceduralized_mmlu_val_10 | 10 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | 1 | 3 | 732 | 732 |
| reasoning_strategies | 18 | 1 | 0 | 17 | 18 | 0 | 0 | 0 | 0 | 0 | 143 | 143 |

## Highest Unidirectional Counts

- `meaning_layer_stars`: 9214388 unidirectional sites
- `Language`: 9100084 unidirectional sites
- `Math`: 28041 unidirectional sites
- `Reality`: 18500 unidirectional sites
- `game_mechanics`: 996 unidirectional sites

## Highest Ad-hoc ID Counts

- `Grammar`: 102914 ad-hoc ids
- `Math`: 37711 ad-hoc ids
- `Reality`: 10872 ad-hoc ids
- `Audio`: 3251 ad-hoc ids
- `3DObjects`: 1796 ad-hoc ids

## Evidence Commands

- Rebuild artifacts: `bash scripts/ingestion/audit/run.sh`
- Confirm raw source line counts: `wc -l /K3D/Knowledge3D.local/galaxies/*.jsonl`
- Inspect staged violation counts: `python3 - <<'PY'\nimport json\nfrom collections import Counter\nfrom pathlib import Path\npath = Path('/K3D/GitHub/Knowledge3D/scripts/ingestion/staging/D1_audit/violations.jsonl')\ncounts = Counter(json.loads(line)['violation_kind'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip())\nprint(dict(sorted(counts.items())))\nPY`

## Notes

- D1 is audit-only. No live JSONL row was rewritten.
- Counts come from the live resident root, not the older 38,144-entry claim in the handoff text.
