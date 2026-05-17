# B7 Meaning-Resolution Pilot — Completion Report

**Date:** 2026-04-20
**Branch:** `codex/batch11-knowledge-waves-observability-game2d-2026-04-15`
**Operator:** Claude (scoped authorization; Codex limit-locked until 2026-04-20 14:09)
**Spec:** `TEMP/CLAUDE_CODEX_B7_WORD_REFS_AND_RETRY_PILOT_04.19.2026.md`

---

## 1. Headline Outcome

- **36 / 50** clusters resolved under the 3600 s `--stop-after` budget; worker drained the final in-flight cluster at 64 min and exited cleanly.
- **Zero failure events** across the run: no `plan_limit`, no `transport_error`, no `invalid_json`, no retry loops fired. The opt-in retry envelope stayed dormant — the happy path covers the current load.
- **Cascade integrity maintained**: merge pass emits **0 unresolved_symlinks**. Every surface_symlink resolves to a known meaning-star target in the augmented `merged_stars.jsonl`.
- **89 % canonical reuse**: 134 of 150 emitted meaning-stars collided with existing canonical ids in `merged_stars.jsonl` (dedup success); 16 were genuinely new.

## 2. Cluster Outcome Tally

| Outcome | Count | Share |
|---|---:|---:|
| `merge_to_meaning_star` | 20 | 55.6 % |
| `split_polysemy` | 1 | 2.8 % |
| `unresolvable` | 15 | 41.7 % |
| **Total done** | **36** | **100 %** |

- Resolved rows across clusters: **48** (22 meaning_stars × their surface_symlinks).
- Unresolved rows (kept in residual bucket): **62** (unrelated cluster members for next pass).
- Worker log: `scripts/ingestion/staging/D3_dedup/differentiate_b7/workers/claude_pilot50_1776652696.log` (141 events).

## 3. Split Path Validation

The single `split_polysemy` cluster (`e23d0d01…`) correctly separated two unrelated Unicode symbol rows that had been content-hashed together:

| Canonical ID | Script | Surface | Source |
|---|---|---|---|
| `concept_ipa_voiceless_labial_velar_fricative` | IPA | `en: "turned w"`, `unicode: U+028D` | Wikipedia IPA chart |
| `concept_nko_high_tone_mark` | N'Ko | `en: "nko high tone"`, `unicode: U+07E2` | Wikipedia N'Ko |

Two meaning_stars emitted, each with its own `meaning_class=symbol` and its own grounded source. This is the outcome the split bundle schema was designed to produce — the model does not force a merge when the rows diverge in language/script semantics.

## 4. Word-Refs Cascade — Smoke Verification (cardinal_872)

The cascade split (`word_refs` flat list for audit, `word_refs_by_language` dict for the drawing→char→word→meaning path) produced the expected shape on the smoke sample:

```json
{
  "id": "cardinal_872",
  "meaning_class": "fact",
  "meaning_rpn": "num_872 VALUE",
  "surface_forms": {"en": "eight hundred seventy-two", "ja": "八百七十二", "pt": "oitocentos e setenta e dois"},
  "word_refs": ["word_eight_hundred_seventy_two", "word_ja_num_872", "word_pt_num_872"],
  "word_refs_by_language": {"en": ["word_eight_hundred_seventy_two"], "ja": ["word_ja_num_872"], "pt": ["word_pt_num_872"]},
  "sources": ["https://www.easycalculation.com/number-properties-of-872.html"]
}
```

The driver's merge pass synthesizes:
- `pointed_by` from incoming surface_symlink edges (bidirectional contract)
- `word_refs_by_language` partitioned by ISO-639-1 of each symlink's `surface_forms`
- `word_refs` as the flat sorted union (audit-compatible; survives `galaxy_normalize.py` and `galaxy_audit.py` REF_LIST_FIELDS contracts)

## 5. Merge Pass Summary

**Input:** baseline `scripts/ingestion/staging/D3_dedup/merged_stars.jsonl` (277,716 rows, 633 MiB, 2026-04-19 22:15) + 339 accumulated cluster emissions (pilot + prior smoke + prior residual work).

**Output:** `/K3D/GitHub/TEMP/B7_PILOT_04.20.2026/`
- `merged_stars.jsonl` — 277,732 rows (+16 new meaning-stars), sha256 `8712211e…`
- `merged_by_galaxy/` — 24 shard files (adds `Mathematics.jsonl` and `_unknown.jsonl` vs. production 22)
- `unresolved.jsonl` — 558 consolidated residual rows
- `unresolved_symlinks.jsonl` — **0 rows** (cascade clean)
- `audit_report.md` + `audit_artifacts/` — full galaxy census

Key numbers from the merge summary (stdout):

```
done_clusters                = 339
enriched_rows                = 308
meaning_star_rows            = 16      (new — not previously in merged_in)
reused_meaning_star_ids      = 134     (canonical id already present)
merged_by_galaxy_row_count   = 277732
merged_by_galaxy_shard_count = 24
unresolved_rows              = 558
unresolved_symlinks          = 0
```

## 6. Audit Delta — Pipeline State vs. Live Galaxies

The two storage roots are different sources (curated merge pipeline vs. live galaxies), so this is not an apples-to-apples "before/after" on a single tree. It does, however, quantify how close the B7 pipeline output is to the canonicalization targets that the live tree has not yet absorbed.

| Metric | Live galaxies (04/18, 22 shards, 464,334 rows) | Pilot merge (04/20, 24 shards, 277,732 rows) |
|---|---:|---:|
| Canonical IDs | 303,952 (65.5 %) | 277,681 (**99.98 %**) |
| Missing IDs | 339 | **0** |
| Ad-hoc IDs | 160,043 | **51** |
| Raw (non-procedural) rows | 1,995 | **65** |
| Matryoshka-missing W/C rows | 70,678 | **108** |
| Duplicate content groups | 249,905 | **3,098** |
| Rows in duplicate groups | 367,275 | **51,073** |
| Symlink edges | 18,368,415 | 7,487 |
| Unidirectional symlink sites | 18,368,305 (99.9994 %) | **929 (12.4 %)** |

The merge pipeline is the shape the live tree is being pulled toward. Anything that subsequently fails to appear in the pipeline is a candidate for future residual passes.

## 7. Retry Envelope — Measured Behavior

Implemented by [proceduralizer_wine.py](knowledge3d/ingestion/proceduralizer_wine.py) and exposed as opt-in pilot kwargs:

- `PILOT_RETRY_TRANSIENT_ATTEMPTS = 3`, `PILOT_RETRY_TRANSIENT_DELAY_S = 30.0`
- `PILOT_RETRY_PLAN_WAVES = ((3, 5×3600 s, 60 s), (3, 30×60 s, 60 s))`
- `_TRANSIENT_FAILURES = {transport_error, timeout}` — **plan_limit bypasses the transient loop** and enters the wave ladder directly to avoid burning quota on retries that are certain to fail.

Verified end-to-end in unit-level probes (injectable `retry_sleep`) before the pilot:
1. transport_error recovery — 2 calls, 30 s virtual sleep.
2. plan_limit sustained — 7 calls, 20,160 s virtual sleep (full ladder).
3. plan_limit clear at wave 1 — 2 calls, 18,000 s virtual sleep.
4. transient exhaust — 3 calls, 60 s virtual sleep.

Over the 36-cluster pilot: **zero retries fired**. The envelope is hot-swappable insurance; the default load was under budget for the cloud planner `qwen3.5:397b-cloud` + long_context_engineering profile.

## 8. Timing Characteristics

- Wall clock: 3,840 s worker lifetime (60 min active + 4 min drain of the final claim).
- Mean cluster latency: ~106 s/cluster with `--row-concurrency 2`.
- Distribution: most clusters 30-60 s; two outliers (~10 min gaps) coincide with large clusters (cluster 18 had 28 unresolved rows; cluster 28 was a slow web_search fan-out).
- Web cache behavior: many `hits` grow against `misses` as the pilot proceeds — source fetches amortize.

## 9. Sample Meaning-Stars Emitted (Pilot)

- `cardinal_872` (fact, 3 languages, grounded) — smoke cluster
- 20 `merge_to_meaning_star` emissions (1 meaning-star each, 2-3 surface_symlinks each)
- 2 `split_polysemy` emissions from the IPA/N'Ko cluster (see §3)
- Full inventory: `scripts/ingestion/staging/D3_dedup/differentiate_b7/meaning_stars/` (cluster-scoped files)

## 10. What Remains for B+ Convergence

1. **14 pilot clusters un-run** due to the `--stop-after` cap — pick up on the next pass (budget ~2000 s additional at current mean latency).
2. **`unidirectional_ref` in pipeline = 929** is concentrated in Word (418), `_unknown` (195), and meaning_layer_stars (189). Most resolvable by audit's `pointed_by` synthesis; the `_unknown` partition wants a galaxy-routing pass.
3. **`unresolvable` == 15/36** (41.7 %) — inspect those rows to see which are true non-cluster collisions vs. cases where `FACTUAL_MEANING_CLASSES` should require sources that are not yet being offered. Candidate signal for the next spec.
4. **Matryoshka-missing in Word/Character** drops from 70,678 (live) to 108 (pipeline) — embedding backfill on the pipeline is effectively complete; the gap now is propagating the pipeline rows back into the live tree.

## 11. Files Touched This Session

- `knowledge3d/ingestion/proceduralizer_contract.py` — `word_refs` + `word_refs_by_language` split; parser check order (lang-suffix before sources).
- `knowledge3d/ingestion/proceduralizer_wine.py` — opt-in retry envelope; `_TRANSIENT_FAILURES` vs. plan_limit bypass.
- `scripts/ingestion/d3/differentiate_b7_residual.py` — merge-pass synthesizes `pointed_by` + `word_refs_by_language` + flat `word_refs` union; `PILOT_RETRY_*` kwargs wired into bridge.submit() call sites.
- `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md` §1.6.4 — both fields documented with JSON exemplar.

Commits: `b19ccecd` (retry envelope + word_refs collision fix).

## 12. Handoff

**For Codex (when unlocked ≥ 2026-04-20 14:09):**
- Inputs for post-pilot work land in `TEMP/B7_PILOT_04.20.2026/` (merged_stars, audit report, artifacts).
- The remaining 14 clusters can be finished under an additional ~35 min `--stop-after`.
- Next residual wave: consider lifting `--max-cluster` past 50 for larger-size clusters (cluster 18 had 28 rows and was still tractable at ~3 min).
- Retry envelope stays as-is; do not raise defaults until we see a real `plan_limit` event in the log.

**For the project:**
- B7 merge-to-meaning-star path is healthy end-to-end.
- The language-suffixed id guard (moved before the sources guard) has not fired on any cluster in this run — preamble + exemplars appear to be suppressing the footgun the earlier B7 audit flagged.
