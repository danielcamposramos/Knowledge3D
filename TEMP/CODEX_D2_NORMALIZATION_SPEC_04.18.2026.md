# CODEX D2 Normalization Spec — 2026-04-18

Continuation from `CODEX_D1_AUDIT_REPORT_04.18.2026.md`.

D1 surfaced the violation topology across 22 galaxies / 464,334 rows. D2 is
the **Correction Engine**: a deterministic, staged rewrite that resolves
every D1 violation class into a parallel clean JSONL tree. **No live
galaxy row is rewritten during D2.** The live swap is a D3 concern and
gated on D2 artifact parity + re-audit.

## Reproducer contract

Entry point (one command):
```
bash scripts/ingestion/normalize/run.sh
```
Must produce, in `scripts/ingestion/staging/D2_normalize/`:
```
normalized/<galaxy>.jsonl            # rewritten rows, canonical IDs, backfilled payloads
refs_rewrite_map.jsonl               # old_id → canonical_id (every ad-hoc + dup collapse)
bidirectional_edges.jsonl            # edges added to make symlinks two-way
orphan_targets.jsonl                 # refs whose target never resolved (tombstoned)
matryoshka_fills.jsonl               # Word/Character rows that got backfilled payloads
procedural_upgrades.jsonl            # raw rows converted to procedural form
D2_NORMALIZATION_REPORT.md           # narrative (mirrors D1 report style)
hashes.txt                           # sha256 of every artifact above
```

Second reproducer run must match byte-for-byte — record hashes in
`hashes.txt` and re-verify in CI.

## Dependency-ordered passes

Run passes in this order — each pass consumes the previous pass's output,
never the live tree.

### D2a — Canonical ID assignment (resolves 339 missing_id + 160,043 ad_hoc_id + 367,275 duplicate_content)

For every row:
1. Compute `content_hash` = blake2b(canonical_serialization(row_without_id_and_refs), digest_size=16).
2. Bucket rows by `content_hash`. Within each bucket:
   - If any row already has a canonical ID (matches `^k3d-[a-z_]+/[0-9a-f]{16}$`
     or the galaxy's canonical pattern — see `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md`),
     pick the lexicographically smallest as **canonical**.
   - Else synthesize `k3d-{galaxy_slug}/{content_hash[:16]}`.
3. Emit canonical row once into `normalized/<galaxy>.jsonl`.
4. For every non-canonical row in the bucket, record `{old_id, canonical_id, galaxy, content_hash}`
   in `refs_rewrite_map.jsonl`. Do NOT emit the duplicate.

Handle the 339 missing-ID rows identically: they fall into whatever
content-hash bucket they land in and inherit the canonical ID.

Accept criteria:
- Sum over galaxies of `normalized/*.jsonl` row counts ==
  303,952 (D1 canonical) + (160,043 − collapsed_to_existing) + 339 − absorbed_dupes.
- `refs_rewrite_map.jsonl` covers at minimum all 367,275 duplicate rows +
  all 160,043 ad-hoc IDs where a canonical form already existed elsewhere.

### D2b — Reference rewrite across the whole tree

Using `refs_rewrite_map.jsonl` as the authoritative substitution table,
rewrite every reference field (`symlinks`, `meaning_layer_id`, `parent_id`,
`see_also`, every `*_id` / `*_ids` field) so that every pointer targets a
canonical ID. Unknown targets are NOT rewritten — they stay as-is and
feed into D2d.

Determinism rule: fields are rewritten in a fixed traversal order
(alphabetical keys, stable JSON). Re-sort lists after rewrite to remove
order-dependent hash drift.

### D2c — Bidirectional symlink materialization (resolves 18,368,305 unidirectional_ref)

After D2b, walk the rewritten tree. For every edge `A --symlink--> B`
that does not have a reciprocal `B --symlink--> A`, emit a back-edge
into `bidirectional_edges.jsonl` and merge into the target row's
symlink list (append-only, dedup, stable sort).

Back-edge labelling: the reverse edge uses the inverse relation from
`docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md` §7 (e.g. `glyph_of`
↔ `has_glyph`). If the inverse is undefined, tag the edge `symlink_generic`
and log a warning in the report — do NOT invent semantics.

Skip edges that target missing IDs (those are D2d's problem).

### D2d — Missing-target tombstoning (resolves 18,365,046 missing_target)

For every `*_id` field where the target does not exist in the normalized
tree and was not resolved by D2b, emit a row into `orphan_targets.jsonl`:
```
{"source_row_id": ..., "field": ..., "target_id": ..., "galaxy": ...}
```
Then null the field in the normalized row (preserve structure; set the
value to `null` for scalars, drop from lists). Do NOT delete the source
row — the orphan field is a data issue, not a row-level failure.

Rationale: 18.3M missing targets is too many to heal in this pass; D2
records them so a later ingestion pass (star_crafter Phase 7.x) can
backfill the missing targets properly.

### D2e — Matryoshka payload backfill (resolves 70,678 missing_matryoshka)

Only Word (68,070) + Character (2,608) galaxies have missing payloads
per D1. For each such row:
1. Look up the meaning_layer_star via `meaning_layer_id` (should exist
   post-D2b — if not, log under `orphan_targets.jsonl` with a
   `missing_matryoshka_source` tag and skip).
2. Compute Matryoshka payloads at the canonical dims (64, 128, 256, 512)
   via the existing ingestion pipeline:
   `knowledge3d/ingestion/matryoshka_embedder.py` — call it as a library,
   not a subprocess. It already uses the Phenom-host Ollama embedder per
   `reference_second_ollama_host_phenom.md`.
3. Attach `matryoshka: {dim_64: [...], dim_128: [...], ...}` to the row
   and record the fill in `matryoshka_fills.jsonl`.

This is ingestion-side; it's fine for this pass to use numpy, sklearn,
or any ingestion library. Only the output JSONL must be sovereign.

### D2f — Raw → procedural conversion (resolves 1,995 raw_payload)

Distribution per D1: Word 648, Reality 272, Drawing 214, Grammar 121,
Math 155, Tool 25, Audio 0, Character 1, Reality 272, Drawing 214. Most
are legacy rows written before the procedural contract.

For each raw row, call the galaxy-specific proceduralizer:
- Drawing → `scripts/ingestion/proceduralize_drawing.py` (RPN
  primitives LINE/CIRCLE/RECT)
- Grammar → `scripts/ingestion/proceduralize_grammar.py` (RPN rule DSL)
- Math → `scripts/ingestion/proceduralize_math.py` (RPN expression tree)
- Word/Character → symlink to canonical procedural form already in the galaxy
- Reality → `scripts/ingestion/proceduralize_reality.py`

If a proceduralizer does not exist, emit a row into
`procedural_upgrades.jsonl` with `status: "deferred"` and leave the
original row in place. Do NOT write a fake procedural body. 1,995 is
small; the real loss if 300-400 remain deferred is acceptable and
scoped for a later pass.

### D2g — Staging validation + re-audit

Re-run `bash scripts/ingestion/audit/run.sh` with the root pointed at
`scripts/ingestion/staging/D2_normalize/normalized/` (add a
`--galaxies-root` flag if not already present). Validate that the
post-D2 census shows:

- `missing_id` ≤ 0
- `ad_hoc_id` ≤ 0 (every row has a canonical ID)
- `duplicate_content` ≤ 0 (collapsed in D2a)
- `missing_matryoshka` close to 0 (only deferred cases from D2e)
- `raw_payload` drops to the count of `procedural_upgrades.jsonl` deferred rows
- `unidirectional_ref` drops to ≤ count of edges into orphan targets
- `missing_target` drops to the `orphan_targets.jsonl` row count

Write `D2_NORMALIZATION_REPORT.md` with the new census table side-by-side
with D1's numbers, in the same format as the D1 report.

Determinism check: rerun the whole D2 pipeline once; the `hashes.txt`
output must be byte-for-byte identical. If not, a non-deterministic sort
or iteration leaked in — fix before handoff.

## Tooling + sovereignty stance

- D2 is ingestion; numpy / sklearn / scipy / pandas / torch are all fine.
  Per Daniel 04.18: "Ingestion is green to use old libraries — as long as
  we end up in sovering standards."
- The OUTPUT contract is sovereign: every normalized JSONL row must be a
  procedural or canonical-symlink row that the sovereign GPU loader can
  ingest without reaching back into Python for formatting.
- Run `bash scripts/sovereignty_preflight.sh` before committing D2 code.
  D2 lives under `scripts/ingestion/`, not the hot path, so the preflight
  should stay green regardless — but confirm, don't assume.

## Out of scope for D2

- Live tree rewrites (that's D3 swap).
- Re-embedding of meaning_layer_stars (D2 only fills Word/Character
  Matryoshka from existing meaning-layer embeddings).
- Symlink semantic validation beyond the canonical inverse table (a
  separate correctness pass belongs to D4).
- Any changes to PTX kernels, RPN opcode registry, or hot-path Python.

## Hand-back

When D2 completes, produce `TEMP/CODEX_D2_NORMALIZATION_REPORT_04.18.2026.md`
mirroring the D1 report style (headline counts, per-galaxy delta table,
determinism hashes, reproducer command). Daniel + Claude will read it
before green-lighting D3.

— Claude (architecture partner)
