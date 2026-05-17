# Codex Spec — B7 Meaning-Resolution Pivot (+ recover-only shard write)

**Date**: 2026-04-19
**Owner**: Claude (spec author) — acting on Daniel's "dedup by meaning, words have many meanings" directive
**Severity**: P0 for B7 architectural direction (current mode is wrong direction); P1 for recover-only plumbing (128 enrichments currently get clobbered)
**Scope**: Ingestion-path only. Patches the recover-only publish path, then pivots the cloud proceduralizer from "differentiation" mode to "meaning resolution" mode. Zero hot-path impact.

---

## Two observations from the 1-hour run

1. **The 128 enrichments got clobbered** — `recover_b6.py` reads from `merged_by_galaxy/*.jsonl` (stage_dir/merged_by_galaxy/), not from `merged_stars.jsonl`. B7's merge pass only writes `merged_stars.jsonl`, so the shards held stale rows; `_write_rows` at line 365 rebuilt `merged_stars.jsonl` from those stale shards; `_publish` at line 569 copied the stale tree over the stage. The published sha256 reverted to `3dfbba79…` on this exact mechanism. The enrichments are still in `differentiate_b7/enriched/*.jsonl`.

2. **75% unresolved means we're asking the wrong question.** The dry-run's 6/10 resolved / 4/10 unresolvable was already a hint; the real run at 128/520 = 24.6% resolved / 75% unresolvable confirms it. Looking at the actual enrichments:

   - `word_pt_num_594` ("quinhentos e noventa e quatro") and `word_five_hundred_ninety_four` ("five hundred ninety-four") are **not duplicates to differentiate** — they are **two language surfaces of one meaning star (cardinal 594)**.
   - `word_ja_num_694` shares the same meaning structure, different language, different number.
   - `word_bench_780d54050917` → "acuity" pulled from a medical dictionary. "Acuity" is polysemous: medical (sharpness of vision), cognitive (mental sharpness), musical (high-pitched clarity), business (judgment/insight). One surface, many meaning stars.

   The existing `DIFFERENTIATION_SYSTEM_PROMPT` asks "make this row distinct from peers." For cross-language surface duplicates, that prompt is architecturally wrong — they SHOULD collapse to one meaning star, not be forced apart. For polysemes, it's architecturally under-specified — one row should become multiple meaning stars, not one enriched row.

---

## The architectural pivot

Replace "differentiation mode" with **"meaning resolution mode."** The cloud model's job in a duplicate cluster becomes:

```
Input:
  - cluster: list of rows sharing a content_hash
  - web_evidence per row (same as before)
  - peer_content_sample (same as before)

Output (one of four decisions per cluster):

  (A) MERGE_TO_MEANING_STAR
      Cluster rows share one meaning across languages / scripts / stylings.
      Emit: 1 canonical meaning_star row (language-agnostic) + N surface
            symlink rows, one per original row, pointing at the meaning_star.

  (B) SPLIT_POLYSEMY
      Cluster rows share a surface form but carry distinct meanings.
      Emit: K meaning_star rows (K = distinct senses the model identifies) +
            N surface symlink rows, each pointing at its correct sense.

  (C) MIXED (partial merge + partial split)
      Some cluster rows merge into one meaning star; others split.
      Emit: M meaning_star rows + N surface symlinks with 1:M fan-out.

  (D) UNRESOLVABLE
      Web evidence does not disambiguate. Leave cluster untouched; log to
      unresolved/<hash>.jsonl exactly as today.
```

Every emitted meaning_star and surface symlink MUST cite web_evidence URLs in `metadata.sources`. No fabrication.

### Why this is dedup-by-meaning

- **Cross-language surface duplicates (most of the 51,385 residual)**: 10,000 rows of "cardinal number N" in 30 languages collapse to ~1,000 meaning stars + 30,000 surface symlinks. Row count grows in a nominal sense but dedup is radical — the Galaxy's content_hash collisions disappear because meaning stars and surface symlinks have inherently different content signatures.
- **Polysemes (minority but high-value)**: "acuity" produces 2-4 meaning stars with distinct `metadata.concept_id` + disambiguated descriptions. Dedup by meaning; multiplication of meaning stars per surface is CORRECT.
- **Genuine one-to-one rows**: stay one-to-one, no-op merge.

### What a meaning_star row looks like

```json
{
  "id": "meaning/cardinal_594",
  "galaxy": "Meaning",
  "star_type": "meaning_concept",
  "content": "cardinal_number value=594",
  "metadata": {
    "concept_id": "cardinal_594",
    "language": "language_agnostic",
    "sources": ["https://en.wikipedia.org/wiki/594_(number)", "…"],
    "resolved_from_cluster": "<content_hash>",
    "surface_count": 30
  },
  "rpn_program": "594 CARDINAL"
}
```

### What a surface_symlink row looks like

```json
{
  "id": "word_pt_num_594",
  "galaxy": "Word",
  "star_type": "surface_symlink",
  "content": "quinhentos e noventa e quatro",
  "metadata": {
    "language_profile": {"language": "pt-BR"},
    "points_to": "meaning/cardinal_594",
    "sources": ["https://largodoscorreios.wordpress.com/…"],
    "polysemy_parent_count": 1,
    "resolved_from_cluster": "<content_hash>"
  },
  "rpn_program": "meaning/cardinal_594 LANG_PT LEX"
}
```

Polysemous surfaces carry `polysemy_parent_count > 1` and `points_to` as an array of meaning-star IDs — the "acuity" case.

### What changes in the dedup pipeline

- `_meaning_group_key()` already keys on `concept_id` first. Meaning-star rows will have concrete `concept_id` values (`cardinal_594`, `acuity_vision_sharpness`, …) that land them in different groups from old generated rows. No edits needed to grouping logic — the enrichment itself fixes the duplicate-content axis because meaning stars carry distinct payloads.

- The matryoshka embedding path consumes meaning stars and surface symlinks via their respective RPN programs. The content-hash signature diverges naturally once `content` carries real semantic payload.

---

## The fix — two independent commits, executed in order

### Commit 1 — Patch recover-only to consume B7 merge output

**The bug**: `recover_b6._load_rows` reads from `stage_dir/merged_by_galaxy/*.jsonl`. B7 merge writes only `merged_stars.jsonl`. Stale shards → rebuilt `merged_stars.jsonl` → clobber.

**Simplest fix** (preserves the existing "shards are source of truth" invariant): extend the B7 merge pass to also overwrite `merged_by_galaxy/*.jsonl` with the enriched rows, sharded by the row's `galaxy` field.

**Location**: `scripts/ingestion/d3/differentiate_b7_residual.py`, the `--merge` mode.

**Edit**:

```python
def _merge(args: argparse.Namespace) -> None:
    # … existing code that builds {row_id: enriched_row} and writes
    # merged_out (merged_stars.jsonl.next) …

    # NEW: also shard the enriched output into merged_by_galaxy/
    merged_by_galaxy = args.merged_out.parent / "merged_by_galaxy"
    # stream-group the OUTPUT merged_stars.jsonl.next by galaxy
    shards: dict[str, list[str]] = {}
    with args.merged_out.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            galaxy = str(row.get("galaxy") or "_unknown")
            # file_name convention: <galaxy_lower>.jsonl (mirror current shard naming)
            shards.setdefault(galaxy_to_filename(galaxy), []).append(text)
    for file_name, lines in shards.items():
        shard_path = merged_by_galaxy / file_name
        tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp_path, shard_path)
```

`galaxy_to_filename` needs to match what `recover_b6._load_rows` expects. Verify by reading `galaxy_d3.py`'s stage-population logic — the file names in `merged_by_galaxy/` are the source of truth.

**Operator re-run** (after this commit lands):

```bash
python scripts/ingestion/d3/differentiate_b7_residual.py --merge \
  --merged-in  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
  --merged-out scripts/ingestion/staging/D3_dedup/merged_stars.jsonl.next \
  --enriched-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/enriched/ \
  --unresolved-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/unresolved/

mv scripts/ingestion/staging/D3_dedup/merged_stars.jsonl{.next,}
bash scripts/ingestion/d3/run.sh --recover-only
```

Expected: the published `merged_stars.jsonl` sha256 is NOT `3dfbba79…` this time. `duplicate_row_count` should drop by roughly 128-256 (each enriched row removes 1 from its cluster; some clusters may fully resolve).

This is a small, confidence-building commit. It also closes the feedback loop for the pivot in Commit 2.

### Commit 2 — Meaning-resolution mode (the pivot)

Introduce a new `ProceduralizerRequest.mode = "meaning_resolution"` alongside the existing `"differentiation"` and `"standard"` modes. Do NOT delete differentiation mode — keep it for corner cases (e.g., distinct fictional characters that share a surface).

**Edit 1**: `knowledge3d/ingestion/proceduralizer_contract.py`

Extend `ProceduralizerBundle` with an outcome field and fan-out rows:

```python
@dataclass
class ProceduralizerBundle:
    # existing fields...
    outcome: str = "enriched"
        # "enriched" (legacy differentiation mode)
        # "merge_to_meaning_star"    (outcome A)
        # "split_polysemy"           (outcome B)
        # "mixed"                    (outcome C)
        # "unresolvable"             (outcome D)
    meaning_stars: list[dict[str, Any]] | None = None
        # list of canonical meaning_concept rows emitted for this cluster
    surface_symlinks: list[dict[str, Any]] | None = None
        # list of surface_symlink rows emitted for this cluster,
        # each carrying a "points_to" field (str for merge/split,
        # list[str] for polysemy fan-out)
```

`parse_bundle()` in `meaning_resolution` mode MUST validate:
- If `outcome` is `merge_to_meaning_star`: `len(meaning_stars) == 1`, `len(surface_symlinks) == cluster_size`, every symlink's `points_to` equals the meaning star's id.
- If `outcome` is `split_polysemy`: `len(meaning_stars) == K > 1`, every symlink's `points_to` is in the meaning-star id set.
- If `outcome` is `mixed`: `len(meaning_stars) >= 1`, symlinks may have fan-out (`points_to` as list).
- If `outcome` is `unresolvable`: `meaning_stars` and `surface_symlinks` are empty.
- Every emitted meaning_star and every emitted symlink has at least one entry in `metadata.sources`.

### Edit 2: System prompt

Add `MEANING_RESOLUTION_SYSTEM_PROMPT` to `proceduralizer_contract.py`. Core instruction:

> "You are in MEANING RESOLUTION MODE. The rows in this cluster share a
> content hash. Your task is to decide whether they share a meaning or not,
> and emit one of four outcomes.
>
> If the rows are language/script/styling variants of ONE concept — e.g.,
> 'hello' (en), 'hola' (es), 'olá' (pt) all meaning greeting — emit
> outcome=merge_to_meaning_star: one canonical language-agnostic
> meaning_concept row, plus one surface_symlink row per original,
> points_to the meaning star.
>
> If the rows share a surface but carry distinct senses — e.g., 'bank'
> as financial institution vs. river edge — emit outcome=split_polysemy:
> K meaning_concept rows (one per distinct sense you can ground in
> web_evidence), plus one or more surface_symlink rows pointing at
> the correct sense.
>
> If some rows merge and others split, emit outcome=mixed.
>
> If web_evidence does not support a confident decision, emit
> outcome=unresolvable with empty arrays. Do NOT guess.
>
> Every meaning_concept and every surface_symlink MUST cite at least
> one web_evidence URL in metadata.sources. Do not fabricate."

### Edit 3: Entry-point

`knowledge_proceduralizer.py`: add `resolve_cluster_by_meaning()` alongside `differentiate_cluster()`. Same shape (peer samples + web evidence + ollama call), but requests `mode="meaning_resolution"` and returns a bundle with the richer outcome structure.

### Edit 4: Driver flag

`scripts/ingestion/d3/differentiate_b7_residual.py`: add `--mode {differentiate, meaning_resolution}`; default changes to `meaning_resolution` once this commit lands. Worker cluster processor dispatches to the correct cluster function per `--mode`.

### Edit 5: Merge pass fan-out

The existing `--merge` pass consumes `enriched/*.jsonl` assuming each file is a list of full rows keyed by original row_id. Meaning resolution mode produces:
- `meaning_stars/<cluster>.jsonl` — NEW rows to append to merged_stars.jsonl
- `symlinks/<cluster>.jsonl` — REPLACEMENT rows keyed by original row_id (same row_id, new star_type = surface_symlink)

Update the merge pass to:
1. Stream-append `meaning_stars/*.jsonl` as brand-new rows to the output merged_stars.jsonl (and to the corresponding Meaning-galaxy shard in merged_by_galaxy/).
2. Stream-replace rows keyed by `row_id` with `symlinks/*.jsonl` entries (same mechanism as the old `enriched` path).
3. Validate: every symlink's `points_to` target(s) exist in either the pre-existing meaning stars or in the new `meaning_stars/*.jsonl`. If a symlink points at a nonexistent meaning star, log to `unresolved_symlinks.jsonl` and skip.

### Edit 6: Doc patch

`docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md`: add "§Y Meaning Resolution Mode" section. Reference this spec. List the four outcomes + the new bundle fields + the merge semantics.

---

## Operator flow after Commit 2

### Step 1 — Clear prior state (fresh claim/done space)

The meaning-resolution outcomes are different artifacts than differentiation's. Move the old staging root aside rather than in-place overwrite:

```bash
mv scripts/ingestion/staging/D3_dedup/differentiate_b7 \
   scripts/ingestion/staging/D3_dedup/differentiate_b7.legacy_20260419
```

### Step 2 — Rebuild the manifest (no code change; uses the same violations.jsonl)

```bash
python scripts/ingestion/d3/differentiate_b7_residual.py \
  --build-manifest \
  --violations scripts/ingestion/staging/D3_dedup/re_audit_d3/violations.jsonl \
  --min-cluster 2 --max-cluster 50 \
  --out-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/clusters/
```

### Step 3 — Launch workers in meaning-resolution mode (4 workers, 1-hour budget)

```bash
bash scripts/ingestion/d3/launch_b7_workers.sh 4 3600 --mode meaning_resolution
```

Note the mode flag. Workers consume the same cluster manifests; they just run a different cluster function.

### Step 4 — Merge (meaning-star fan-out included)

```bash
python scripts/ingestion/d3/differentiate_b7_residual.py --merge \
  --merged-in  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
  --merged-out scripts/ingestion/staging/D3_dedup/merged_stars.jsonl.next \
  --enriched-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/enriched/ \
  --meaning-stars-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/meaning_stars/ \
  --symlinks-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/symlinks/ \
  --unresolved-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/unresolved/

mv scripts/ingestion/staging/D3_dedup/merged_stars.jsonl{.next,}
bash scripts/ingestion/d3/run.sh --recover-only
```

### Step 5 — Report back to me

For the first 1-hour meaning-resolution run:
- clusters_done / 3083
- outcome breakdown: `merge_to_meaning_star / split_polysemy / mixed / unresolvable`
- meaning_star rows emitted (new row count to merged_stars.jsonl)
- surface_symlink rows emitted (replacement count)
- NEW `duplicate_row_count` — expected to drop substantially (hypothesis: 20-40% reduction on the 2-50 band, driven by cross-language surface collapse)
- 3 sample outcomes of each type (pasted bundle summaries — I want to see the model's reasoning on a concrete merge, a concrete polysemy split, and a concrete unresolvable)
- web_cache hit rate (should be much higher this run; ~1.92% was first-warm — subsequent runs share the cache)

---

## Sovereignty + hot-path impact: zero

Everything here is ingestion-path Python + cloud HTTP + shared filesystem. No hot-path file changes. No registry changes. No kernel changes. The matryoshka .bin cross-check between Python-path and C++/CUDA-path still holds byte-for-byte — meaning stars and surface symlinks are just different row shapes on the same pipeline.

---

## What NOT to do

- **Do NOT delete `differentiation` mode.** Keep it as a fallback for true-differentiation cases (named entities, distinct artifacts that happen to hash-collide). Meaning resolution is the new default, not the only mode.
- **Do NOT** re-run differentiation mode on the 128 clusters already enriched in the 1-hour run. Move `differentiate_b7/` aside as `differentiate_b7.legacy_20260419/` first. Those 128 rows will be re-evaluated under meaning_resolution and likely get resolved correctly (most were cross-language duplicates that should have been merges, not differentiations).
- **Do NOT** touch `_meaning_group_key()`. The grouping logic is correct; it was the downstream enrichment direction that was wrong.
- **Do NOT** auto-link polysemous surfaces to meaning stars without web citation. `metadata.sources` is mandatory on every emitted row.
- **Do NOT** expand to clusters > 50 yet. The large monsters (up to 400 rows) will mostly be merge_to_meaning_star outcomes, but let the small-band pass validate the prompt first.

---

## Related files

- `scripts/ingestion/d3/differentiate_b7_residual.py` — add `--mode`, shard-write fix, meaning-star merge fan-out
- `scripts/ingestion/d3/recover_b6.py` — no edits needed once shard-write lands
- `knowledge3d/ingestion/proceduralizer_contract.py` — extend bundle + prompt + validator
- `knowledge3d/ingestion/mcp_web_search.py` — unchanged
- `knowledge3d/tools/knowledge_proceduralizer.py` — add `resolve_cluster_by_meaning()`
- `knowledge3d/knowledgeverse/proceduralizer_stargate.py` — if it materializes bundles to rows, extend to meaning-star + symlink shapes
- `docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md` — §Y Meaning Resolution Mode
- `TEMP/CLAUDE_CODEX_B7_CLOUD_PROCEDURALIZER_DIFFERENTIATION_04.19.2026.md` — predecessor spec (still in force for differentiation mode)
- `TEMP/CLAUDE_CODEX_B7_PARALLEL_DIFFERENTIATION_DRIVER_04.19.2026.md` — predecessor spec (the parallel driver stays as-is)

---

**Estimated effort**: Commit 1 ~45 min (one file, ~50-line addition). Commit 2 ~4-6 h (contract extension + prompt + validator + entry-point + driver flag + merge fan-out + doc patch + smoke test).
**Blocks**: Gate 1 meaningful close-out. Current 51,385 residual is 90%+ cross-language surface duplicates that meaning resolution collapses by symlinking.
**Blocked by**: Nothing. Commit 1 is independent; Commit 2 builds on Commit 1 (symlinks need shard write to persist).
**Location**: `TEMP/CLAUDE_CODEX_B7_MEANING_RESOLUTION_PIVOT_04.19.2026.md`
