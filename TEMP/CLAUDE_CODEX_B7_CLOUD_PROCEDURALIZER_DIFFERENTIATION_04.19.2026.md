# Codex Spec — Cloud Proceduralizer Differentiation Pass (B7 Residual Cleanup)

**Date**: 2026-04-19
**Owner**: Claude (spec author) — acting on Daniel's "spawn a proceduralizer
ollama cloud model to complete the original set" directive
**Severity**: P1 for D3 dedup correctness (Gate 1 residual: 51,385 duplicate_content rows)
**Scope**: Ingestion-path only. Extends existing `knowledge_proceduralizer.py` with a
cluster-differentiation mode driven by `qwen3.5:397b-cloud` and the MCP
`web_search` tool. No hot-path touches. No kernel touches. No registry touches.

---

## Why this spec exists

After the 2026-04-19 narrow `_meaning_group_key()` fix (benchmark-prefix + language
fallback + slug punctuation), the recovered audit state is:

| Metric | Value |
|---|---|
| `duplicate_row_count` | **51,385** (all `violation_kind = duplicate_content`) |
| `unidirectional_site_count` | 38 (stable) |
| `missing_target` | 0 |
| `missing_matryoshka` | 0 |
| `raw_payload` | 0 |

Sample from `scripts/ingestion/staging/D3_dedup/re_audit_d3/violations.jsonl`:

- `k3d-3dobjects/004b4211755a45ad` → `content hash shared by 4 rows` (small cluster)
- `k3d-3dobjects/019a868d8a51fa68` → **`content hash shared by 400 rows`** (large cluster)

These are NOT benchmark-prefix collisions. They are rows that were generated from
the same procedural template with no distinguishing per-row metadata. The 400-row
3DObjects cluster almost certainly represents 400 different real-world objects
that were stubbed out with one shared template and never enriched.

Daniel's directive (verbatim):

> *"we can now spawn a proceduralizer ollama cloud model to complete the
> original set (extend what's missing, properly give it metadata and detail -
> using the internet, not only training data!)."*

The solution is enrichment, not deletion: fetch per-instance metadata from the
web so each row becomes legitimately distinct (and genuinely more complete), at
which point `duplicate_content` collisions resolve naturally because the rows
actually ARE different once they carry real metadata.

---

## What's already in place (do NOT rebuild)

- `knowledge3d/tools/knowledge_proceduralizer.py` — existing proceduralizer CLI
  + library, Ollama-driven, parses `ProceduralizerBundle` output
- `knowledge3d/ingestion/proceduralizer_contract.py` — bundle / packet / request
  types and the canonical system prompt
- `knowledge3d/ingestion/ollama_manager.py` — Ollama driver
- `knowledge3d/knowledgeverse/proceduralizer_stargate.py:bundle_to_payload_rows`
  — converts bundles to merged_stars.jsonl row shape
- `docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md` — canonical spec
  (includes `qwen3.5:397b-cloud` as the `long_context_engineering` profile)
- MCP `ollama-specialists` server exposes `web_search` and `ask_cloud` tools

**What's missing**: (1) a B7-duplicate-cluster input adapter, (2) a web_search
enrichment hook per row, (3) a write-back path that updates merged_stars.jsonl
with the enriched rows, (4) a dedup-friendly re-audit gate that confirms the
delta cleared residual duplicates.

---

## The fix — five narrow edits + one new entry-point script

### Edit 1 — Add a `differentiation` mode to `ProceduralizerRequest`

**Location**: `knowledge3d/ingestion/proceduralizer_contract.py`

Extend `ProceduralizerRequest` with two optional fields:

```python
@dataclass
class ProceduralizerRequest:
    # ... existing fields ...
    mode: str = "standard"              # "standard" | "differentiation"
    peer_content_sample: list[str] | None = None
    # peer_content_sample: for "differentiation" mode, up to 3 short text
    # summaries of OTHER rows in the same duplicate cluster, so the model
    # can see "here is what siblings already say; make this row distinct
    # by extending it with web-sourced specifics that the siblings lack."
    web_evidence: list[dict[str, Any]] | None = None
    # web_evidence: a list of {"url": ..., "title": ..., "snippet": ...}
    # items returned from web_search for this row's anchor concept. The
    # proceduralizer MUST cite at least one when it introduces a concrete
    # detail in differentiation mode.
```

Extend `PROCEDURALIZER_SYSTEM_PROMPT` (or add a `DIFFERENTIATION_SYSTEM_PROMPT`)
with a differentiation-mode preamble:

> "You are in DIFFERENTIATION MODE. The row you see shares a content hash with
> N peer rows in the same galaxy. Peer content samples are provided so you can
> see what they already cover. Your task: using the `web_evidence` attached,
> extend this row with concrete, sourced details (dimensions, dates, taxonomic
> names, named parts, authoritative definitions, measurable properties) that
> the peers do NOT already contain. Cite at least one `web_evidence` URL in
> `metadata.sources`. If the `web_evidence` is empty OR returns nothing
> distinctive for this row, emit a bundle with `status = "unresolvable"` and
> an empty payload — do NOT fabricate."

### Edit 2 — Add a `web_search` client shim

**New file**: `knowledge3d/ingestion/mcp_web_search.py`

Thin HTTP client over the MCP ollama-specialists `web_search` tool (running at
`:8502`). Must:

- Accept a query string + `max_results` (default 5)
- Return a list of `{"url", "title", "snippet"}` dicts
- Cache results on disk (`scripts/ingestion/staging/D3_dedup/cache/web_search/<sha256(query)>.json`)
  so re-runs are idempotent and cheap
- On timeout / 5xx: raise `WebSearchUnavailable`, do NOT silently return empty
  (per sovereignty doctrine: "We fail and fix" — no Python fallbacks)

Query shape per row: derive from the row's concept anchor, not the template.
For 3DObjects cluster: use `row.metadata.asset_name` or `row.name` if present,
else `row.id` stem, else `row.source_hint`. Append a small disambiguator
(`{anchor} object`, `{anchor} 3d model reference`, etc.) depending on galaxy.

**Sovereignty note**: This is ingestion-path Python. `requests` + stdlib JSON
are acceptable here per CLAUDE.md "Ingestion Path = Flexible".

### Edit 3 — Add a `differentiate_cluster()` entry-point in knowledge_proceduralizer.py

**Location**: `knowledge3d/tools/knowledge_proceduralizer.py`

New public function:

```python
def differentiate_cluster(
    cluster_row_ids: list[str],
    merged_stars_path: Path,
    web_search: Callable[[str, int], list[dict[str, Any]]],
    ollama: OllamaManager,
    model: str = "qwen3.5:397b-cloud",
    num_ctx: int = 65536,
    max_peer_samples: int = 3,
    max_web_results: int = 5,
) -> list[ProceduralizerBundle]:
    """
    For each row in the cluster:
      1. Load the current row from merged_stars.jsonl by row_id
      2. Build a peer_content_sample by pulling up to max_peer_samples OTHER
         rows from the same cluster (short text only — keep prompt bounded)
      3. Derive a web query from the row's anchor concept and fetch
         max_web_results hits via the web_search callable
      4. Construct a ProceduralizerRequest with mode="differentiation",
         peer_content_sample, web_evidence
      5. Call ollama.generate(model, system=DIFFERENTIATION_SYSTEM_PROMPT, ...)
         with num_predict=3072, temperature=0.1
      6. parse_bundle() the result and return the list
    Bundles with status="unresolvable" are returned as-is and handled by the
    write-back step (see Edit 4).
    """
```

Must NOT:
- Call the model without a web_evidence attempt first. If web_search raises
  `WebSearchUnavailable`, propagate; do not continue without evidence.
- Proceed if `cluster_row_ids` has length 1 (not a cluster).

### Edit 4 — Write-back script: `scripts/ingestion/d3/differentiate_b7_residual.py`

**New file**. Single-entry-point CLI driver. Steps:

1. Parse args: `--violations`, `--merged-in`, `--merged-out`, `--min-cluster`,
   `--max-cluster`, `--dry-run`, `--limit`
2. Load `violations.jsonl`, group `duplicate_content` rows by `content_hash`
3. For each cluster above `--min-cluster` (default 2) and below `--max-cluster`
   (default 50 for the first pass — skip the 400-row monster until small ones
   prove the pipeline):
   a. Call `differentiate_cluster()`
   b. For each resolved bundle: convert via `bundle_to_payload_rows()` and
      splice the enriched row into a fresh merged_stars.jsonl at
      `--merged-out` (keep row ordering stable; replace the old row by row_id)
   c. For `unresolvable` bundles: leave the row untouched AND write a
      `{"row_id": ..., "reason": "unresolvable"}` line into
      `scripts/ingestion/staging/D3_dedup/differentiate_b7_unresolved.jsonl`
      so the follow-up audit-tightening pass can see which clusters need
      human curation or true-merge treatment
4. Emit a summary: `{clusters_attempted, clusters_resolved, rows_enriched,
   rows_unresolved, web_searches_issued, web_cache_hits}`
5. Do NOT rerun D3 automatically; that's the Step 3 verification pass below.

### Edit 5 — Small doc patch to KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md

Add a new section "§X Differentiation Mode" pointing to this spec, listing the
new request fields, and noting the web_search dependency. One paragraph plus
the field list.

---

## Sovereignty boundary — explicit

- **Ingestion path**: `differentiate_b7_residual.py`, `mcp_web_search.py`,
  Ollama cloud calls, web fetches. ALL allowed per CLAUDE.md "Ingestion Path =
  Flexible". The output is sovereign (Galaxy-consumable merged_stars.jsonl).
- **Hot path**: untouched. No kernel, no PTX, no TRM adapter, no
  knowledgeverse.py change. Zero new dependencies leak into the hot path.
- **`qwen3.5:397b-cloud`** is already profiled in the existing proceduralizer
  (`long_context_engineering` profile in MODEL_OPTIONS at
  `knowledge3d/tools/knowledge_proceduralizer.py:69`). Differentiation mode
  uses this profile by default.
- **"Using the internet, not only training data"**: the web_evidence pipeline
  is MANDATORY. A bundle that introduces a concrete detail without a cited
  URL in `metadata.sources` is a spec violation and must be rejected by
  `parse_bundle()` in differentiation mode. Add that validation.

---

## Execution order

### Step 1 — Land Edits 1-3 + Edit 5 + unit smoke test (one commit)

Commit the contract/prompt extension, the web_search client, the
`differentiate_cluster()` function, and the doc patch. Smoke test with a
hand-crafted 2-row cluster to confirm the prompt+tool plumbing works
end-to-end against the live `qwen3.5:397b-cloud` endpoint.

### Step 2 — Land Edit 4 (the driver script) + dry-run report (separate commit)

Commit the driver. Then run:

```bash
python scripts/ingestion/d3/differentiate_b7_residual.py \
  --violations scripts/ingestion/staging/D3_dedup/re_audit_d3/violations.jsonl \
  --merged-in  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
  --merged-out /tmp/merged_stars.diff_preview.jsonl \
  --min-cluster 2 --max-cluster 50 --limit 10 \
  --dry-run
```

Paste the summary counts into the commit body. Sanity-check 3 enriched bundles
by hand: each must cite a URL in `metadata.sources`, each must differ from its
peers in at least one concrete field beyond the template.

### Step 3 — Full pass on clusters 2-50 (separate commit + re-audit)

```bash
python scripts/ingestion/d3/differentiate_b7_residual.py \
  --violations scripts/ingestion/staging/D3_dedup/re_audit_d3/violations.jsonl \
  --merged-in  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
  --merged-out scripts/ingestion/staging/D3_dedup/merged_stars.jsonl.next \
  --min-cluster 2 --max-cluster 50
```

Then swap in the new merged_stars.jsonl and rerun the recovery audit:

```bash
mv scripts/ingestion/staging/D3_dedup/merged_stars.jsonl{.next,}
bash scripts/ingestion/d3/run.sh --recover-only
```

Target: `duplicate_row_count` drops materially (estimate ≥ 30-40% of residual
51,385 cleared — clusters in the 2-50 size band are likely the bulk by count
even if not by mass).

### Step 4 — Large-cluster pass (the 400-row monsters) — separate commit

Only after Step 3 is green:

```bash
python scripts/ingestion/d3/differentiate_b7_residual.py \
  --violations scripts/ingestion/staging/D3_dedup/re_audit_d3/violations.jsonl \
  --merged-in  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
  --merged-out scripts/ingestion/staging/D3_dedup/merged_stars.jsonl.next \
  --min-cluster 51 --max-cluster 1000
```

Large clusters are likely to produce more `unresolvable` bundles — the
web_search cache helps here because many peer rows will share parent
concepts. Expect `differentiate_b7_unresolved.jsonl` to grow; that file is the
input for the next (separate) audit-tightening spec.

### Step 5 — Final re-audit + cross-check matryoshka .bin

After Step 4's merged_stars.jsonl lands, rerun the full pipeline:

```bash
bash scripts/ingestion/d3/run.sh
sha256sum scripts/ingestion/staging/D3_dedup/matryoshka_embeddings.bin

scripts/ingestion/d3/build/matryoshka_bin_producer \
  --input  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
  --output /tmp/cc_cpp.bin \
  --force-regenerate --verbose
sha256sum /tmp/cc_cpp.bin
```

Both hashes must match. The .bin hash itself will legitimately change from
`898cb773…` because the row set is now richer. That's expected; what matters
is the Python-path and C++-path agree byte-for-byte on the new set.

---

## What NOT to do in this work

- **Do NOT** delete rows as a shortcut to close the gate. Differentiation is
  enrichment, not culling.
- **Do NOT** fabricate metadata without web_evidence citations. Bundle
  validation must reject this.
- **Do NOT** merge peer rows into one canonical row as a shortcut. That's a
  separate audit-tightening decision and requires human sign-off per cluster.
- **Do NOT** run the large-cluster pass before the small-cluster pass proves
  the pipeline.
- **Do NOT** touch the hot path, the registry, any PTX kernel, or any
  `knowledgeverse.py` file.
- **Do NOT** bypass the web_search cache — a full re-run must be reproducible
  offline once the cache is warm.

---

## Related files

- `knowledge3d/tools/knowledge_proceduralizer.py` — extend with `differentiate_cluster()`
- `knowledge3d/ingestion/proceduralizer_contract.py` — extend request schema
- `knowledge3d/ingestion/ollama_manager.py` — call site (unchanged)
- `knowledge3d/knowledgeverse/proceduralizer_stargate.py` — write-back shape
- `docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md` — §X Differentiation Mode
- `scripts/ingestion/d3/differentiate_b7_residual.py` — NEW driver
- `knowledge3d/ingestion/mcp_web_search.py` — NEW web_search client
- `scripts/ingestion/staging/D3_dedup/re_audit_d3/violations.jsonl` — input
- `scripts/ingestion/staging/D3_dedup/cache/web_search/` — NEW cache dir

---

**Estimated effort**: Step 1 ~2h, Step 2 ~1h, Step 3 ~run-time + 30min verify,
Step 4 ~run-time + 1h verify, Step 5 ~run-time + cross-check.
**Blocks**: Gate 1 final close-out and the sovereign procedural symlinked
architecture going end-to-end-live.
**Blocked by**: Nothing. The B7 narrow fix has landed; this is the follow-up.
**Location**: `TEMP/CLAUDE_CODEX_B7_CLOUD_PROCEDURALIZER_DIFFERENTIATION_04.19.2026.md`
