# Codex Handoff: B7 Word-Refs + Retry-Envelope Pilot

**Date:** 2026-04-19
**Author:** Claude (architecture)
**Commit:** `542cbb3d` — "B7 meaning cascade: word_refs, retry envelope, scoped sources"

## What Landed (5 files)

1. `knowledge3d/ingestion/proceduralizer_contract.py`
   - Added `word_refs` field to schema (dict keyed by ISO-639-1, values = arrays of word-star ids)
   - Added `FACTUAL_MEANING_CLASSES` + `_is_factual_meaning()` helper
   - Sources now required only for factual meaning classes (`fact`, `scientific_constant`, `physical_constant`, `mathematical_constant`, `named_entity`, `person`, `place`, `organization`, `title`, `product`, `iso_standard`, `standard`, `date`, `taxonomic_name`, `unit`, `si_unit`, `chemical_element`, `compound`) OR when `domain` is in `{reality, physics, chemistry, biology, math, mathematics}`
   - Two positive exemplars in preamble: factual (numeric "208") and common vocabulary ("after")
   - Parser reordered: language-suffix guard fires BEFORE sources guard so bad ids surface with specific diagnostic
   - Schema no longer requires `sources` (was `minItems: 1`, removed from `required`)

2. `knowledge3d/ingestion/proceduralizer_wine.py`
   - Retry envelope constants:
     - `DEFAULT_RETRY_TRANSIENT_ATTEMPTS = 3`
     - `DEFAULT_RETRY_TRANSIENT_DELAY_S = 30.0`
     - `DEFAULT_RETRY_PLAN_WAVES = ((3, 5h, 60s), (3, 30min, 60s))`
   - `_RETRYABLE_FAILURES = {"transport_error", "timeout", "plan_limit_consumed"}`
   - `submit()` now iterates Step 1 (transient retries 30s x 3) then Step 2+ (plan waves) if `failure_code == plan_limit_consumed`
   - Non-limit transient failures also retry 30s x 3 (matches directive)
   - **Bug fix:** when `result.returncode != 0`, old code used `parse_bundle("", …)` → `invalid_json` as the failure_code, preventing retry. Now defaults to `transport_error` so the envelope fires.
   - `_one_chat_attempt()` extracted: single bridge round-trip (prepare → chat → write raw → parse bundle).

3. `scripts/ingestion/d3/differentiate_b7_residual.py`
   - Merge pass (lines ~1038–1095) synthesises `word_refs` alongside `pointed_by` from surface_symlink → meaning_star edges.
   - `word_refs` derived from `surface_forms` language keys on symlink rows, merged with any already-present `word_refs` on the meaning star.

4. `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md`
   - New §1.6.4 "Meaning-Layer Symlinks: `word_refs` by Language"
   - Cascade diagram: drawing primitives → character stars → word stars → meaning stars
   - Scoped sources rationale (factual required, common optional)
   - Missing-word fallback (surface_forms strings remain valid during proceduralization; compaction upgrades to `word_refs` once word stars materialise)

5. `TEMP/CODEX_D3_FINAL_REPORT_04.19.2026.md` — auto-regenerated B6 recovery numbers (not new content).

## What Codex Runs Next

Sovereignty rule: Claude does not run pipelines. These are yours.

### Smoke Pilot (5–10 clusters)

```bash
conda activate k3d-cranium
python -m scripts.ingestion.d3.differentiate_b7_residual --worker \
  --mode meaning_resolution --stop-after 300 --row-concurrency 2
```

Verify:
- Retry envelope fires cleanly on any transient failure (look for log lines from the new `_LOG` instance).
- No `missing_sources` failures on common-vocabulary clusters (e.g., function words like "after", "because").
- `word_refs` populates on emitted meaning_star rows.

### 50-Cluster Pilot

```bash
python -m scripts.ingestion.d3.differentiate_b7_residual --worker \
  --mode meaning_resolution --stop-after 900 --row-concurrency 2
python -m scripts.ingestion.d3.differentiate_b7_residual --merge
```

Then run the standard audit and report the deltas versus the prior 301/3033:
- `merge_to_meaning_star` count
- `invalid_json` count (should drop significantly with the retry envelope)
- `language_variant_over_split` count (should drop with the reordered parser + preamble)
- `missing_sources` count
- `unidirectional_site_count` (informational; `pointed_by` synthesis is architecturally correct but not tracked by `REF_LIST_FIELDS` — separate audit-scope work)

## Known Non-Goal

`unidirectional_site_count = 927` will NOT move from this commit alone. `galaxy_audit.py` `REF_LIST_FIELDS` tracks `taxonomy_refs`, `grammar_refs`, `meta_refs`, `reality_refs`, `visual_refs`, `audio_refs`, `component_refs`, `composite_of`, plus `surface_forms.*.{word_ref,char_refs}`. `points_to` / `pointed_by` are not in that list. Fixing the unidirectional count is a separate audit-scope decision (add `points_to`/`pointed_by` to `REF_LIST_FIELDS` OR materialise the inverse edges as `meta_refs`).

## Escalation

If the retry envelope ever waits 5h and the run is still wedged, capture the `plan_limit_consumed` trace and ping Claude for a spec revision. Do not add a Python fallback.
