# CLAUDE → CODEX — Phase 6.B.2: Live Integration Repair — 2026-04-11

## The three distinct drops

All three of the probe failures trace to specific lines I already inspected. Do not pattern-match against earlier assumptions — these are concrete and independent.

### Drop 1 — The exemption flag is over-zealous (role/answer/refs zeroed)

[sovereign_hot_path.py:1282-1293](../knowledge3d/knowledgeverse/sovereign_hot_path.py#L1282):

```python
if sovereign_route_exempt:
    layer_id = 0
    selection_role = "unknown"
    answer_eligible = False
    route_policy = {}
    explicit_role_refs_present = False
    routing_refs = {
        "router_refs": [],
        "executor_refs": [],
        "validator_refs": [],
        "anti_pattern_refs": [],
    }
```

This is wrong. Phase 6.B.1 scoped `_validate_route_link_coverage` to skip exempt stars — that was the correct fix. This block is a *second*, more aggressive "fix" from an earlier attempt: when exemption is true, it strips the star's declared role, answer-eligibility, and all ref sets before the star dict is built. The result is exactly what the reprobe shows: `role=0 answer=0` on `math_operator_addition`, `concept_digit_two`, `rpn_program_addition`, `grammar_binary_op_infix`.

`sovereign_route_exempt` means **"this star is not bound by the legacy router/executor/validator triple invariant"** — it does NOT mean "this star has no role or semantics." Meaning-centric stars still have roles (executor, answer, router per schema §2.3). They dispatch via symlinks, not via triple-closure.

**Fix:** delete lines 1282-1293 entirely. The exemption branch in the error block at line 1294 (`route_active = (not sovereign_route_exempt) and (...)`) already protects exempt stars from missing-field errors. The second stripping block is redundant and actively destructive.

**Audit after deletion:** if removing the block causes `_validate_route_link_coverage` to re-explode on exempt stars, that means 6.B.1's skip guard is incomplete — fix the skip guard, do NOT re-introduce the stripping.

### Drop 2 — Native program fields never reach the star dict

[sovereign_hot_path.py:3262-3265](../knowledge3d/knowledgeverse/sovereign_hot_path.py#L3262) reads `meta_rule_addr`, `program_flags`, `program_length`, `program_opcode_count` off `row` inside `_pack_catalog_input_row`. The `row` there is the **star dict** produced by `_translate_catalog_row`.

Look at the star dict construction at [sovereign_hot_path.py:1353-1385](../knowledge3d/knowledgeverse/sovereign_hot_path.py#L1353). None of those four keys are ever written into the star dict. The crafter puts them on the *source* dict via `_CraftedEntry.to_row()`'s `row.update(dict(self.extra_top_level))`, but `_translate_catalog_row` constructs a brand new star dict and does not copy them over. `_pack_catalog_input_row` then reads them off the star dict with a default of 0 and writes 0 into the 400-byte GPU record.

**Fix:** in `_translate_catalog_row`, add explicit propagation from source/metadata to the star dict for the four native program fields. Put them right next to the existing `star_hash` / `embedding` lines:

```python
"meta_rule_addr": int(source.get("meta_rule_addr", metadata.get("meta_rule_addr", 0)) or 0),
"program_flags": int(source.get("program_flags", metadata.get("program_flags", 0)) or 0),
"program_length": int(source.get("program_length", metadata.get("program_length", 0)) or 0),
"program_opcode_count": int(source.get("program_opcode_count", metadata.get("program_opcode_count", 0)) or 0),
```

**Crafter side:** confirm `_CraftedEntry.to_row()` actually surfaces these at the top level of the row that feeds into the catalog. It currently does via `row.update(dict(self.extra_top_level))` at [star_crafter.py:135](../knowledge3d/ingestion/star_crafter.py#L135), but there is one suspicious step right after: `wrapped = wrap_galaxy_entry_with_meaning_star(row, self.meaning_star)`. If that wrapper returns a fresh dict and drops keys it doesn't know about, the program fields vanish before they ever reach the catalog. Print the wrapped dict keys for one crafter entry in the Phase 6.B.2 probe to confirm whether the wrapper preserves them. If it doesn't, either re-apply `extra_top_level` *after* the wrap, or teach `wrap_galaxy_entry_with_meaning_star` to preserve the four native program fields.

### Drop 3 — Crafter reinvented its own embedder; real answer is the Matryoshka stack

[star_crafter.py:46](../knowledge3d/ingestion/star_crafter.py#L46) defines `EMBEDDING_DIMS = 32` and [star_crafter.py:219-242](../knowledge3d/ingestion/star_crafter.py#L219) computes embeddings via FNV hash buckets plus manual term boosts. That is a toy substrate that has nothing to do with how the rest of the system embeds text. That is why `"plus"` lands `math_operator_addition` at rank 1134 — the crafter's space is cosine-orthogonal to whatever the live query path uses.

**The real fix is deeper than "use the 64D embedder that already exists."** Daniel's correction:

> why is that embedding fixed when we have a matryoshka rpn embedding engine that can shrink and expand on need?

He is right. Per [MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §7](../docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md#7-matryoshka-embeddings-search-lod), every meaning-centric star carries a **Matryoshka stack** — 64/128/512/2048 tier prefixes of a single projection — and consumers pick the tier they need for the search granularity (room → shelf → star → fingerprint). The [MatryoshkaProjectionBridge](../knowledge3d/cranium/bridges/matryoshka_bridge.py) and [MatryoshkaTRM](../knowledge3d/cranium/matryoshka_trm.py) already exist. No fixed dim is canonical — the max-dim projection is canonical, every tier is a prefix of it.

What went wrong on our side:

- Galaxy VRAM table's `EMBEDDING_DIMS = 64` at [galaxy_vram_table.py:13](../knowledge3d/knowledgeverse/galaxy_vram_table.py#L13) is **the coarse tier_64 slot** (spec §7: "coarse search key, room-level navigation"). That is legitimately the on-GPU cheap navigation tier. It is NOT a canonical fixed dim — it is the LOD tier for the cheapest cosine pass.
- The crafter invented EMBEDDING_DIMS=32 as a third, unrelated space and poisoned the 64D slot by padding half of it with zeros.
- Neither path goes through the Matryoshka bridge, so the higher tiers (128/512/2048) are never written onto the star, which breaks any finer search later.

**Fix shape:**

1. **Canonical projection source = MatryoshkaProjectionBridge.** The crafter calls the Matryoshka bridge to project the star's text (name + domain + meaning_rpn + meaning_description) into the **full max-dim vector**. Ingestion path — sentence-transformer + Matryoshka projection is allowed per schema §5.1.

2. **Stars carry the Matryoshka stack.** The crafter row stores all four tier prefixes (tier_64, tier_128, tier_512, tier_2048) as four fields on the row, sourced by prefix-truncation of the single max-dim projection. No re-projection, no re-hashing — the Matryoshka property is that tier_N is literally the first N elements of the full vector, normalized.

3. **Galaxy VRAM table's 64D slot is fed by tier_64.** The sovereign path reads `tier_64` from the star dict (or falls through to projecting it on demand) and writes it into the coarse GPU slot. The higher tiers live either in a secondary VRAM region or in House persistence — that decision is architectural and not blocking for 6.B.2; for now, land tier_64 into the live GPU slot and stash the full stack on the host-side star dict so 6.C/later can expand.

4. **Query-side must use the same projection.** The cosine routing probe embeds `"plus"` through the **same Matryoshka bridge**, truncates to tier_64, and cosines against the GPU table. Same projection on both sides is non-negotiable — otherwise the spaces don't match and cosine is meaningless.

5. **Investigate `_entry_embedding64` at [knowledgeverse.py:4148](../knowledge3d/knowledgeverse/knowledgeverse.py#L4148) and `_precomputed_entry_embedding64_raw` at [knowledgeverse.py:3977](../knowledge3d/knowledgeverse/knowledgeverse.py#L3977).** If either already routes through the Matryoshka bridge, the crafter just needs to provide rich enough text and omit `embedding` from the row so the sovereign translator calls the canonical path. If they route through raw sentence-transformers (bypassing Matryoshka), that is a pre-existing bug and the fix is to point them at the Matryoshka bridge — which benefits Drop 3 and everything else at once.

6. **Delete `build_text_embedding`, `cosine_similarity`, and `EMBEDDING_DIMS = 32`** from star_crafter.py. Toy substrate. A future reader will be confused about which embedder is canonical.

**Sanity anchor:** the principle is "one projection, many tiers." The crafter does not own an embedder. The Matryoshka bridge owns the projection. Stars are decorated with prefix views of that projection. Consumers pick their tier. That is the spec, and it's in the repo waiting to be wired.

**Reprobe expectation after Drop 3:** `"plus"` → `math_operator_addition` rank ≤ 10 at tier_64, rank ≤ 3 at tier_128+. If the tier_64 rank is still bad but higher tiers recover, that tells us tier_64 is too coarse for operator-vs-digit distinctions and the probe should step up to tier_128 by default.

### Drop 4 — `word_digit_two_en` missing from the live host table (diagnose before fixing)

Unlike the first three, I cannot point at a single line. Candidates in descending likelihood:

1. The legacy `foundational_galaxy_builder.build_foundational_galaxy_table()` path dedups by `id` against an earlier pass that has a star with the same key, and the crafter entry loses.
2. A galaxy filter in the sovereign build path drops Word-galaxy entries with `star_type = STAR_TYPE_CHARACTER = 2` before materialization.
3. The embedding-nonzero guard at line 1221 is silently dropping `word_digit_two_en` because its hash-bucket vector happens to collapse to zero after padding (less likely given the term profile, but still possible while Drop 3 is outstanding).
4. The `_CraftedEntry.to_row()` wrapper (`wrap_galaxy_entry_with_meaning_star`) is collapsing surface-form stars into their parent concept star.

**Diagnostic before fixing:** add a one-shot trace at three points and run full boot once:

- after `crafter.craft_all()` — assert `word_digit_two_en` is present in the returned list and log its row keys
- after `_translate_catalog_entries` — log whether the star with id `word_digit_two_en` is in `stars`, and if not, log any `metadata_errors` entry that mentions it
- after `_build_stars_sovereign` — log whether it reached the final host_stars list

Whichever point loses it is the fix site. Report the verdict before patching.

---

## What the fix order is

Fix strictly in this order, with the probe re-run after each step to confirm the signal moves in the expected direction. Do **not** bundle fixes — bundling is how drops get masked.

1. **Drop 1 — delete the stripping block at sovereign_hot_path.py:1282-1293.**
   Reprobe expectation: `math_operator_addition.selection_role_id = ROLE_EXECUTOR`, `answer_eligible = 1`. Still `meta_rule_addr=0` (Drop 2 still pending). Still bad cosine ranks (Drop 3 still pending).

2. **Drop 2 — propagate native program fields through `_translate_catalog_row`** and verify `wrap_galaxy_entry_with_meaning_star` preserves them on the crafter side.
   Reprobe expectation: `math_operator_addition.meta_rule_addr > 0`, `program_length > 0`, `program_opcode_count > 0`; `rpn_program_addition` mirrors the same program offset.

3. **Drop 3 — remove crafter embeddings, use the canonical embedder.**
   Reprobe expectation: `"plus"` → `math_operator_addition` rank ≤ 10 (ideally ≤ 3); `"two"` → `concept_digit_two` rank ≤ 10; `"3"` → `concept_digit_three` rank ≤ 10. If top-3 isn't hit but ranks drop from 1000+ to under 50, we are on the right embedder and the remaining rank gap is acceptable for 6.C (TRM y_new navigation will do better than raw surface-string queries).

4. **Drop 4 — diagnostic first, then fix the identified drop site.**
   Reprobe expectation: `word_digit_two_en` present in the live host table; all 10 English number word stars present; all 5 operator word and 5 operator symbol stars present.

## Validation to land with 6.B.2

Add `tests/test_phase6b2_live_integration.py` under `K3D_PYTEST_PROBE_CUDA=1`:

1. Boot the full Knowledgeverse (no shortcuts — the integration path is what failed).
2. Look up `math_operator_addition` by id in the host star table. Assert `selection_role == "executor"`, `answer_eligible == True`, `meta_rule_addr > 0`, `program_length > 0`, `program_opcode_count > 0`.
3. Look up `concept_digit_two` by id. Assert `selection_role == "answer"`, `answer_eligible == True`, `meta_rule_addr == 0` (digits don't execute).
4. Look up `rpn_program_addition` by id. Assert `meta_rule_addr` equals the offset for addition in the sovereign program table.
5. Look up `word_digit_two_en` by id. Assert present.
6. Cosine routing: embed `"plus"` via `Knowledgeverse._entry_embedding64({"name": "plus", ...})` — same embedder as live query path — cosine against the live 41,134-star table, assert `math_operator_addition` appears in top 10. Same for `"two"`/`concept_digit_two` and `"3"`/`concept_digit_three`.
7. Sovereignty grep: no `numpy|cupy|scipy|sympy` introduced into any touched hot-path file. Ingestion-side crafter remains free to use whatever it needs.
8. `tests/test_phase6b_star_crafter.py` still 7/7. `tests/test_phase6b1_route_validator_scoping.py` still passes. Phase 1–5 regression batch still green.

## Non-goals

- Do **not** start 6.C. HANDLING_QUERY wiring, operand packing, and end-to-end `"2+3?"` → `"5"` are blocked until all four drops are closed.
- Do **not** add fallbacks to mask any drop. If a fix fails, diagnose why.
- Do **not** reintroduce the stripping block with a narrower scope. The exemption flag's entire purpose is to *preserve* meaning-star semantics while opting out of the legacy triple check. Stripping is the opposite of that.
- Do **not** patch the crafter to emit its own sentence-transformer call until you confirm path (1) in Drop 3 is infeasible. One canonical embedder is better than two.

## One principle to hold

Everything in this phase is a *scoping correction*: the exemption must scope validation, the native program fields must scope into the star dict, the embedder must scope to one canonical path, and the drop-site for `word_digit_two_en` must be scoped by trace before it gets patched. No new code paths. No new fallbacks. Just making the crafter's declared state actually survive the translation and materialization pipeline.

Report back per step. I want the signal moving fix-by-fix, not a single bundled "everything passes now" message — bundling is how we lost Drop 4 in the first place.
