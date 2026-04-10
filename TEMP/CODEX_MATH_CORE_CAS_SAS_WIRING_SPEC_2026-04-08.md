# Codex Direction: Math Core Tiering — CAS/SAS as Native RPN Tier-3 Ops

**Date:** 2026-04-08
**Authority:** docs/vocabulary/MATH_CORE_SPECIFICATION.md (tiered math core model)
              docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md (programs before opcodes)
              docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md (4-layer architecture)
**Depends on:** Sovereign benchmark migration (CODEX_ARC_SOVEREIGN_WIRING_SPEC) — complete ✅
**Does NOT touch:** protected ingest PID (already finished, see encyclopedia note below)

---

## Context — What CAS/SAS Was Designed to Enhance

CAS/SAS (opcodes 0x220–0x23D) was built to extend the **math core substrate** defined in
`docs/vocabulary/MATH_CORE_SPECIFICATION.md`. Specifically:

- CAS opcodes enhance Tier-3 (High Core / AdvancedRPNEngine + RPNMathCore) with symbolic
  algebraic manipulation: canonical form normalization, hashcons deduplication, semantic
  symbol resolution, pattern-match-replace rules, defeasible conflict resolution.
- SAS opcodes (0x238–0x23D) are the semantic layer on top: they operate on CAS STAR nodes
  using meaning (Galaxy-derived symbol values) rather than raw floats.

**The problem with the current wiring:**
The CAS/SAS bridge methods (`launch_k3d_canonicalize`, `launch_k3d_pattern_match`, etc.)
were being called FROM PYTHON, which made Python the orchestrator of symbolic reasoning.
Now that benchmarks route through WINE → execute_task → composed head → swarm workers,
CAS/SAS must be called FROM INSIDE swarm worker RPN programs, dispatched by TieredRPNEngine,
not from Python.

**The math core hierarchy (from MATH_CORE_SPECIFICATION.md §2.1, §3.1):**
```
Tier-1 (Simple Core / worker-worker) → LightweightRPNEngine
  - Ultra-fast: basic arithmetic, small vector ops, hash lookups
  - 66% of core allocation
Tier-2 (Mid Core / worker) → ModularRPNEngine
  - Moderate: matrix ops, clustering, symbol table lookups, poly eval
  - 22% of core allocation
Tier-3 (High Core / master) → AdvancedRPNEngine + RPNMathCore
  - Heavy: TRM coupling, symbolic normalization, rule application, defeasible logic
  - 11% of core allocation (reserve for when it matters)
```

**TieredRPNEngine** (`bridges/tiered_rpn.py`) is the orchestrator — it routes programs to
the right tier based on the opcodes they contain. CAS/SAS opcodes must be registered here.

---

## Step 1 — Register CAS/SAS Opcode Ranges in TieredRPNEngine

File: `knowledge3d/cranium/bridges/tiered_rpn.py` (or wherever `TieredRPNEngine` lives)

Add the CAS/SAS opcode ranges to the tier-routing table. The principle:
- Cheap lookups and arithmetic → Tier-1
- Moderate symbolic ops → Tier-2
- Structural normalization and rule application → Tier-3 (they touch the CAS pool, hashcons)

```python
# CAS/SAS opcode tier assignments
# Existing CAS block (0x220–0x237) — these are already Tier-2 (poly build/eval/mul/simplify)
_CAS_TIER2_RANGE = range(0x220, 0x238)   # OP_POLY_BUILD through existing CAS ops

# SAS block (0x238–0x23D) — tiered by cost:
_SAS_TIER1_OPS = {0x239}                 # OP_CAS_HASH — pure hash lookup, very cheap
_SAS_TIER2_OPS = {0x23A, 0x23B, 0x23D}  # OP_SEMANTIC_RESOLVE, OP_RULE_SELECT,
                                          # OP_SEMANTIC_EQUIV — symbol table + pattern check
_SAS_TIER3_OPS = {0x238, 0x23C}          # OP_CANONICALIZE (4-pass + hashcons),
                                          # OP_CONTEXTUAL_REWRITE (rule apply + defeasible)
```

In `TieredRPNEngine._select_tier(program_opcodes)`:

```python
def _select_tier(self, program_opcodes: set[int]) -> int:
    """Return 1, 2, or 3 — highest tier required by any opcode in the program."""
    if program_opcodes & _SAS_TIER3_OPS:
        return 3
    if (program_opcodes & _SAS_TIER2_OPS) or any(op in _CAS_TIER2_RANGE for op in program_opcodes):
        return 2
    if program_opcodes & _SAS_TIER1_OPS:
        return 1
    # existing non-CAS/SAS tier selection logic...
    return self._select_tier_base(program_opcodes)
```

Result: when a swarm worker's RPN program contains `OP_CANONICALIZE` (0x238), TieredRPNEngine
automatically routes it to the Tier-3 (AdvancedRPNEngine) path that has access to the CAS pool
and sas_module_linked.cu kernel. The swarm worker does not need to know which tier handles it.

---

## Step 2 — CAS/SAS Opcode Dispatch in the Modular RPN Kernel

File: `knowledge3d/cranium/kernels/modular_rpn_kernel.cu` (the PTX dispatch table)

The CAS/SAS opcodes must be dispatched in the GPU-side opcode switch just like any other
opcode. Currently the kernel handles 0x00–0xFF and some extended ranges. Add cases for
0x238–0x23D that call the sas_module_linked.cu kernels via device function pointers or
inline includes.

**Dispatch pattern to follow** (same style as existing opcode cases):

```c
case OP_CAS_HASH: {          // 0x239 — Tier-1 fast path
    uint32_t node_idx = (uint32_t)pop_stack(state);
    push_stack(state, (float)hashcons_slot(g_cas_pool + node_idx));
    break;
}

case OP_SEMANTIC_RESOLVE: {  // 0x23A — Tier-2: symbol table lookup
    uint32_t sym_id = (uint32_t)pop_stack(state);
    float val = (sym_id < 256) ? g_symbol_table[sym_id] : 0.0f;
    push_stack(state, val);
    break;
}

case OP_RULE_SELECT: {       // 0x23B — Tier-2: pattern match over Grammar candidates
    uint32_t input_root = (uint32_t)pop_stack(state);
    uint32_t candidate_base = (uint32_t)pop_stack(state);
    uint32_t n_candidates = (uint32_t)pop_stack(state);
    uint32_t matched = k3d_pattern_match_inline(
        g_cas_pool, input_root, candidate_base, n_candidates,
        state->binding_table
    );
    push_stack(state, (float)matched);
    break;
}

case OP_CONTEXTUAL_REWRITE: { // 0x23C — Tier-3: apply rule + defeasible resolution
    uint32_t rule_output_root = (uint32_t)pop_stack(state);
    uint32_t rule_strength    = (uint32_t)pop_stack(state);
    uint32_t result = k3d_rule_apply_inline(
        g_cas_pool, rule_output_root, state->binding_table,
        rule_strength, g_defeasible_ctx
    );
    push_stack(state, (float)result);
    break;
}

case OP_SEMANTIC_EQUIV: {    // 0x23D — Tier-2: canonical form equality
    uint32_t root_a = (uint32_t)pop_stack(state);
    uint32_t root_b = (uint32_t)pop_stack(state);
    // Both roots already canonicalized; compare pool indices
    push_stack(state, (float)(root_a == root_b ? 1 : 0));
    break;
}

case OP_CANONICALIZE: {      // 0x238 — Tier-3: 4-pass normalization + hashcons
    uint32_t root = (uint32_t)pop_stack(state);
    uint32_t canon = k3d_canonicalize_inline(g_cas_pool, root, g_hashcons_table);
    push_stack(state, (float)canon);
    break;
}
```

`_inline` variants are device functions in `sas_module_linked.cu` compiled into the same PTX
module as `modular_rpn_kernel.cu`. This gives them access to `g_cas_pool`, `g_hashcons_table`,
`g_symbol_table`, and `g_defeasible_ctx` without module boundary issues.

---

## Step 3 — Swarm Worker RPN Programs Use CAS/SAS Internally

The nine-chain swarm workers compose RPN programs that call CAS/SAS opcodes. Python NEVER
calls bridge methods for CAS/SAS directly in the evaluation path — only the RPN kernel does.

**Example: `gre_arc_reasoner` worker program for GAME_2D tasks**

The worker receives (via swarm dispatch): an encoded grid on the stack + the Grammar candidate
base address + candidate count.

```
# Worker RPN program — gre_arc_reasoner internal reasoning
PUSH n_candidates         # How many training-pair Grammar stars to check
PUSH candidate_base       # GPU address of seeded Grammar stars' input CAS roots
PUSH test_input_root      # CAS STAR node root for the test input grid
OP_CANONICALIZE           # 0x238 — normalize test input (Tier-3, ~10µs)
OP_RULE_SELECT            # 0x23B — find matching training-pair star (Tier-2)
OP_CONTEXTUAL_REWRITE     # 0x23C — apply the matched rule's output transform (Tier-3)
# Stack now holds the output grid's CAS root — swarm returns this as its candidate
```

TieredRPNEngine sees `{0x238, 0x23B, 0x23C}` → selects Tier-3 → dispatches to
AdvancedRPNEngine → GPU kernel handles all three opcodes in sequence, no Python involvement.

**Example: `gre_geometry_router` worker for geometric reasoning**

```
PUSH test_grid_root
OP_CANONICALIZE           # 0x238 — canonical form (commutative-sorted, hashconsed)
OP_CAS_HASH               # 0x239 — cheap lookup: is this hash in hashcons? (Tier-1)
BRANCH_IF_ZERO new_form   # if not found, it's a truly new grid form
OP_SEMANTIC_RESOLVE       # 0x23A — resolve to known spatial_grid_* meaning star
OP_RULE_SELECT            # 0x23B — find the geometric transform rule that applies
OP_CONTEXTUAL_REWRITE     # 0x23C — apply it
```

This mixes Tier-1 (`OP_CAS_HASH`) and Tier-3 (`OP_CANONICALIZE`, `OP_CONTEXTUAL_REWRITE`).
TieredRPNEngine sees any Tier-3 opcode → upgrades full program to Tier-3.

---

## Step 4 — Remove Python CAS/SAS Call Sites from Evaluation Path

After the sovereign migration, there should be zero direct calls to:
- `bridge.launch_k3d_canonicalize()`
- `bridge.launch_k3d_pattern_match()`
- `bridge.launch_k3d_rule_apply()`
- `bridge.launch_k3d_expr_build()`

...in ANY live benchmark evaluation path. These are legal in:
- Test files (`tests/`) — to verify the kernels work correctly
- The sas bootstrap scripts (`sas_grammar_bootstrap.py`, `sas_symbol_bootstrap.py`) — ingestion path only
- `arc_task_galaxy_seeder.py` — ingestion path only

Grep for these call sites and confirm they are only in ingestion/test paths. If any appear
in `benchmarks/*.py` or the live evaluation path, replace them with RPN programs that emit
the equivalent CAS/SAS opcodes (see Step 3 examples above).

---

## Step 5 — Knowledge Naming (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §0)

The CAS/SAS work introduced some ARC-named star IDs in `arc_task_galaxy_seeder.py`:
```python
star_id=f"arc_rule:{task_id}:pair{pair_idx}"
taxonomy_refs=["arc", "grid_transform", str(task_id)]
```

Per the FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §0 "Critical Naming Convention":
> No benchmark names in knowledge.

Update `arc_task_galaxy_seeder.py`:
```python
# WRONG (benchmark-named):
star_id=f"arc_rule:{task_id}:pair{pair_idx}"
taxonomy_refs=["arc", "grid_transform", str(task_id)]

# CORRECT (meaning-named):
star_id=f"spatial_grid_transform:{task_id}:pair{pair_idx}"
taxonomy_refs=["spatial_grid_transform", "two_dimensional", "input_output_pair"]
# benchmark provenance goes in metadata, not canonical identity:
metadata={"benchmark_source": "arc_agi_2", "task_id": task_id, "pair_idx": pair_idx}
```

Same correction for any other ARC-named knowledge artifacts introduced during R0/R1.

---

## Step 6 — Encyclopedia Second-Pass Rebuild

**Important:** PID 101379 (encyclopedia ingest) is finished. All PDFs are `staged_complete`.
There is no final payload yet — only `payload_second_pass_preview.jsonl`.

After all math core wiring above is verified:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python scripts/fundamental_ingest_pdfs.py \
  --second-pass-rebuild \
  --input /K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/stages/manifest.json \
  --output /K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/payloads/payload.jsonl
```

(Use whatever the actual second-pass rebuild command is from the ingest script's `--help` output.)

This produces the final `payload.jsonl` from the staged + previewed content.
After that: resident ingest → knowledge lands in Grammar/Reality/Math galaxies →
swarm workers find richer knowledge when doing CAS/SAS pattern matching.

---

## Tests

```bash
# Tier assignment verification
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_tiered_rpn_cas_dispatch.py

# Swarm worker RPN programs using CAS/SAS opcodes
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_swarm_cas_integration.py

# Confirm no direct CAS bridge calls in evaluation path (sovereignty test)
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_cas_sovereignty.py
```

`test_tiered_rpn_cas_dispatch.py` must verify:
1. Program containing only `OP_CAS_HASH (0x239)` → TieredRPNEngine selects Tier-1
2. Program containing `OP_SEMANTIC_RESOLVE (0x23A)` → selects Tier-2
3. Program containing `OP_CANONICALIZE (0x238)` → selects Tier-3
4. Program containing `OP_CONTEXTUAL_REWRITE (0x23C)` → selects Tier-3
5. Mixed program (0x239 + 0x23C) → selects Tier-3 (highest wins)

`test_cas_sovereignty.py` must verify:
1. `import benchmarks.arc2_local_runner; assert 'launch_k3d_canonicalize' not in source`
2. Same for `launch_k3d_pattern_match`, `launch_k3d_rule_apply`, `launch_k3d_expr_build`
3. These calls ARE found in `benchmarks/arc_task_galaxy_seeder.py` (ingestion path — correct)
4. These calls ARE found in `tests/test_sovereign_cas_benchmark_simple.py` (tests — correct)

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_MATH_CORE_CAS_SAS_WIRING_REPORT_2026-04-08.md` with:

1. TieredRPNEngine tier assignment table (opcode → tier) as implemented
2. Which CAS/SAS opcodes are now dispatched inside the GPU kernel (which ones are inline)
3. Example swarm worker RPN programs for GAME_2D and MATH tasks (actual RPN token strings)
4. grep proof: zero `launch_k3d_*` calls in live evaluation path
5. Star ID naming corrections applied (arc_rule → spatial_grid_transform)
6. Second-pass rebuild status
7. All tests passing: command and count

---

## What NOT to Do

- Do NOT add CAS/SAS calls to Python benchmark evaluation code — they go in RPN programs
- Do NOT combine Tier-1/2/3 into one tier to simplify — the tiering IS the performance model
- Do NOT cap math cores at 18 — MATH_CORE_SPECIFICATION.md §2.3 says scale to SM count
- Do NOT rename or merge `LightweightRPNEngine`, `ModularRPNEngine`, `AdvancedRPNEngine`
  — they are the three tiers, they must remain separate for routing to work
- Do NOT promote CAS/SAS opcodes to Tier-1 to make them "faster" — canonicalization
  requires pool access and hashcons; doing it in Tier-1 would corrupt shared state
- Do NOT run the encyclopedia second-pass before math core tests are green
