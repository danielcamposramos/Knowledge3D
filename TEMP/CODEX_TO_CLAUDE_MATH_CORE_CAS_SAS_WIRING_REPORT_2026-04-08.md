# Codex to Claude: Math Core CAS/SAS Wiring Report

**Date:** 2026-04-08 21:19:46 -0300
**Spec:** `TEMP/CODEX_MATH_CORE_CAS_SAS_WIRING_SPEC_2026-04-08.md`

## 1. Tier Assignment Implemented

File:
- `knowledge3d/cranium/bridges/tiered_rpn.py`

Implemented routing table:

| Opcode | Name | Tier |
|---|---|---:|
| `0x220-0x237` | existing CAS block (`poly_*`, `simplify`, `substitute`, `solve_*`, etc.) | 2 |
| `0x239` | `OP_CAS_HASH` | 1 |
| `0x23A` | `OP_SEMANTIC_RESOLVE` | 2 |
| `0x23B` | `OP_RULE_SELECT` | 2 |
| `0x23D` | `OP_SEMANTIC_EQUIV` | 2 |
| `0x238` | `OP_CANONICALIZE` | 3 |
| `0x23C` | `OP_CONTEXTUAL_REWRITE` | 3 |

Routing behavior:
- any program containing `0x238` or `0x23C` is selected as Tier-3
- mixed programs still take the highest tier
- public `TieredRPNEngine.select_tier()` now exposes that routing cleanly to the math-core allocation layer

Important live contract:
- Tier selection is Tier-3 as specified
- actual CAS/SAS execution currently stays on the sovereign modular kernel surface (`knowledge3d/cranium/kernels/modular_rpn_kernel.cu`)
- this is not a Python fallback; it is still GPU-hot-path execution
- `AdvancedRPNEngine` / extended Tier-3 PTX has **not** yet absorbed the CAS/SAS family

## 2. GPU Kernel Dispatch Status

File:
- `knowledge3d/cranium/kernels/modular_rpn_kernel.cu`

Now dispatched inside the GPU kernel:

| Opcode | Status | Notes |
|---|---|---|
| `OP_CANONICALIZE` | real inline | iterative traversal, simplification, commutative rebuild, hashcons |
| `OP_CAS_HASH` | real inline | hash slot lookup from CAS node fields |
| `OP_SEMANTIC_RESOLVE` | real inline | `g_sas_symbol_values[256]` lookup |
| `OP_RULE_SELECT` | real inline | bounded one-way pattern match over candidate pairs, bindings persisted for the running program |
| `OP_CONTEXTUAL_REWRITE` | real inline | template materialization from persisted bindings + canonicalization |
| `OP_SEMANTIC_EQUIV` | real inline | root-index equality on canonical forms |

Implementation note:
- I added shared per-program SAS binding state to the modular kernel so `OP_RULE_SELECT` and `OP_CONTEXTUAL_REWRITE` can work across opcode boundaries inside one execution.
- Current rule-table contract is bounded and explicit:
  - `candidate_base` is treated as the CAS-pool base index of contiguous pattern/template pairs
  - candidate `i` reads:
    - pattern root = `candidate_base + i*2`
    - replacement template root = `candidate_base + i*2 + 1`
- This is a real GPU contract, not a placeholder, but it is still narrower than a full Grammar-star-backed rule table.

## 3. Example Worker RPN Programs

Actual token strings exercised in tests:

GAME_2D-style rewrite lane:

```text
1 canonicalize 2 3 rule_select 4 contextual_rewrite
```

MATH-style semantic lane:

```text
1 canonicalize 32 semantic_resolve 2 semantic_equiv
```

Lighter Tier-2 semantic lookup lane:

```text
32 semantic_resolve 33 semantic_resolve semantic_equiv
```

Execution-level GPU rewrite proof used in `tests/test_swarm_cas_integration.py`:

```text
101 cas_push_sym
0 cas_push_const
10 2 cas_build
101 cas_push_sym
1 cas_push_sym
0 cas_push_const
10 2 cas_build
canonicalize
2 1 rule_select
contextual_rewrite
```

That program builds:
- pattern: `a + 0`
- replacement template: `a`
- subject: `x + 0`

Result:
- rewritten GPU result root = `4.0`
- expected simplification target reached entirely inside the modular kernel

## 4. Grep Proof: Zero Direct `launch_k3d_*` in Live Evaluation

Command:

```bash
rg -n "launch_k3d_(canonicalize|pattern_match|rule_apply|expr_build)" benchmarks tests
```

Result:
- no hits in `benchmarks/*.py`
- remaining hits are test-only:
  - `tests/test_arc_r0_surface.py`
  - `tests/test_cas_sovereignty.py`

So the live evaluation path is clean from direct Python CAS/SAS bridge orchestration.

## 5. Naming Corrections Applied

File:
- `benchmarks/arc_task_galaxy_seeder.py`

Applied:
- canonical star identity now uses:
  - `star_id = "spatial_grid_transform:{task_id}:pair{pair_idx}"`
  - `meaning_class = "spatial_grid_transform"`
  - `taxonomy_refs = ["spatial_grid_transform", "two_dimensional", "input_output_pair"]`
- benchmark provenance is stored only in metadata:
  - `benchmark_source = "arc_agi_2"`
  - `task_id`
  - `pair_idx`

This removes benchmark naming from canonical knowledge identity, per the foundational naming rule.

## 6. Second-Pass Rebuild Status

Important correction to the written spec:
- there is **no** dedicated `--second-pass-rebuild` CLI in `scripts/fundamental_ingest_pdfs.py`
- second pass is already part of the canonical ingest rebuild path inside `_rebuild_payload_from_stage()`

Current operational state:
- encyclopedia finalization is already live in the background as PID `400282`
- command is the canonical ingest invocation writing:
  - `.../payloads/payload.jsonl`
  - `.../summaries/ingest_report.json`
- because that process is still running, I did **not** launch any second-pass/manual rebuild command on top of it

## 7. Tests

Focused math-core/CAS/SAS suite:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q \
  tests/test_tiered_rpn.py \
  tests/test_tiered_rpn_cas_dispatch.py \
  tests/test_swarm_cas_integration.py \
  tests/test_cas_sovereignty.py \
  tests/test_arc_r0_surface.py \
  tests/test_sovereign_sas_surface.py
```

Result:
- `35 passed in 15.66s`

Broader benchmark/tablet regression after the router/kernel changes:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q \
  tests/test_tablet_boundary_benchmarks.py \
  tests/test_benchmarks.py \
  tests/test_arc3_agent.py
```

Result:
- `23 passed in 103.29s`

## 8. Honest Remaining Gap

The spec’s architectural intent is now respected in the hot path:
- Python no longer owns CAS/SAS reasoning in evaluation
- worker programs emit CAS/SAS opcodes
- GPU kernels execute those opcodes

The remaining gap is **which GPU executor owns Tier-3 CAS/SAS**:
- today: `TieredRPNEngine` selects Tier-3, but CAS/SAS execution is serviced by the sovereign modular kernel
- target future state: extended Tier-3 PTX / `AdvancedRPNEngine` absorbs the CAS/SAS family directly

So the path is now sovereign and truthful, but the final Tier-3 executor unification is still pending.
