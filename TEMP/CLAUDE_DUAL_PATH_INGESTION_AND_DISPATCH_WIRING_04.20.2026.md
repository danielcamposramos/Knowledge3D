# Dual-Path Spec — Layer 0 Ingestion + Codec Dispatcher Wiring
**Date:** 2026-04-20
**Author:** Claude (Architecture Partner, Claude-pilot session)
**Session context:** Codex limit-locked; Claude piloting per [project_claude_runs_during_codex_limit.md](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/project_claude_runs_during_codex_limit.md)

---

## §1 Why this spec exists

Post-audit findings (2026-04-20):

1. **73 new opcodes minted** in [rpn_opcodes.py](../knowledge3d/cranium/ptx_runtime/rpn_opcodes.py) and tokenized in [modular_rpn_engine.py](../knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py) OPCODES dict, **but unreachable by TRM** because the codec gate at `modular_rpn_engine.py:659` is a hardcoded 18-token frozenset, not a range check. New 0x280–0x2DF tokens fall through to generic RPN evaluation which has no kernel for them.
2. **ARC3 bridge PTX compiled clean** ([arc3_screen_bridge.ptx](../knowledge3d/cranium/codecs/kernels/arc3_screen_bridge.ptx), 5 kernel entries, zero spills) but **no Python launcher exists**. Precedent: [ternary_codec_ops.py](../knowledge3d/cranium/codecs/ternary_codec_ops.py) loads codec PTX via `loader.load_module_from_file()` + `loader.get_function()`.
3. **Tablet WINE is actually sovereign** — pure envelope construction. All offline benchmark harnesses (MMLU, GSM8K, Math, LHE, ARC-2) already route through it.
4. **Real offline-benchmark blocker: Layer 0.** Drawing primitives stranded in `galaxy_pending/` with non-canonical IDs (`PRIM_LINE` vs `drawing_primitive_line`). Daemon bootstrap hangs on House load. Fix spec exists: [CODEX_LAYER0_SEED_AND_PARALLEL_INGEST_04.18.2026.md](CODEX_LAYER0_SEED_AND_PARALLEL_INGEST_04.18.2026.md).

The two paths are independent — offline benchmarks do **not** need 0x280+ opcodes; the new opcodes do not need Layer 0 fixed. They can run in parallel.

---

## §2 Path A — Layer 0 ingestion fix → offline benchmark suite

**Authoritative spec:** [CODEX_LAYER0_SEED_AND_PARALLEL_INGEST_04.18.2026.md](CODEX_LAYER0_SEED_AND_PARALLEL_INGEST_04.18.2026.md). This document re-scopes Path A to the *minimum* work needed to unblock offline benchmarks (the parallel-ingest / OCR sidecar work is deferred until after first green bench).

### §2.1 Step A1 — canonicalize Drawing primitive IDs

**File:** [knowledge3d/ingestion/atomic/drawing_grammar_builder.py](../knowledge3d/ingestion/atomic/drawing_grammar_builder.py)

**Action:** Replace every `"PRIM_LINE"`, `"PRIM_ARC"`, `"PRIM_QUAD"`, `"PRIM_CUBIC"`, `"PRIM_CIRCLE"`, `"PRIM_RECT"`, `"PRIM_TRI"` literal with the canonical form `canonical_drawing_primitive_id("<name>")` which resolves to `drawing_primitive_line`, `drawing_primitive_arc`, etc.

Source of truth: [docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md §7](../docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md).

**Acceptance:**
```bash
cd /K3D/GitHub/Knowledge3D
grep -n "PRIM_LINE\|PRIM_ARC\|PRIM_QUAD\|PRIM_CUBIC\|PRIM_CIRCLE\|PRIM_RECT\|PRIM_TRI" \
  knowledge3d/ingestion/atomic/drawing_grammar_builder.py
# expected: zero hits
```

### §2.2 Step A2 — extend canonical registry seed

**File:** [scripts/ingest_canonical_to_qdrant.py](../scripts/ingest_canonical_to_qdrant.py)

**Action:**
1. Extend the drawing primitive seed list from 3 → all 7 (line, arc, quad, cubic, circle, rect, tri) using the canonical names from §2.1.
2. Add the 11 `meaning_class` vocabulary items (see [CANONICAL_REGISTRY_SPECIFICATION.md §7](../docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md)).
3. Add the 12 `symlink_kind` field-path entries.

**Acceptance:**
```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium env PYTHONPATH=$(pwd) \
  python -c "
from knowledge3d.ingestion.canonical_lookup import CanonicalLookup
lookup = CanonicalLookup()
for name in ['line','arc','quad','cubic','circle','rect','tri']:
    sid = lookup.find_star_id('drawing_primitive', name)
    assert sid == f'drawing_primitive_{name}', f'{name} → {sid}'
print('Layer 0 registry seed OK')
"
```

### §2.3 Step A3 — promote `galaxy_pending/` Drawing stars

Once IDs are canonical, run the existing promotion script:
```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium env PYTHONPATH=$(pwd) \
  python scripts/promote_pending_galaxies.py --galaxy drawing
```

**Acceptance:** `galaxy_pending/drawing/` is emptied; entries appear in the live House JSONL (`/K3D/Knowledge3D.local/galaxies/Drawing_enriched.jsonl`).

### §2.4 Step A4 — offline benchmark suite run

```bash
export CUDA_VISIBLE_DEVICES=0
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium env PYTHONPATH=$(pwd) \
  python scripts/run_diagnostic_slices.py --benchmarks mmlu,gsm8k,math,lhe,arc2 --tag 2026-04-20_dual_path
```

**Acceptance (all must be logged, even regressions — paper evidence):**
- MMLU: some value in 10–20/50 range. Variance OK — just need a recorded pass with Layer 0 green.
- GSM8K: 2/10 floor (per CODEX.md current state).
- Math: 20/20 pinned.
- LHE: 6/10 pinned.
- ARC-2: 10/10 pinned.

Output written to `data/benchmark_runs/2026-04-20_dual_path/`.

---

## §3 Path B — Codec dispatcher extension (ARC3 subset, honest scope)

**Scope today:** Wire only the opcodes that have PTX kernels actually compiled. That means ARC3 (0x2A0–0x2A9) only. Texture Forge (0x280–0x28F), Image→3D (0x290–0x29F), Memory-as-Image (0x2B0–0x2B5), MVCIC extensions (0x2C0–0x2C4), and Document Galaxy (0x2D0–0x2DB) **are deferred** until their kernels exist — Codex's follow-up lane.

This is partnership honesty: reservations hold; implementation tracks actual kernels, not minted constants.

### §3.1 Step B1 — extend `CODEC_TOKENS` with ARC3 tokens

**File:** [knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py](../knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py) at line ~659.

**Action:** Add these 10 string tokens to the `CODEC_TOKENS` frozenset:
```python
"arc3_frame_decode", "arc3_palette_set", "arc3_frame_to_dotmap",
"arc3_project_to_screen", "arc3_click_invert", "arc3_action_emit",
"arc3_replay_step", "arc3_diff_highlight", "arc3_lives_hud",
"arc3_game_id_bind",
```

**Do not** add Texture Forge / Document Galaxy / etc. tokens yet. When a kernel lands, the opcode's token is added in the same commit.

### §3.2 Step B2 — `Arc3ScreenBridge` class

**File:** NEW, [knowledge3d/cranium/bridges/arc3_screen_bridge.py](../knowledge3d/cranium/bridges/arc3_screen_bridge.py)

**Pattern:** mirror [ternary_codec_ops.py](../knowledge3d/cranium/codecs/ternary_codec_ops.py).

**Requirements:**
- Zero numpy / cupy / scipy / sympy imports (per [feedback_no_numpy_no_bulk_libraries_sovereign_only.md](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_no_numpy_no_bulk_libraries_sovereign_only.md))
- Loads `knowledge3d/cranium/codecs/kernels/arc3_screen_bridge.ptx` via `loader.load_module_from_file`
- Binds the 5 kernel entries (`arc3_frame_decode_kernel`, `arc3_click_invert_kernel`, `arc3_action_emit_kernel`, `arc3_diff_highlight_kernel`, `arc3_lives_hud_kernel`)
- Palette upload via `cuMemcpyToSymbol(c_arc3_palette, ...)` — constant memory symbol at top of .cu file
- Exposes methods: `decode_frame(frame_idx_dev, W, H) → rgba_dev`, `invert_click(sx, sy, rect, grid) → (gx, gy)`, `emit_action(action_id, gx, gy) → record`, `diff_highlight(frame_a, frame_b, W, H, hi_rgba) → overlay`, `lives_hud(Hw, Hh, lives_rem, lives_tot, moves_rem, moves_tot) → rgba`
- All device buffers allocated via existing `knowledge3d.cranium.kernels` utilities (not raw cupy/numpy)

### §3.3 Step B3 — register ARC3 bridge in `_CODEC_TOKEN_MAP`

**File:** [knowledge3d/cranium/bridges/tiered_rpn.py](../knowledge3d/cranium/bridges/tiered_rpn.py) at line ~23 (_CODEC_TOKEN_MAP) + the `execute_codec` dispatcher (line ~305).

**Action:** For each of the 10 arc3_* string tokens, add a dispatch case that calls the corresponding `Arc3ScreenBridge` method with arguments decoded from the RPN tape.

### §3.4 Step B4 — smoke test

**File:** NEW, [tests/cranium/test_arc3_screen_bridge.py](../tests/cranium/test_arc3_screen_bridge.py)

Minimum:
- Load palette, decode a known 4×4 frame with 2 colors, verify RGBA output matches expected bytes
- Click-invert: rect (0,0,64,64), grid 4×4, click (16,16) → cell (1,1)
- Action emit: (action_id=6, gx=2, gy=3) round-trips

Test may use numpy for constructing test vectors / comparing bytes — that's test scaffolding, not hot path (per memory rules).

**Acceptance:**
```bash
export CUDA_VISIBLE_DEVICES=0
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium env PYTHONPATH=$(pwd) \
  pytest tests/cranium/test_arc3_screen_bridge.py -v
# expected: 3 passed
```

### §3.5 Deferred — follow-up lanes for Codex

When Codex is back online, the remaining 63 opcodes need their PTX kernels. Landing order recommendation:

1. **Memory-as-Image (0x2B0–0x2B5)** — smallest (6 opcodes), enables reasoning-trace bake. Kernels to write: `mem_to_dotmap.cu`, `dotmap_to_mem.cu`, `mem_image_bind.cu`, `mem_foveal_encode.cu`, `mem_image_diff.cu`.
2. **Document Galaxy (0x2D0–0x2DB)** — 12 opcodes, enables the symlink-document spec. Per [DOCUMENT_GALAXY_SYMLINK_SPECIFICATION.md](../docs/vocabulary/DOCUMENT_GALAXY_SYMLINK_SPECIFICATION.md) §4 opcode table.
3. **Texture Forge (0x280–0x28F)** — 16 opcodes, visual output for Forge pane.
4. **Image→3D (0x290–0x29F)** — 16 opcodes, volumetric mesh generation.

Each new kernel lands with: .cu file + .ptx compiled + `CODEC_TOKENS` entry + bridge method + `_CODEC_TOKEN_MAP` entry + smoke test. Additive. Never renumber, never replace (per [feedback_expand_not_replace_opcodes.md](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_expand_not_replace_opcodes.md)).

---

## §4 Sovereignty gates (both paths, must hold before any merge)

```bash
cd /K3D/GitHub/Knowledge3D

# No numpy/cupy/scipy/sympy in codec bridges or hot path
grep -Rn "^import numpy\|^import cupy\|^from numpy\|^from cupy\|^import scipy\|^import sympy" \
  knowledge3d/cranium/bridges/ knowledge3d/cranium/codecs/ knowledge3d/cranium/ptx_runtime/ \
  | grep -v "# sovereignty-exempt:"

# No Python regex/string-matching for reasoning in hot path
grep -Rn "re\.\(findall\|search\|match\|sub\)" \
  knowledge3d/cranium/bridges/ knowledge3d/knowledgeverse/

# Every new opcode has a §11 registry row
grep -n "0x28[0-9A-Fa-f]\|0x29[0-9A-Fa-f]\|0x2A[0-9A-Fa-f]\|0x2B[0-9A-Fa-f]\|0x2C[0-9A-Fa-f]\|0x2D[0-9A-Fa-f]" \
  docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md
```

Expected: zero hits on the first two, populated on the third.

---

## §5 Dispatch plan — sub-agent task briefs

Two Sonnet sub-agents run in parallel under Claude-pilot oversight:

**Agent 1 (Path A):** Landed brief: "Execute §2.1 + §2.2 + §2.3 per this spec. Grep before each edit; read before each edit. Run §2.1 and §2.2 acceptance gates. Stop before §2.4 (benchmark run) and report back for Claude review."

**Agent 2 (Path B):** Landed brief: "Execute §3.1 + §3.2 + §3.3 + §3.4 per this spec. Mirror `ternary_codec_ops.py` as the pattern template. No numpy in bridge code. Run §3.4 smoke test if GPU available; if sandbox has no GPU, report compile/import success only. Leave benchmark run to Claude-pilot review."

Both agents report back. Claude-pilot reviews + runs the benchmark suite (§2.4) once Layer 0 is green.

---

## §6 Paper-evidence artifacts produced by this spec

1. `data/benchmark_runs/2026-04-20_dual_path/` — offline bench run results (MMLU, GSM8K, Math, LHE, ARC-2)
2. Git commits: `fix: Layer 0 canonical Drawing primitive IDs`, `feat: Arc3ScreenBridge codec launcher`
3. Test green: `tests/cranium/test_arc3_screen_bridge.py`
4. This spec committed alongside → next-session traceability

---

## §7 Not in scope (explicitly deferred)

- Texture Forge / Image→3D / Memory-as-Image / MVCIC / Document Galaxy PTX kernels (Codex lane, after Layer 0 green)
- Parallel-agent corpus manifest locking + OCR sidecar (Apr-18 spec §3+ deferred until first bench green)
- Viewer Forge pane TypeScript design
- ROADMAP.md phase-update
- TRM composed-head extraction deeper refactor

These are real but out-of-scope for today's paired push. Track in CODEX.md P0 list.
