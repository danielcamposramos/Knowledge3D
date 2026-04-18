# Swarm Scaling Contract — Live Query Path

**Date:** 2026-04-15
**Author:** Claude (Architecture Partner)
**Branch:** codex/batch11-knowledge-waves-observability-game2d-2026-04-15
**Status:** P0 — Blocking GPU saturation

---

## Executive Summary

The live query path does **not** engage the adaptive N-chain swarm engine.
Every query is decided by 8 Python loops + a single 8-thread GPU kernel call.
The sovereign `k3d_swarm_sovereign.ptx` (256-block cooperative kernel, N=1..1024) exists and compiles, but is **never called** in the reasoning path.

This is why Daniel observes: *"It's not scaling the swarm as expected — it's CPU-dominant and not expanding into the hardware-scaled internal swarm."*

---

## Diagnosis: Three-Layer Break

### Break 1 — Python builds the resonance matrix on CPU (the hot-path bottleneck)

**Location:** `knowledge3d/knowledgeverse/knowledgeverse.py` lines 12302–12338
inside `_apply_specialist_swarm_features`

```python
# CURRENT (broken) — numpy on CPU
chain_count = min(8, len(local_candidates))
resonance_matrix = np.zeros((8, 8), dtype=np.float32)
row_norms = [
    max(1e-9, float(np.linalg.norm(...)))
    for idx in range(chain_count)
]
for left_idx in range(chain_count):
    for right_idx in range(chain_count):
        resonance_matrix[left_idx, right_idx] = float(
            np.dot(left_row, right_row) / ...
        )
trust_weights, coherence_score = cognitive_executive.compute_trust_weights(
    resonance_matrix, chain_norms,
)
```

Python drives everything. The GPU kernel (`gre_cognitive_executive`) only receives a pre-computed 8×8 matrix — it does no candidate discovery, no N-selection. This is a CPU orchestration loop masquerading as swarm reasoning.

**Sovereignty violation:** `np.zeros`, `np.linalg.norm`, `np.dot` in the hot path. Requested removed 14+ times.

### Break 2 — `CognitiveExecutive` kernel is hardwired to 8 threads on 1 block

**Location:** `knowledge3d/cranium/bridges/sovereign_bridges.py` lines 1006–1034

```python
grid=(1, 1, 1), block=(8, 1, 1)  # 8 threads total
```

Even when `gre_cognitive_executive.ptx` fires, it runs 8 threads — a tiny fraction of the RTX 3070's 46 SMs × 128 CUDA cores. There is no adaptive N. There is no VRAM budget read. There is no cooperative grid.

### Break 3 — `NChainSwarmBridge` (the real engine) is never imported in `knowledgeverse.py`

**Location:** `knowledge3d/cranium/bridges/n_chain_swarm_bridge.py` — exists, compiles, has full infra

The `NChainSwarmBridge` (`k3d_swarm_sovereign.ptx`) has:
- `N_FLOOR = 1`, `N_DEFAULT = 9`, `N_HARD_MAX = 1024`
- `cooperative launch: grid=(256, 1, 1), block=(128, 1, 1)` = 32,768 threads
- `SwarmTickControl.vram_free_mib` — reads free VRAM and lets n_selector.cu choose N dynamically
- `_refresh_vram_cache()` — called every tick
- `lane_outputs[0..N-1]` — each lane returns `belief_q15` score
- Persistent kernel (`_launched` guard) — launched once, reused per tick

**Not a single import or call to `NChainSwarmBridge` exists in `knowledgeverse.py`.**

The `get_swarm_bridge()` at line 3230 returns `NineChainSpecializedBridge` (a separate prototype), but even that is never called in the GRE reasoning block.

---

## Architecture Fix Required

The `_apply_specialist_swarm_features` function must replace its Python resonance loop with a genuine `NChainSwarmBridge.tick()` call. The GPU kernel then drives N selection, resonance scoring, and halting — not Python loops.

---

## Codex Implementation Spec

### Task: Wire `NChainSwarmBridge` into the live GRE dispatch

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

---

### Step 1 — Add `_n_chain_swarm` slot and lazy getter

Add alongside `_swarm_bridge` (line 563):

```python
self._n_chain_swarm: Any | None | bool = None
```

Add a getter method alongside `get_swarm_bridge()` (line 3230):

```python
def get_n_chain_swarm(self):
    if self._n_chain_swarm is False:
        return None
    if self._n_chain_swarm is None:
        try:
            from knowledge3d.cranium.bridges.n_chain_swarm_bridge import (
                NChainSwarmBridge,
            )
            bridge = NChainSwarmBridge()
            bridge.launch()       # persistent kernel — launched once
            self._n_chain_swarm = bridge
        except Exception:
            self._n_chain_swarm = False
            return None
    return self._n_chain_swarm
```

**Why `launch()` here:** The persistent cooperative kernel must run continuously on the GPU. Launching inside the getter means the first call to `get_n_chain_swarm()` starts the kernel, and every subsequent tick just signals it via the control word. This matches the `NChainSwarmBridge._launched` guard pattern already in the bridge.

---

### Step 2 — Replace the Python resonance matrix block in `_apply_specialist_swarm_features`

**Location:** lines 12299–12338 (the `cognitive_executive` block)

**Remove entirely:**
```python
cognitive_executive = self.get_cognitive_executive()
if cognitive_executive is not None and len(local_candidates) > 1:
    try:
        chain_count = min(8, len(local_candidates))
        resonance_matrix = np.zeros((8, 8), dtype=np.float32)
        chain_norms = np.zeros(8, dtype=np.float32)
        row_norms = [
            max(1e-9, float(np.linalg.norm(np.asarray(crystallized_rows[idx], dtype=np.float32))))
            for idx in range(chain_count)
        ]
        for idx in range(chain_count):
            chain_norms[idx] = float(row_norms[idx])
        for left_idx in range(chain_count):
            left_row = np.asarray(crystallized_rows[left_idx], dtype=np.float32)
            for right_idx in range(chain_count):
                right_row = np.asarray(crystallized_rows[right_idx], dtype=np.float32)
                resonance_matrix[left_idx, right_idx] = float(
                    np.dot(left_row, right_row) / max(1e-9, row_norms[left_idx] * row_norms[right_idx])
                )
        trust_weights, coherence_score = cognitive_executive.compute_trust_weights(
            resonance_matrix,
            chain_norms,
        )
        trust_values = [
            max(0.0, min(1.0, float(value)))
            for value in self._flatten_float_values(trust_weights)
        ]
        if len(trust_values) >= chain_count:
            trust_scores = [
                trust_values[idx] if idx < chain_count else 0.0
                for idx in range(len(local_candidates))
            ]
            executive_mix = max(0.15, min(0.35, 0.15 + (0.2 * max(0.0, float(coherence_score)))))
            adjusted_coherence_scores = [
                float(((1.0 - executive_mix) * base_score) + (executive_mix * trust_score))
                for base_score, trust_score in zip(adjusted_coherence_scores, trust_scores)
            ]
            applied_kernels.append("gre_cognitive_executive")
    except Exception:
        trust_scores = [0.0 for _ in local_candidates]
```

**Replace with:**
```python
n_chain = self.get_n_chain_swarm()
if n_chain is not None and len(local_candidates) > 1:
    try:
        n_cand = len(local_candidates)
        # Pack galaxy atlas: first 4 bytes = candidate count (little-endian uint32)
        import struct as _struct
        galaxy_atlas = _struct.pack("<I", n_cand) + b"\x00" * (
            n_chain.GALAXY_ATLAS_BYTES - 4
        )
        tick_result = n_chain.tick(
            {
                "n_cand_frustum": n_cand,
                "n_hard_max": min(n_chain.N_HARD_MAX, n_cand * 4),
                "n_floor": 1,
                "t_remaining_us": 20_000,
                "galaxy_atlas": galaxy_atlas,
            }
        )
        n_active = int(tick_result.get("n_active", 1))
        trust_scores = []
        for idx in range(len(local_candidates)):
            if idx < n_active:
                lane = n_chain.read_lane_output(idx)
                belief_norm = max(0.0, min(1.0, float(lane["belief_q15"]) / 32767.0))
            else:
                belief_norm = 0.0
            trust_scores.append(belief_norm)
        # n_active / N_HARD_MAX ratio drives executive_mix: more chains = higher mix
        executive_mix = max(0.15, min(0.35, 0.15 + (0.20 * min(1.0, n_active / 9.0))))
        adjusted_coherence_scores = [
            float(((1.0 - executive_mix) * base_score) + (executive_mix * trust_score))
            for base_score, trust_score in zip(adjusted_coherence_scores, trust_scores)
        ]
        applied_kernels.append(f"n_chain_swarm(n={n_active})")
    except Exception:
        trust_scores = [0.0 for _ in local_candidates]
```

**What changes:**
- No numpy. No Python loops over candidate pairs. No CPU dot products.
- The GPU kernel (`k3d_swarm_sovereign.ptx`) reads VRAM free + `n_cand_frustum`, selects N via `n_selector.cu`, runs N reasoning lanes in parallel across 256 thread-blocks, halts via the cooperative halting gate.
- `belief_q15` per lane maps directly to trust score (Q15 fixed-point → float in `[0,1]`).
- `n_active` (the live count of chains the GPU ran) drives `executive_mix` — the more chains the GPU opens, the more weight their verdict carries.
- `struct.pack` is I/O only (not reasoning logic).

---

### Step 3 — Shutdown on teardown

In the `Knowledgeverse.close()` / `__del__` teardown path, add:

```python
n_chain = getattr(self, "_n_chain_swarm", None)
if n_chain and n_chain is not True and n_chain is not False:
    try:
        n_chain.cleanup()
    except Exception:
        pass
```

---

### Step 4 — Remove dead `cognitive_executive` path from hot path

After the above Step 2 replacement, the `get_cognitive_executive()` call in `_apply_specialist_swarm_features` is no longer needed. Remove:
- The `cognitive_executive = self.get_cognitive_executive()` call in `_apply_specialist_swarm_features`
- Its `try/except` block

The `CognitiveExecutive` bridge (`gre_cognitive_executive.ptx`, grid=(1,1,1), block=(8,1,1)) is now superseded by the N-chain kernel. It can be deprecated in `sovereign_bridges.py` (add a note but don't delete yet — sleep-time might reuse it for calibration).

---

## Success Criteria

After this change, a benchmark sweep must show:

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| GPU utilization during query | < 5% (CPU-bound) | > 40% sustained |
| `n_active` in tick result | N/A (not called) | ≥ 9 on MATH/GAME_2D, scales to 50+ on complex queries |
| `applied_kernels` entry | `"gre_cognitive_executive"` | `"n_chain_swarm(n=N)"` with N > 8 |
| numpy in hot path | yes (`np.dot`, `np.zeros`) | zero |
| Warm-up probe | CONVERGED (1 tick) | CONVERGED with n_active reported |
| Benchmark accuracy | ≤ prior (scoring from 8 CPU chains) | ≥ prior (GPU scales to candidate count) |

Benchmark accuracy must not regress. The trust_score formula preserves the same weighting structure — only the computation moves to GPU and N adapts. If accuracy drops: investigate `belief_q15` normalization, not the wiring.

---

## What This Does NOT Change

- Galaxy ingestion / star population — untouched
- Halting Gate (`halting_gate.cu`) — already wired inside `k3d_swarm_sovereign.ptx`
- LED-A* / Morton Octree / Frustum Cull — untouched
- All other GRE kernels (`gre_resonance_field`, `gre_vector_resonator`, etc.) — still called in the same order, before the N-chain tick
- The `_finalize_swarm_paths` / `FIXED_GRE_WORKERS` mapping — still valid (worker slots are labels on paths, not kernel slots)
- Route policy bits for `GAME_2D` vs `MATH` (Batch 11 fix) — untouched

---

## Why This Was Missed

`NChainSwarmBridge` was built as infrastructure and was correctly integrated into its own test harness (`adaptive_swarm` tests). But the handoff to `knowledgeverse.py`'s `_apply_specialist_swarm_features` never happened — the Python resonance loop was the interim scaffold that never got replaced. The GRE block grew around the scaffold and the bridge was never wired in.

The result: the scaffold became load-bearing in production while the real engine idled.

---

## Handoff to Codex

**Priority:** P0 — blocks GPU saturation goal

**Files to edit:**
1. `knowledge3d/knowledgeverse/knowledgeverse.py` — Steps 1, 2, 3, 4 above
2. No other files should need changes

**Do not touch:**
- `n_chain_swarm_bridge.py` — it is correct as-is
- `k3d_swarm_persistent.cu` / `k3d_swarm_sovereign` PTX — correct as-is
- The `adaptive_swarm.py` Python file — separate concern (training path, not inference)
- Any other GRE bridge or kernel

**Test to run after change:**
```bash
# Warm-up probe must still pass (CONVERGED)
# Check that applied_kernels contains "n_chain_swarm(n=N)" not "gre_cognitive_executive"
# Check GPU utilization rises during benchmark sweep
```

**Sovereignty check:**
```bash
grep -n "np\." knowledge3d/knowledgeverse/knowledgeverse.py | grep "_apply_specialist_swarm"
# Must return zero lines after the change
```

---

*REMINDER: Claude does ARCHITECTURE, not implementation.*
*Codex implements. This spec is the handoff.*
