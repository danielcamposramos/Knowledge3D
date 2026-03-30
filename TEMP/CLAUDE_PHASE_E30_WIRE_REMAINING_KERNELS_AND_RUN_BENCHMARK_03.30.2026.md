# Claude -- Phase E.30: Wire Remaining Kernels + Run Full Benchmark

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** HIGH -- validate 248K stars + wired kernels with real numbers

---

## Context

E.26-E.29 delivered:
- 248,078 stars across 19 galaxies (was ~13K effective before)
- TRM always-on, no env-var gating
- Warm boot from consolidated checkpoint
- Sleep-time saves state to disk
- Local ARC3 benchmark with goal-relative encoding
- gre_world_model wired into swarm dispatch

**Now we need real numbers.** Run the full benchmark with 248K stars and see where
we stand compared to the March 19 run (3,324/15,601 with only ~5K stars).

---

## Part 1: Wire 3 Remaining Kernel Bridges into Swarm Dispatch

### Current State

From `SOVEREIGN_NSI_SPECIFICATION.md` section 9.1, 15 GRE specialist kernels exist.
7 are already wired into `_apply_specialist_swarm_features()`. Of the remaining 8:

**3 should be wired into the swarm dispatch:**

| Kernel | Bridge | Current Usage | Should Be |
|--------|--------|---------------|-----------|
| `gre_cognitive_executive` | `CognitiveExecutive.compute_trust_weights()` | Sleep-time only (line 11313) | Also in swarm: compute inter-chain trust weights, feed into candidate scoring |
| `gre_atomic_fission_fusion` | `AtomicFissionFusion.decompose()` | Used in composition analysis (lines 7933, 8502) | Also in swarm: check compositional consistency of multi-galaxy candidates |
| `gre_defeasible_resolver` | `DefeasibleResolver.resolve()` | Used in defeasible scoring (lines 2770, 2886) | Also in swarm: resolve conflicting candidate scores via rule superiority |

**5 are correctly NOT in the swarm:**

| Kernel | Reason |
|--------|--------|
| `gre_halting_gate` | Pipeline control, not scoring |
| `gre_sub100micro_gate` | Latency measurement, not scoring |
| `gre_oom_spill` | Memory management, not scoring |
| `gre_arc_reasoner` | Bridge exists, zero references anywhere (needs future wiring separately) |
| `modular_rpn_kernel` | RPN execution engine, not specialist scoring |

### How to Wire (No New Python Orchestration)

These three kernels already have bridge accessors (`get_cognitive_executive()`,
`get_atomic_fission_fusion()`, `get_defeasible_resolver()`) and are already called
elsewhere in the codebase. The wiring adds them to the swarm dispatch alongside the
7 already there.

**In `_apply_specialist_swarm_features()`**, after the existing kernel dispatch block:

#### 1. `gre_cognitive_executive` -- Swarm Trust Weights

```python
cognitive_executive = self.get_cognitive_executive()
if cognitive_executive is not None and len(local_candidates) > 1:
    try:
        # Build inter-chain resonance from candidate embeddings
        chain_count = min(8, len(local_candidates))
        resonance_matrix = np.zeros((8, 8), dtype=np.float32)
        chain_norms = np.zeros(8, dtype=np.float32)
        for i in range(chain_count):
            chain_norms[i] = float(np.linalg.norm(crystallized_rows[i]))
            for j in range(chain_count):
                resonance_matrix[i, j] = float(
                    np.dot(crystallized_rows[i], crystallized_rows[j])
                    / max(1e-9, chain_norms[i] * float(np.linalg.norm(crystallized_rows[j])))
                )
        trust_weights, coherence_score = cognitive_executive.compute_trust_weights(
            resonance_matrix, chain_norms
        )
        # Boost high-trust candidates
        for idx in range(min(len(local_candidates), 8)):
            local_candidates[idx]["specialist_trust"] = float(trust_weights[idx])
        applied_kernels.append("gre_cognitive_executive")
    except Exception:
        pass
```

#### 2. `gre_atomic_fission_fusion` -- Compositional Consistency

```python
atomic_bridge = self.get_atomic_fission_fusion()
if atomic_bridge is not None and len(local_candidates) > 0:
    try:
        for idx, candidate in enumerate(local_candidates):
            compound = np.asarray(crystallized_rows[idx], dtype=np.float32)
            # Decompose against focus vector
            _, consistency = atomic_bridge.decompose(
                compound.reshape(1, -1),
                np.asarray(focus_vector, dtype=np.float32).reshape(1, -1),
            )
            candidate["specialist_composition"] = float(consistency)
        applied_kernels.append("gre_atomic_fission_fusion")
    except Exception:
        pass
```

#### 3. `gre_defeasible_resolver` -- already used at path/intra/cross stages

The defeasible resolver is already called at 3 pipeline stages (lines 2770, 2886,
3086). Its scores already flow into `specialist_intra_defeasible` and
`specialist_defeasible_verdict`. This one is already wired correctly -- it just
doesn't go through `_apply_specialist_swarm_features` because it operates at a
different pipeline stage (path-level, not candidate-level).

**Net result: 9 of 15 kernels actively dispatched** (was 7). The remaining 6 are
pipeline control, memory management, or need separate ARC3 wiring.

---

## Part 2: Run Full Benchmark with 248K Stars

### Command

```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 scripts/run_full_benchmark.py \
    --mmlu-count 50 \
    --gsm8k-count 10 \
    --lhe-count 10 \
    --arc2-count 10 \
    --storage-root /K3D/Knowledge3D.local
```

### Expected Improvements Over March 19 (3,324/15,601)

| Suite | March 19 | Why It Should Improve |
|-------|----------|----------------------|
| MMLU 50 | 23.18% (3255/14042) | 248K stars vs 5K; meaning_layer_stars now queryable |
| GSM8K 10 | 2.27% (30/1319) | 100 new Grammar rules (unit conversions, math transforms) |
| LHE 10 | 9% (9/100) | 117K meaning_layer_stars provide semantic depth |
| ARC-AGI-2 10 | 8.33% (10/120) | 116K Language.jsonl symlinks for visual reasoning |

### What to Log

The run should produce:
- `summary.json` with suite-level results
- Per-suite JSONL with individual results
- Galaxy bind count (should show 19 galaxies, 248K entries)
- TRM tick count (should show ticks on every query -- always-on)

### After the Run

Report:
1. Accuracy per suite
2. Total stars loaded at boot (should be 248K)
3. GPU utilization during run (nvidia-smi)
4. Whether warm boot was used (check for "warm boot" in stdout)
5. Any hangs or failures

---

## Part 3: Run Local ARC3 Benchmark

### Command

```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 scripts/run_arc3_local.py \
    --count 20 \
    --grid-size 8 \
    --max-actions 40 \
    --storage-root /K3D/Knowledge3D.local
```

### Success Criterion (Daniel's Minimum)

> "pass one game at least or hit right on at least one movement"

With goal-relative encoding and the direct decode fix, single-axis tasks should
solve trivially: object above goal -> "move down" -> action 1. We expect 14-18/20
solved.

---

## Execution Sequence

1. Wire `gre_cognitive_executive` and `gre_atomic_fission_fusion` into swarm (Part 1)
2. Run local ARC3 benchmark (Part 3) -- quick validation
3. Run full benchmark (Part 2) -- the real test with 248K stars
4. Report results with star counts, GPU util, warm boot status

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Wire 2 kernels into `_apply_specialist_swarm_features()` |

## No Files to Create

This is a wiring + validation phase. The infrastructure exists.

---

## Success Criteria

- [ ] `gre_cognitive_executive` dispatched in swarm (trust weights influence scoring)
- [ ] `gre_atomic_fission_fusion` dispatched in swarm (compositional consistency)
- [ ] 9/15 GRE kernels actively dispatched (was 7)
- [ ] Local ARC3: at least 1 game solved or 1 correct first move
- [ ] Full benchmark: results with 248K stars (compare to March 19 baseline)
- [ ] Boot log shows 19 galaxies, 248K entries
- [ ] TRM ticks on every query (no env-var gating)
- [ ] Warm boot used if checkpoint exists
