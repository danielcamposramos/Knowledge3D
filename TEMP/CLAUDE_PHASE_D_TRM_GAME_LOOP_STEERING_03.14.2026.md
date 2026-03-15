# Phase D Steering Directive: TRM Game Loop Migration

**Author:** Claude (Architecture Partner)
**Date:** March 14, 2026
**Status:** ACTIVE — Ready for Codex implementation
**Depends on:** Phase C daemon (DONE), Phase B+ benchmarks (DONE)

---

## Goal

Migrate reasoning orchestration FROM Python INTO the TRM game loop on GPU.

**Before Phase D:** `knowledgeverse.py` is 8,182 lines. The `query()` method (line 7871) calls `_select_composed_head_candidate` (line 5761, **1,157 lines of Python**). TRMLauncher exists (`cranium/sovereign/trm_launcher.py`, 644 lines) but is **NOT imported or called** anywhere in the query path.

**After Phase D:** `knowledgeverse.py` shrinks to ~200 lines (boot + I/O). TRM runs `trm_step_fused.ptx` as an autonomous game loop. The swarm, scoring, candidate selection, and convergence all happen on GPU via kernel composition.

---

## Critical Constraint: Incremental Migration

**DO NOT rewrite knowledgeverse.py.** Phase D is a phased extraction:
- Each sub-step replaces ONE Python orchestration block with a GPU kernel call
- After each sub-step, the quartet must pass: ARC 10/10, Math 20/20, GSM8K 1/10, LHE 6/10, MMLU 13-14/50
- If a sub-step breaks the quartet, REVERT and diagnose before proceeding

---

## Architecture: TRM Game Loop

### What `trm_step_fused.ptx` Does Today

**File:** `knowledge3d/cranium/ptx/trm_step_fused.cu` (128 lines)
**Signature:** `trm_step_fused(q, y, z, W1, W2, W3, W4, z_new, y_new, workspace)`
**Semantics:** Single TRM forward pass — 2-layer SwiGLU MLP
- Input: `q` (query, 512d), `y` (memory state, 512d), `z` (hidden state, 512d)
- Weights: `W1` (512→1024), `W2` (1024→512), `W3` (512→1024), `W4` (1024→512)
- Output: `z_new` (updated hidden), `y_new` (updated memory)
- ~95µs per step on RTX 3070

### What TRM Game Loop MUST Become

Each "game tick" is ONE call to `trm_step_fused.ptx` where:
1. **q = stimulus** — the query embedding (from `_embed_query_gpu`)
2. **y = memory** — accumulated Galaxy navigation state
3. **z = hidden** — current reasoning state
4. The TRM output `y_new` encodes which Galaxy neighborhood to visit next
5. The TRM output `z_new` feeds back as `z` for the next tick
6. Multiple ticks → the TRM navigates the Galaxy, scores candidates, converges

### TRM Game Tick Composition

```
Per tick:
  q (stimulus) ──┐
  y (memory)  ───┤──→ trm_step_fused ──→ z_new, y_new
  z (hidden)  ───┘
                                           │
                                    ┌──────┴──────┐
                                    │  Dispatch   │
                                    └──────┬──────┘
                                           │
                    ┌──────────────────────┬┴┬──────────────────────┐
                    ▼                      ▼ ▼                      ▼
             Morton Octree          LED-A*   Frustum          Dynamic LOD
             (spatial index)      (navigate) (field-of-view)  (detail level)
                    │                      │                       │
                    └──────────┬───────────┘                       │
                               ▼                                   │
                        Nine-Chain Swarm ◄─────────────────────────┘
                         (parallel reason)
                               │
                               ▼
                         Halting Gate
                        (convergence?)
                           │      │
                       YES ▼      ▼ NO
                     Answer    z=z_new, y=y_new → next tick
```

---

## Implementation Plan: 6 Sub-Steps

### Sub-Step D.1: Wire TRMLauncher into Knowledgeverse

**Goal:** Import TRMLauncher, initialize it during boot, make it accessible in the query path. No behavior change yet.

**Files to modify:**
- `knowledge3d/knowledgeverse/knowledgeverse.py` — add TRMLauncher initialization

**What to do:**
1. Import `TRMLauncher` from `knowledge3d.cranium.sovereign.trm_launcher`
2. In the `__init__` or boot method, create `self._trm = TRMLauncher(use_fused=True)`
3. Allocate TRM weight buffers in VRAM (random init for now — Phase D.5 trains them)
4. Add a `_trm_ready` flag (False until weights loaded)
5. **DO NOT call TRM in query path yet** — just wire the plumbing

**Validation:** Quartet unchanged (TRM is initialized but not called)

---

### Sub-Step D.2: TRM State Buffers + Stimulus Encoding

**Goal:** Create the q/y/z state machine that feeds the TRM game loop.

**Files to modify:**
- `knowledge3d/knowledgeverse/knowledgeverse.py` — add state buffer management

**What to do:**
1. Allocate persistent VRAM buffers:
   - `d_q` (512 floats) — stimulus (query embedding projected to 512d)
   - `d_y` (512 floats) — memory state (zero-initialized per query)
   - `d_z` (512 floats) — hidden state (zero-initialized per query)
   - `d_z_new`, `d_y_new` (512 floats each) — outputs
   - `d_workspace` (3072 floats) — scratch for trm_step_fused
2. Create `_encode_stimulus(query_embedding) → d_q`:
   - The current `_embed_query_gpu` returns 16d or 512d embeddings
   - If 16d, project to 512d (existing `_project_embedding16_to512` in `query_head_substrate.py`)
   - Copy to `d_q` via `memcpy_htod`
3. Create `_reset_trm_state()` — zeros `d_y` and `d_z` at query start
4. **DO NOT run TRM ticks yet** — just prepare buffers

**Validation:** Quartet unchanged (buffers allocated but not used in hot path)

---

### Sub-Step D.3: Single-Tick TRM Probe (Shadow Mode)

**Goal:** Run ONE TRM tick in shadow mode alongside the existing Python path. Compare outputs. No effect on answers.

**Files to modify:**
- `knowledge3d/knowledgeverse/knowledgeverse.py` — add shadow TRM call in `query()`

**What to do:**
1. After `_embed_query_gpu` in `query()`, add:
   ```python
   if self._trm_ready:
       self._encode_stimulus(query_embedding)
       self._reset_trm_state()
       self._trm.refine(d_q, d_y, d_z, W1, W2, W3, W4, n_steps=1, eps=0.0)
       # Read back y_new — this is the TRM's "suggestion" for Galaxy neighborhood
       y_new_host = self._read_trm_y_new()
       # Log but don't use: compare y_new direction vs Python's target_galaxies
   ```
2. Log the TRM output alongside Python's galaxy selection for diagnostic comparison
3. Add a `K3D_TRM_SHADOW=1` env var to enable/disable

**Validation:** Quartet unchanged (shadow mode only logs, doesn't affect answers). TRM tick latency logged (~95µs — negligible vs total query time).

---

### Sub-Step D.4: TRM-Guided Galaxy Navigation (Replace `_select_gpu_profile`)

**Goal:** Replace the Python galaxy selection logic with TRM-guided navigation. This is the FIRST behavioral change.

**Current Python path (to replace):**
- `_select_gpu_profile()` — Python function that picks `target_galaxies` and `reasoning_program_id` based on task type, route, specialist, and heuristics

**What to do:**
1. Create `_trm_select_galaxies(query_embedding, task_type) → (target_galaxies, reasoning_program_id)`:
   - Encode stimulus (query embedding)
   - Run 1-3 TRM ticks
   - Decode `y_new` → Galaxy neighborhood selection:
     - `y_new[0:10]` → softmax → weights over the 10 default galaxies
     - Pick galaxies where weight > threshold (e.g., 0.05)
   - Decode `z_new` → reasoning program selection:
     - Cosine similarity between `z_new` and stored program embeddings
     - Pick closest program
2. **Fall back to Python path** if TRM output is degenerate (all zeros, all same weight)
3. Add `K3D_TRM_NAVIGATE=1` env var — when enabled, use TRM navigation; when disabled, use Python path

**Validation:**
- With `K3D_TRM_NAVIGATE=0`: Quartet unchanged
- With `K3D_TRM_NAVIGATE=1`: Measure quartet — EXPECT degradation initially (TRM weights are random)
- Train TRM weights (D.5) before expecting parity

**Why this step first:** Galaxy selection is a small, self-contained decision. If wrong, the downstream pipeline still runs — just with worse neighborhood. Low blast radius.

---

### Sub-Step D.5: TRM Weight Training Loop

**Goal:** Train TRM weights using recorded query traces so that TRM navigation matches (or improves on) Python heuristic navigation.

**What to do:**
1. Create `scripts/train_trm_weights.py`:
   - Load query traces from the existing benchmark runs (task → galaxy selection → answer → correct/incorrect)
   - For each trace, compute target:
     - `y_target` = embedding of the galaxy set that produced the correct answer
     - `z_target` = embedding of the reasoning program that won
   - Loss = MSE(y_new, y_target) + MSE(z_new, z_target)
   - Backprop through `trm_step_fused` (use the existing `trm_extensions.ptx` backward kernels or compute gradients numerically)
   - Save weights to `checkpoints/trm_weights.pt`
2. Training data source: run all benchmarks with `K3D_TRM_SHADOW=1`, capture traces
3. ~7M parameters, ~300 training examples → should converge in minutes

**Validation:**
- After training, enable `K3D_TRM_NAVIGATE=1` and run quartet
- Target: TRM navigation matches Python heuristic (within ±1 on each benchmark)
- Once matched, the Python `_select_gpu_profile` can be deleted

---

### Sub-Step D.6: TRM-Driven Candidate Selection (Replace `_select_composed_head_candidate`)

**Goal:** Replace the 1,157-line Python scoring monster with TRM-driven multi-tick reasoning.

**Current Python path (to replace):**
- `_select_composed_head_candidate()` at line 5761 — builds RPN scoring expressions per candidate, evaluates batch, picks best
- Internally calls: `_parse_bundle_embeddings`, `_navigate_galaxy_neighborhood`, `_build_gpu_candidate_score_expression` (which builds ~50 RPN tokens per candidate with hardcoded weights)

**What to do:**
1. Create `_trm_select_candidate(binding, paths, query_embedding, task_type) → best_candidate`:
   - For each candidate path, encode candidate embedding into stimulus
   - Run TRM tick: `trm_step_fused(candidate_embed, y, z, ...)`
   - Score = magnitude of `z_new` (or dot product with query embedding)
   - Multi-tick: run 3-5 ticks per candidate, use final z_new as score
   - Pick highest-scoring candidate
2. The nine-chain swarm integration:
   - Each of the 9 swarm workers runs a TRM tick with a DIFFERENT candidate
   - This replaces the Python loop over candidates with parallel GPU execution
   - Use `nine_chain_swarm_kernel.ptx` to dispatch parallel TRM evaluations
3. Halting gate integration:
   - After swarm produces 9 scored candidates, run `gre_multimodal_halting_gate.ptx`
   - If halting gate says "converged" → return best candidate
   - If not → refine (Morton → LED-A* → widen neighborhood) and re-run swarm
4. **Fall back to Python path** initially, switch after TRM training covers candidate scoring

**This is the big one.** ~1,157 lines of Python → ~50 lines of TRM dispatch + existing kernel composition.

**Validation:**
- Train TRM on benchmark traces for candidate scoring
- Quartet must hold: ARC 10/10, Math 20/20, GSM8K 1/10, LHE 6/10, MMLU 13-14/50
- Once parity achieved, delete `_select_composed_head_candidate` and all its helpers

---

## Kernel Inventory for Phase D

### Already Available (compiled PTX, wired through bridges):

| Kernel | PTX File | Role in Game Loop |
|--------|----------|-------------------|
| `trm_step_fused` | `trm_step_fused.ptx` | Core game tick |
| `morton_octree_*` | `morton_octree.ptx` | Spatial indexing |
| `led_astar_*` | `led_astar.ptx` | Graph navigation |
| `frustum_cull_*` | `frustum_cull_simd.ptx` | Field-of-view filtering |
| `dynamic_lod_tune` | `dynamic_lod_tune.ptx` | Detail level control |
| `nine_chain_swarm_*` | `nine_chain_swarm_kernel.ptx` | Parallel reasoning |
| `gre_multimodal_halting_gate` | `gre_multimodal_halting_gate.ptx` | Convergence check |
| `cosine_similarity` | `cosine_similarity.ptx` | Embedding comparison |
| `modular_rpn_kernel` | `modular_rpn_kernel.ptx` | RPN execution (3 tiers) |

### May Need New Kernels:

| Need | Description | Priority |
|------|-------------|----------|
| `trm_decode_galaxy_weights` | Softmax over y_new → galaxy weights | D.4 |
| `trm_score_candidates_batch` | Batch TRM ticks over N candidates | D.6 |
| `trm_backward_step` | Gradient for training loop | D.5 (or use numerical) |

---

## Python Deletion Roadmap

As each sub-step succeeds and quartet holds:

| Sub-Step | Python to Delete | Approx Lines |
|----------|-----------------|--------------|
| D.4 done | `_select_gpu_profile` + helpers | ~200 |
| D.6 done | `_select_composed_head_candidate` + all scoring helpers | ~1,500 |
| D.6 done | `_build_gpu_candidate_score_expression` | ~300 |
| D.6 done | `_score_gpu_candidates_batch` | ~50 |
| D.6 done | `_parse_bundle_embeddings` | ~150 |
| D.6 done | `_navigate_galaxy_neighborhood` + helpers | ~400 |
| Future | `_answer_*_query` methods (per-task-type answer formatters) | ~1,500 |
| Future | Remaining orchestration → TRM autonomous | ~2,000 |

**Total Phase D deletion target:** ~2,600 lines (D.4 + D.6)
**Total future deletion:** ~3,500 more lines (post-Phase D)
**Final target:** ~200 lines (boot + I/O + shutdown)

---

## TRMLauncher Current State

**File:** `knowledge3d/cranium/sovereign/trm_launcher.py` (644 lines)

**3 backends:**
1. **Legacy PTX** — `trm_extensions.ptx` (default, individual kernel calls)
2. **RPN** — Tier-3 RPN interpreter (`K3D_USE_RPN_TRM=1`)
3. **Fused PTX** — `trm_step_fused.ptx` (`K3D_USE_FUSED_TRM=1`) ← **USE THIS**

**API:** `launcher.refine(q, y, z, W1, W2, W3, W4, n_steps, eps)` — runs N steps, returns final (z_new, y_new)

**Current problem:** `refine()` is a single-pass refinement. For the game loop, we need:
- Per-tick access to intermediate states (not just final)
- Per-tick dispatch to spatial/swarm kernels between TRM ticks
- Ability to inject new stimulus between ticks

**Solution:** Either extend `refine()` to yield per-tick, or call the raw PTX kernel directly (simpler). The kernel is already loaded as `self.kernel_fused` in TRMLauncher.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `K3D_TRM_SHADOW` | `0` | Enable shadow TRM logging (D.3) |
| `K3D_TRM_NAVIGATE` | `0` | Use TRM for galaxy navigation (D.4) |
| `K3D_TRM_SELECT` | `0` | Use TRM for candidate selection (D.6) |
| `K3D_USE_FUSED_TRM` | `1` | Use fused PTX backend (always on for Phase D) |

---

## Success Criteria

1. **Quartet non-regression** at every sub-step
2. **TRM tick latency** < 200µs (currently ~95µs — plenty of headroom)
3. **knowledgeverse.py** line count decreasing monotonically after D.4
4. **Zero Python in hot path** for galaxy selection (after D.4) and candidate scoring (after D.6)
5. **TRM weights** converge on training data within 1000 iterations

---

## Recommended Order

```
D.1 (wire TRMLauncher) → D.2 (state buffers) → D.3 (shadow mode)
                                                       │
                                              Run benchmarks, collect traces
                                                       │
                                               D.5 (train weights)
                                                       │
                                               D.4 (galaxy navigation)
                                                       │
                                              Validate quartet holds
                                                       │
                                               D.6 (candidate selection)
                                                       │
                                              Delete Python, validate quartet
```

D.1-D.3 are safe (no behavior change). D.5 should run BEFORE D.4 so TRM has useful weights. D.4 is the first real behavior change. D.6 is the big payoff.

---

## Daniel's Mandate

> "The TRM is the avatar. The swarm is how it thinks. Python is just the door."

> "We fail and fix — this is the goal."

Every line of Python deleted from the reasoning path is a victory. Every TRM tick that replaces a Python heuristic is progress toward sovereignty. The goal is not optimization — it's AUTONOMY. The TRM must learn to navigate and reason, not be told what to do by Python.

---

**Codex:** Start with D.1. Wire TRMLauncher into knowledgeverse.py. Run quartet. Report back.
