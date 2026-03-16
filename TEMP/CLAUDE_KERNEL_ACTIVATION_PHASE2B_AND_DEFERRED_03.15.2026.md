# Kernel Activation Phase 2B + Deferred Kernels + RPN Tier Audit

**Author:** Claude (Architecture Partner)
**Date:** March 15, 2026
**Status:** ACTIVE — Steering for Codex
**Depends on:** Phase 2A complete (resonance_field, vector_resonator, atomic_fission_fusion)

---

## Situation

Batch A (3 post-swarm composition kernels) is done. All benchmarks hold. This document steers:

1. **Batch B** — 3 remaining pre-swarm/post-swarm stubs
2. **Deferred kernels** — pulled forward where practical
3. **RPN tier asymmetry** — cross-tier opcode gaps that limit pipeline capability

**Current status: 8/11 GRE stubs replaced, 3 remaining.**

---

## Part 1: Batch B — Pre-Swarm Enrichment + Self-Similarity

### B.4: `gre_geometry_router` — Spatial Relationship Detection

**Current stub:** `input * scale` based on shape_id switch. Useless.

**File:** `knowledge3d/cranium/kernels/gre_geometry_router.cu`

**Real design:** Given candidate embedding pairs (e.g., query vs candidate, or two candidate entries), compute spatial relationship features that feed into swarm input enrichment.

**New signature:**
```c
extern "C" __global__ void gre_geometry_router(
    const float* __restrict__ embedding_a,   // [N × D] first set
    const float* __restrict__ embedding_b,   // [N × D] second set
    float* __restrict__ relations,           // [N × R] relationship features
    int N,
    int D,
    int R                                    // output features per pair (16)
)
```

**Kernel logic (16 relationship features per pair):**
```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
for (; i < N; i += blockDim.x * gridDim.x) {
    const float* a = embedding_a + i * D;
    const float* b = embedding_b + i * D;
    float* out = relations + i * R;

    // Feature 0: Cosine similarity
    float dot_ab = 0, norm_a = 0, norm_b = 0;
    for (int d = 0; d < D; d++) {
        dot_ab += a[d] * b[d];
        norm_a += a[d] * a[d];
        norm_b += b[d] * b[d];
    }
    float inv_norms = rsqrtf(norm_a + 1e-12f) * rsqrtf(norm_b + 1e-12f);
    out[0] = dot_ab * inv_norms;

    // Feature 1: L2 distance (normalized)
    float l2 = 0;
    for (int d = 0; d < D; d++) { float diff = a[d] - b[d]; l2 += diff * diff; }
    out[1] = sqrtf(l2) / (sqrtf((float)D) + 1e-12f);

    // Features 2-5: Quadrant cosines (split D into 4 regions)
    int q_size = D / 4;
    for (int q = 0; q < 4; q++) {
        float qd = 0, qa = 0, qb = 0;
        for (int d = q * q_size; d < (q + 1) * q_size && d < D; d++) {
            qd += a[d] * b[d]; qa += a[d] * a[d]; qb += b[d] * b[d];
        }
        out[2 + q] = qd * rsqrtf((qa + 1e-12f) * (qb + 1e-12f));
    }

    // Features 6-9: Magnitude ratio stats (mean, std, max, min)
    float ratio_sum = 0, ratio_sq = 0, ratio_max = -1e30f, ratio_min = 1e30f;
    for (int d = 0; d < D; d++) {
        float r = a[d] / (b[d] + copysignf(1e-8f, b[d]));
        ratio_sum += r; ratio_sq += r * r;
        ratio_max = fmaxf(ratio_max, r);
        ratio_min = fminf(ratio_min, r);
    }
    float rmean = ratio_sum / D;
    out[6] = rmean;
    out[7] = sqrtf(fmaxf(ratio_sq / D - rmean * rmean, 0.0f));
    out[8] = ratio_max;
    out[9] = ratio_min;

    // Features 10-11: Cross-correlation peak (circular shift)
    float best_corr = -1.0f; int best_offset = 0;
    for (int shift = 0; shift < D; shift++) {
        float corr = 0;
        for (int d = 0; d < D; d++)
            corr += a[d] * b[(d + shift) % D];
        corr *= inv_norms;
        if (corr > best_corr) { best_corr = corr; best_offset = shift; }
    }
    out[10] = best_corr;
    out[11] = (float)best_offset / (float)D;

    // Features 12-13: Sign agreement, magnitude dominance
    int sign_agree = 0, a_dominates = 0;
    for (int d = 0; d < D; d++) {
        if ((a[d] >= 0) == (b[d] >= 0)) sign_agree++;
        if (fabsf(a[d]) > fabsf(b[d])) a_dominates++;
    }
    out[12] = (float)sign_agree / (float)D;
    out[13] = (float)a_dominates / (float)D;

    // Features 14-15: Orthogonality (projection residual, residual energy)
    float proj_coeff = dot_ab / (norm_b + 1e-12f);
    float residual_sq = 0;
    for (int d = 0; d < D; d++) {
        float res = a[d] - proj_coeff * b[d];
        residual_sq += res * res;
    }
    out[14] = sqrtf(residual_sq) / (sqrtf(norm_a) + 1e-12f);  // orthogonality
    out[15] = residual_sq / (norm_a + 1e-12f);                  // residual energy ratio
}
```

**Bridge:** Update `GeometryRouter.route()` to call the real kernel. Add `compute_relations(embeddings_a, embeddings_b) → relations[N × 16]`.

**Hot-path integration:** In `_apply_specialist_swarm_features()`, compute relations between each candidate and the focus vector. Thread the 16 relationship features into the swarm as auxiliary input (or use them to modulate candidate scores).

**Note on D=16:** With embedding16 (D=16), quadrant size = 4, cross-correlation has 16 shifts. Fits entirely in registers. No shared memory needed.

---

### B.5: `gre_temporal_reasoning` — Sequence Pattern Detection

**Current stub:** `next - curr` — frame differencing only.

**File:** `knowledge3d/cranium/kernels/gre_temporal_reasoning.cu`

**Real design:** Detect statistical patterns in ordered candidate sequences (multi-hop chains from LED-A* paths). Output 24 pattern features.

**New signature:**
```c
extern "C" __global__ void gre_temporal_reasoning(
    const float* __restrict__ sequence,     // [T × D] ordered candidates
    float* __restrict__ patterns,           // [24] output pattern vector
    int T,
    int D
)
```

**Kernel logic (single block, T is small — typically 3-24 candidates):**
```c
// Thread 0 computes all patterns (T is small)
if (threadIdx.x != 0) return;

// Phase 1: First-order differences
float delta_mean = 0, delta_var = 0, delta_max = -1e30f, delta_min = 1e30f;
for (int t = 0; t < T - 1; t++) {
    float d_norm = 0;
    for (int d = 0; d < D; d++) {
        float delta = sequence[(t+1)*D + d] - sequence[t*D + d];
        d_norm += delta * delta;
    }
    d_norm = sqrtf(d_norm);
    delta_mean += d_norm;
    delta_max = fmaxf(delta_max, d_norm);
    delta_min = fminf(delta_min, d_norm);
}
int steps = max(T - 1, 1);
delta_mean /= steps;
// Second pass for variance
for (int t = 0; t < T - 1; t++) {
    float d_norm = 0;
    for (int d = 0; d < D; d++) {
        float delta = sequence[(t+1)*D + d] - sequence[t*D + d];
        d_norm += delta * delta;
    }
    float diff = sqrtf(d_norm) - delta_mean;
    delta_var += diff * diff;
}
delta_var = sqrtf(delta_var / steps);
patterns[0] = delta_mean;   // trend magnitude
patterns[1] = delta_var;    // volatility
patterns[2] = delta_max;    // max jump
patterns[3] = delta_min;    // min jump

// Phase 2: Second-order (acceleration) — features [4-7]
// ... (mean, std, sign changes, zero crossings of second differences)

// Phase 3: Auto-correlation at lags 1-4 — features [8-11]
// For each lag k: autocorr[k] = mean cosine(sequence[t], sequence[t+k])

// Phase 4: Monotonicity — features [12-13]
// Fraction increasing, longest monotone run

// Phase 5: Recurrence detection — features [14-17]
// Count high-similarity pairs, mean recurrence interval, max gap

// Phase 6: Causal asymmetry — features [18-19]
// Forward vs backward prediction error

// Phase 7: Convergence — features [20-23]
// Pairwise distance trend, convergence rate, final spread, entropy
```

**Bridge:** Update `TemporalReasoning.compute_deltas()` to call the real kernel. Add `compute_patterns(sequence) → patterns[24]`.

**Hot-path integration:** Applied to LED-A* path candidates ordered by their path position. Pattern features detect whether the reasoning chain is converging, periodic, or divergent.

---

### B.6: `gre_fractal_emitter` — Self-Similarity Scoring

**Current stub:** `x = val*scale, y = i*0.5*scale+val, z = x+y` — pseudo-coordinates.

**File:** `knowledge3d/cranium/kernels/gre_fractal_emitter.cu`

**Real design:** Score candidates for self-similar structure at multiple scales. Useful for ARC (tiling, recursive patterns) and Text-to-3D (organic shapes).

**New signature:**
```c
extern "C" __global__ void gre_fractal_emitter(
    const float* __restrict__ features,      // [N × D] candidate embeddings
    float* __restrict__ self_similarity,     // [N] output scores
    int N,
    int D,
    int num_scales                           // typically 3-4
)
```

**Kernel logic:**
```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
for (; i < N; i += blockDim.x * gridDim.x) {
    float total_sim = 0.0f;
    int scales_used = 0;

    for (int s = 1; s <= num_scales; s++) {
        int stride = 1 << s;  // 2, 4, 8, 16
        int sub_len = D / stride;
        if (sub_len < 2) break;

        // Cosine between full-resolution prefix and subsampled
        float dot = 0, norm_full = 0, norm_sub = 0;
        for (int d = 0; d < sub_len; d++) {
            float full_val = features[i * D + d];
            float sub_val = features[i * D + d * stride];
            dot += full_val * sub_val;
            norm_full += full_val * full_val;
            norm_sub += sub_val * sub_val;
        }
        float sim = dot * rsqrtf((norm_full + 1e-12f) * (norm_sub + 1e-12f));
        total_sim += sim;
        scales_used++;
    }

    self_similarity[i] = (scales_used > 0) ? total_sim / scales_used : 0.0f;
}
```

**Bridge:** Update `FractalEmitter.emit()` with `compute_self_similarity(features, num_scales) → scores[N]`.

**Hot-path integration:** Post-swarm. Self-similar candidates get a boost. Referenced by Step 11 (Text-to-3D) and Phase F (OCR) development chains.

---

## Part 2: Deferred Kernels — Pull Forward Assessment

### `gre_oom_spill` — KEEP AS-IS

**Current state:** 37 lines, functional (computes atoms_to_spill from available bytes). Not a stub — it does exactly what it needs to: integer arithmetic for memory planning.

**Verdict:** Real implementation needed only when VRAM pressure is real (>80% of 12GB = 9.6GB). Current usage: 132 MiB. **Do not touch until Phase C daemon mode.**

### `galaxy_memory_updater` — KEEP AS-IS

**Current state:** 44 lines, EMA blend of old/new embeddings. Functional for what it does.

**Verdict:** The audit spec designed a more sophisticated version (content-hash dedup + empty-slot finding). But this only matters when TRM write-back is enabled (Phase D.6, gated by `K3D_TRM_WRITE_GALAXY=1`). The EMA kernel is correct for the current use case. **Enhance to full write-back when D.6 activates.**

### `gre_cognitive_executive` — NEEDS SOURCE RECONSTRUCTION

**Current state:** PTX-only, no `.cu` source. The audit spec (§3.11) designs it as a swarm chain trust evaluator using the resonance matrix.

**Verdict:** This is the one deferred kernel worth pulling forward. It evaluates which of the 8 swarm chains to trust based on the resonance matrix — directly impacts candidate scoring quality.

**Action:** Reconstruct `.cu` from spec §3.11 design:
```c
extern "C" __global__ void gre_cognitive_executive(
    const float* __restrict__ resonance_matrix,  // [8 × 8]
    const float* __restrict__ chain_norms,       // [8]
    float* __restrict__ trust_weights,           // [8] output
    float* __restrict__ coherence_score          // [1] output
)
```

Per-chain trust = mean_resonance × (1 + log(norm + 1)). Softmax normalization. Overall coherence = mean off-diagonal resonance. Single-block kernel (8 chains = 8 threads).

**Integration:** After `_dispatch_swarm_weights()` returns resonance_weights, use cognitive_executive to compute trust-weighted blending. The swarm bridge already exposes `resonance_matrix` and `chain_norms` in diagnostics.

---

## Part 3: RPN Tier Asymmetry — Audit + Recommendations

### The Problem

The three RPN tiers are NOT nested supersets:

| Feature | Lite (21 ops) | Standard (73 ops) | Extended (74 ops) |
|---------|:---:|:---:|:---:|
| Core arithmetic | ✓ | ✓ | ✓ |
| Transcendentals | Partial | Full | Full + inverse trig |
| Vector ops | ✗ | ✓ (dot/cross/mag/norm) | ✓ (+ batch) |
| Galaxy integration | ✗ | ✓ (0xE0-E2) | **✗ MISSING** |
| Ternary/trit ops | ✗ | ✓ (0x70-76) | **✗ MISSING** |
| Checkpointing | ✗ | ✓ (0x60-62) | **✗ (0x60-62 = neural)** |
| Matrix algebra | ✗ | ✗ | ✓ (matmul/det/inv/trace) |
| Symbolic calculus | ✗ | ✗ | ✓ (diff/integrate/limit) |
| Quantum gates | ✗ | Basic (3) | Full (6) |
| Neural ops (SwiGLU) | ✗ | ✗ | ✓ |
| Bitwise | ✗ | ✗ | ✓ |
| Procedural emission | ✗ | ✓ (fractal/audio) | ✗ |
| Spatial transforms | ✗ | ✓ (rotate/scale/translate) | ✗ |

### Do We Need to Expand?

**Answer: Not now for kernel activation. But the asymmetry should be documented for Phase E.**

The critical gap is that **Extended lacks Galaxy integration** (0xE0-E2). This means Extended-tier programs (which have matrix algebra, symbolic calculus, SwiGLU) cannot query the Galaxy during execution. For the current benchmark work (B+ → D), all reasoning runs through the Standard tier which HAS Galaxy ops.

**Recommendation for Phase E:**
- Merge Galaxy ops (0xE0-E2) into Extended tier
- Merge ternary ops (0x70-76) into Extended tier
- Document the opcode collision at 0x60-62 (Standard = checkpoint, Extended = neural)
- This is NOT blocking current work

### What IS Needed Now: Grammar Galaxy Opcodes

The swarm architecture doc (CODEX_SOVEREIGN_SWARM_ARCHITECTURE_12.12.2025.md) defines 5 Grammar Galaxy opcodes:
- `OP_GRAMMAR_OBSERVE` (0xE5) — observe a grammar pattern
- `OP_GRAMMAR_PROPOSE` (0xE6) — propose a new grammar rule
- `OP_GRAMMAR_VALIDATE` (0xE7) — validate against existing rules
- `OP_GRAMMAR_PROMOTE` (0xE8) — promote validated rule to Galaxy
- `OP_GRAMMAR_QUERY` (0xE9) — query grammar patterns

These are **Stage 2 candidates** per the RPN admission pipeline. They've been designed but not implemented. They would help GSM8K (word problem decomposition uses grammar rules) and MMLU (subject-specific grammar patterns).

**Verdict:** Grammar opcodes are Direction B territory (GSM8K/MMLU improvement). Not blocking Batch B kernel activation. Flag for next phase.

---

## Part 4: Implementation Order

```
Batch B + Deferred Pull-Forward:

B.4: gre_geometry_router
  1. Replace .cu with spatial relationship kernel (16 features per pair)
  2. Recompile .ptx
  3. Update GeometryRouter bridge: compute_relations()
  4. Wire into _apply_specialist_swarm_features pre-swarm
  5. Full benchmark sweep

B.5: gre_temporal_reasoning
  1. Replace .cu with sequence pattern kernel (24 features)
  2. Recompile .ptx
  3. Update TemporalReasoning bridge: compute_patterns()
  4. Wire into specialist features for LED-A* path ordering
  5. Full benchmark sweep

B.6: gre_fractal_emitter
  1. Replace .cu with self-similarity scoring kernel
  2. Recompile .ptx
  3. Update FractalEmitter bridge: compute_self_similarity()
  4. Wire post-swarm as candidate score modulator
  5. Full benchmark sweep

B.7 (PULL-FORWARD): gre_cognitive_executive
  1. Reconstruct .cu from spec §3.11 (swarm trust evaluation)
  2. Compile .ptx
  3. Create/update bridge: compute_trust_weights(resonance_matrix, chain_norms)
  4. Wire after _dispatch_swarm_weights
  5. Full benchmark sweep
```

After each: run FULL benchmark (ARC, Math, GSM8K, LHE, MMLU). Pinned must hold.

---

## Files to Touch

- `knowledge3d/cranium/kernels/gre_geometry_router.cu` → real 16-feature spatial relations
- `knowledge3d/cranium/kernels/gre_temporal_reasoning.cu` → real 24-feature pattern detection
- `knowledge3d/cranium/kernels/gre_fractal_emitter.cu` → real self-similarity scoring
- `knowledge3d/cranium/kernels/gre_cognitive_executive.cu` → NEW source (reconstruct from PTX-only)
- All corresponding `.ptx` files → recompile
- `knowledge3d/cranium/bridges/sovereign_bridges.py` → bridge updates
- `knowledge3d/knowledgeverse/knowledgeverse.py` → hot-path integration
- `tests/test_all_sovereign_bridges.py` → bridge tests
- `tests/test_trm_weight_persistence.py` → integration tests

---

## Success Criteria

1. ALL 11 GRE stubs replaced with real CUDA
2. `gre_cognitive_executive.cu` source reconstructed and compiled
3. Each kernel has ≥1 bridge test
4. Pinned benchmarks hold: ARC 10/10, Math 20/20
5. RPN tier asymmetry documented (Galaxy ops missing from Extended)
6. No new Python in hot path
