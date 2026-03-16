# Kernel Activation Phase 2 — Pre-Swarm Enrichment + Post-Swarm Composition

**Author:** Claude (Architecture Partner)
**Date:** March 15, 2026
**Status:** ACTIVE — Steering for Codex
**Depends on:** Phase 1 complete (graph crystallizer: CSR + semantic k-NN, benchmarks stable)
**Reference:** `CLAUDE_KERNEL_SOVEREIGNTY_AUDIT_AND_ACTIVATION_SPEC_03.15.2026.md` §3.2-3.8

---

## Situation

Phase 1 (graph crystallizer) is complete — real graph propagation, semantic k-NN fallback, task-type tuning. All benchmarks hold. The crystallizer diagnostic proved that **post-swarm refinement alone doesn't move LHE** because the bottleneck is upstream (Galaxy content, not graph mechanics).

This phase activates the remaining 6 stub kernels in priority order, grouped by integration point and benchmark impact.

**Current benchmark baseline (navigate=1, strength=0.5):**

| Benchmark | Score | Time |
|-----------|-------|------|
| ARC 10 | 10/10 | ~4s |
| Math 20 | 20/20 | ~8s |
| GSM8K 10 | 1/10 | ~9s |
| LHE 10 | 6/10 | ~5.5s |
| MMLU 50 | 16/50 | ~36s |

---

## Priority Order

### Batch A: Post-Swarm Composition (Immediate — Score Movers)

These operate on data already in the pipeline. No new data needed.

| # | Kernel | Integration | Target Benchmark | Why First |
|---|--------|-------------|-----------------|-----------|
| 1 | `gre_resonance_field` | Post-swarm | MMLU, LHE | Cross-galaxy agreement scoring |
| 2 | `gre_vector_resonator` | Post-swarm | All | Attention-weighted blending (replaces fixed lerp) |
| 3 | `gre_atomic_fission_fusion` | Post-swarm | GSM8K, Math | Compositional consistency check |

### Batch B: Pre-Swarm Feature Extraction (After A — Needs Integration Work)

These require threading new feature vectors through the pipeline.

| # | Kernel | Integration | Target Benchmark | Why Second |
|---|--------|-------------|-----------------|-----------|
| 4 | `gre_geometry_router` | Pre-swarm | MMLU, ARC | Spatial relationship detection |
| 5 | `gre_temporal_reasoning` | Pre-swarm | LHE, GSM8K | Sequence pattern features |
| 6 | `gre_fractal_emitter` | Post-swarm | ARC | Self-similarity scoring |

---

## Batch A Specifications

### A.1: `gre_resonance_field` — Cross-Galaxy Interference

**Current stub:** `sqrt(x²+y²+z²) * density` — distance times scalar.

**Real design:** Score candidates by cross-galaxy agreement. Candidates supported by entries from OTHER galaxies get boosted; conflicting cross-galaxy candidates get attenuated.

**File:** `knowledge3d/cranium/kernels/gre_resonance_field.cu`

**New signature:**
```c
extern "C" __global__ void gre_resonance_field(
    const float* __restrict__ candidates,    // [N × D] candidate embeddings
    const int* __restrict__ galaxy_ids,      // [N] galaxy index per candidate
    const float* __restrict__ base_scores,   // [N] pre-existing scores
    float* __restrict__ resonance_scores,    // [N] output: interference-adjusted
    int N,
    int D
)
```

**Kernel logic (real CUDA):**
```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (; i < N; i += stride) {
    float base = base_scores[i];
    int my_galaxy = galaxy_ids[i];

    // Compute cross-galaxy resonance
    float constructive = 0.0f;
    float destructive = 0.0f;
    int cross_count = 0;

    // Self-norm for cosine similarity
    float self_norm_sq = 0.0f;
    for (int d = 0; d < D; d++) {
        float v = candidates[i * D + d];
        self_norm_sq += v * v;
    }
    float self_norm_inv = (self_norm_sq > 1e-12f) ? rsqrtf(self_norm_sq) : 0.0f;

    for (int j = 0; j < N; j++) {
        if (j == i || galaxy_ids[j] == my_galaxy) continue;

        // Cosine similarity
        float dot = 0.0f, other_norm_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            float a = candidates[i * D + d];
            float b = candidates[j * D + d];
            dot += a * b;
            other_norm_sq += b * b;
        }
        float other_norm_inv = (other_norm_sq > 1e-12f) ? rsqrtf(other_norm_sq) : 0.0f;
        float sim = dot * self_norm_inv * other_norm_inv;

        if (sim > 0.3f) {
            constructive += sim * base_scores[j];
        } else if (sim < -0.2f) {
            destructive += fabsf(sim) * base_scores[j];
        }
        cross_count++;
    }

    if (cross_count > 0) {
        constructive /= (float)cross_count;
        destructive /= (float)cross_count;
    }

    // Resonance = base * (1 + boost - attenuation)
    resonance_scores[i] = base * (1.0f + 0.3f * constructive - 0.15f * destructive);
}
```

**Bridge update in `sovereign_bridges.py`:**

The existing `ResonanceField` bridge has method `compute(positions, density)`. Replace with:
```python
def compute_resonance(self, candidate_embeddings, galaxy_ids, base_scores):
    """Cross-galaxy interference scoring."""
    # GPU: upload embeddings [N×D], galaxy_ids [N], scores [N]
    # Launch gre_resonance_field kernel
    # Download resonance_scores [N]
    # Return as list[float]
```

Keep `compute()` as compatibility wrapper (positions=embeddings, density=scores, ignore galaxy_ids).

**Hot-path integration in `_apply_specialist_swarm_features()`:**

Currently at line ~5168, `galaxy_resonance.resonate_list()` is called. The resonance FIELD is different from the resonance ENGINE. Add a NEW call after crystallization (around line 5263):

```python
# After crystallization, before scoring
resonance_field = self.get_resonance_field()
if resonance_field is not None and len(local_candidates) > 1:
    galaxy_ids = [
        self._gpu_galaxy_index(str(c["match"].get("galaxy", "")))
        for c in local_candidates
    ]
    base_scores = coherence_scores  # from crystallizer output
    try:
        adjusted = resonance_field.compute_resonance(
            crystallized_rows, galaxy_ids, base_scores
        )
        for c, score in zip(local_candidates, adjusted):
            c["cross_galaxy_resonance"] = float(score)
        applied_kernels.append("gre_resonance_field")
    except Exception:
        pass
```

**Benchmark expectation:** MMLU questions that span multiple knowledge domains (science+math, history+geography) get cross-galaxy boosting. May shift 1-3 questions.

---

### A.2: `gre_vector_resonator` — Attention-Weighted Blending

**Current stub:** `a*α + b*(1-α)` — fixed-alpha lerp.

**Real design:** Compute blending weights from the vectors themselves using energy × cross-relevance scoring, then softmax attention.

**File:** `knowledge3d/cranium/kernels/gre_vector_resonator.cu`

**New signature:**
```c
extern "C" __global__ void gre_vector_resonator(
    const float* __restrict__ vectors,      // [K × D] input vectors to blend
    float* __restrict__ blended,            // [D] output blended vector
    float* __restrict__ attention_weights,  // [K] output attention weights
    int K,
    int D
)
```

**Kernel logic:**
```c
// Phase 1: Compute per-vector relevance scores
// Single block, K threads (K <= 32 typically)
int k = threadIdx.x;
if (k >= K) return;

__shared__ float scores[32];  // max K=32
__shared__ float max_score;
__shared__ float sum_exp;

// Energy: self dot product
float energy = 0.0f;
for (int d = 0; d < D; d++) {
    float v = vectors[k * D + d];
    energy += v * v;
}
energy = sqrtf(energy + 1e-12f);

// Cross-relevance: mean cosine with other vectors
float cross = 0.0f;
for (int j = 0; j < K; j++) {
    if (j == k) continue;
    float dot = 0.0f, norm_j = 0.0f;
    for (int d = 0; d < D; d++) {
        dot += vectors[k * D + d] * vectors[j * D + d];
        float vj = vectors[j * D + d];
        norm_j += vj * vj;
    }
    cross += dot / (energy * sqrtf(norm_j + 1e-12f) + 1e-12f);
}
if (K > 1) cross /= (float)(K - 1);

scores[k] = energy * (1.0f + fmaxf(cross, 0.0f));
__syncthreads();

// Softmax
if (k == 0) {
    max_score = scores[0];
    for (int i = 1; i < K; i++)
        if (scores[i] > max_score) max_score = scores[i];
}
__syncthreads();

float exp_val = expf(scores[k] - max_score);
atomicAdd(&sum_exp, exp_val);  // or use shared reduction
__syncthreads();

float w = exp_val / (sum_exp + 1e-12f);
attention_weights[k] = w;

// Phase 2: Weighted blend (each thread contributes its weighted vector)
for (int d = 0; d < D; d++) {
    atomicAdd(&blended[d], w * vectors[k * D + d]);
}
```

**Bridge update:** The existing `VectorResonator` bridge at ~line 631 in sovereign_bridges.py has `resonate(a, b, alpha)`. Replace core logic but keep the 2-vector API as a special case of K=2.

Add `resonate_attention(vectors: list[list[float]]) -> tuple[list[float], list[float]]` that returns (blended, weights).

**Hot-path integration:** The vector resonator is already called at line ~5157:
```python
focus_vector = resonator.resonate_list(focus_vector, lead_embedding, alpha=...)
```

This can stay as-is for now (K=2 case). The attention variant becomes valuable when blending multiple path candidates:

```python
# In _dispatch_swarm_weights or _score_gpu_candidates_batch
# When multiple reasoning paths contribute, blend with attention
```

**Benchmark expectation:** Modest across all benchmarks. The attention mechanism replaces fixed alpha with data-dependent weighting.

---

### A.3: `gre_atomic_fission_fusion` — Compositional Consistency

**Current stub:** `val * ratio` or `val / ratio`.

**Real design:** Fission decomposes a compound embedding into atomic projections + residual. Fusion verifies that atoms compose coherently. Both output a consistency score.

**File:** `knowledge3d/cranium/kernels/gre_atomic_fission_fusion.cu`

**New signature:**
```c
extern "C" __global__ void gre_atomic_fission_fusion(
    const float* __restrict__ compound,     // [D] compound embedding
    const float* __restrict__ atoms,        // [K × D] atomic embeddings
    float* __restrict__ result,             // [D] reconstructed/fused result
    float* __restrict__ consistency,        // [1] consistency score 0-1
    int K,
    int D,
    int mode                                // 0=fission, 1=fusion
)
```

**Kernel logic (single block, serialized for correctness with small K):**
```c
// Thread 0 does all work (K is small, typically 2-8)
if (threadIdx.x != 0) return;

if (mode == 0) {
    // FISSION: decompose compound onto atom directions
    float residual[MAX_D];  // stack or shared memory
    for (int d = 0; d < D; d++) residual[d] = compound[d];

    for (int k = 0; k < K; k++) {
        float dot = 0.0f, norm = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += residual[d] * atoms[k * D + d];
            norm += atoms[k * D + d] * atoms[k * D + d];
        }
        float coeff = dot / (norm + 1e-12f);
        for (int d = 0; d < D; d++) {
            residual[d] -= coeff * atoms[k * D + d];
        }
    }

    // Consistency = 1 - |residual| / |compound|
    float res_norm = 0.0f, comp_norm = 0.0f;
    for (int d = 0; d < D; d++) {
        res_norm += residual[d] * residual[d];
        comp_norm += compound[d] * compound[d];
        result[d] = compound[d] - residual[d];  // reconstruction
    }
    *consistency = 1.0f - sqrtf(res_norm) / (sqrtf(comp_norm) + 1e-12f);

} else {
    // FUSION: verify atoms compose coherently
    // Centroid
    float centroid[MAX_D];
    for (int d = 0; d < D; d++) {
        centroid[d] = 0.0f;
        for (int k = 0; k < K; k++)
            centroid[d] += atoms[k * D + d];
        centroid[d] /= (float)K;
    }

    // Per-atom similarity to centroid → weights
    float weights[MAX_K], weight_sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float dot = 0.0f, a_norm = 0.0f, c_norm = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += atoms[k * D + d] * centroid[d];
            a_norm += atoms[k * D + d] * atoms[k * D + d];
            c_norm += centroid[d] * centroid[d];
        }
        weights[k] = fmaxf(dot / (sqrtf(a_norm * c_norm) + 1e-12f), 0.0f);
        weight_sum += weights[k];
    }

    // Weighted combination
    for (int d = 0; d < D; d++) {
        result[d] = 0.0f;
        for (int k = 0; k < K; k++)
            result[d] += (weights[k] / (weight_sum + 1e-12f)) * atoms[k * D + d];
    }

    *consistency = weight_sum / (float)K;  // mean atom-centroid agreement
}
```

**Note on MAX_D:** Feature dim is 16 (embedding16). Use `__shared__` or register arrays. 16 floats = 64 bytes — fits in registers trivially. Set `MAX_D = 128` and `MAX_K = 16` to be safe.

**Bridge update:** Replace the multiply/divide in `AtomicFissionFusion.transform()` with `decompose(compound, atoms)` and `compose(atoms)`. Keep `transform()` as compatibility.

**Hot-path integration:** In the candidate scoring path, use fission to verify that a compound answer (e.g., GSM8K multi-step result) is consistent with its component Galaxy entries:

```python
# After swarm selects top candidate
# If candidate was composed from multiple Galaxy entries, verify consistency
if len(candidate_atoms) > 1:
    compound = candidate_embedding
    atoms = [atom["embedding16"] for atom in candidate_atoms]
    _, consistency = fission_fusion.decompose(compound, atoms)
    candidate["compositional_consistency"] = consistency
```

**Benchmark expectation:** GSM8K (word problems with multiple steps). Currently 1/10 — if the answer is a compound of steps, consistency checking can break ties between plausible but incorrect compositions.

---

## Implementation Order for Batch A

```
A.1: gre_resonance_field
  1. Replace .cu with real cross-galaxy interference kernel
  2. Recompile .ptx
  3. Update ResonanceField bridge: compute_resonance() + compatibility
  4. Wire into _apply_specialist_swarm_features after crystallization
  5. Run full benchmark sweep — all must hold

A.2: gre_vector_resonator
  1. Replace .cu with attention-weighted blending kernel
  2. Recompile .ptx
  3. Update VectorResonator bridge: resonate_attention() + keep resonate()
  4. The existing hot-path call already uses resonate() — no integration needed
  5. Run full benchmark sweep

A.3: gre_atomic_fission_fusion
  1. Replace .cu with compositional fission/fusion kernel
  2. Recompile .ptx
  3. Update AtomicFissionFusion bridge: decompose() + compose() + compatibility
  4. Wire into candidate scoring for multi-step answers
  5. Run full benchmark sweep
```

**After each kernel:** Run the FULL benchmark sweep (ARC, Math, GSM8K, LHE, MMLU). Pinned benchmarks MUST hold. If any regress, disable the new kernel and diagnose.

---

## Batch B: Deferred to After Batch A Results

Batch B (geometry_router, temporal_reasoning, fractal_emitter) requires threading new feature data through the pipeline — more plumbing. Design those after seeing Batch A's benchmark impact. If Batch A moves MMLU or GSM8K, Batch B builds on that signal. If Batch A is flat, we pivot to Direction B (GSM8K/MMLU content enrichment).

---

## Files to Touch

**Batch A:**
- `knowledge3d/cranium/kernels/gre_resonance_field.cu` → real kernel
- `knowledge3d/cranium/kernels/gre_resonance_field.ptx` → recompile
- `knowledge3d/cranium/kernels/gre_vector_resonator.cu` → real kernel
- `knowledge3d/cranium/kernels/gre_vector_resonator.ptx` → recompile
- `knowledge3d/cranium/kernels/gre_atomic_fission_fusion.cu` → real kernel
- `knowledge3d/cranium/kernels/gre_atomic_fission_fusion.ptx` → recompile
- `knowledge3d/cranium/bridges/sovereign_bridges.py` → bridge updates
- `knowledge3d/knowledgeverse/knowledgeverse.py` → hot-path integration
- `tests/test_all_sovereign_bridges.py` → bridge tests
- `tests/test_trm_weight_persistence.py` → integration tests

**Do NOT touch:**
- `nine_chain_specialized.cu` — the swarm is working, leave it alone
- `gre_multimodal_halting_gate.cu` — convergence check is working
- `gre_graph_crystallizer.cu` — just finished, stable

---

## Success Criteria

1. All 3 Batch A stubs replaced with real CUDA (no multiply-by-scalar, no EMA, no lerp)
2. Each kernel has ≥1 bridge test verifying non-trivial computation
3. All pinned benchmarks hold: ARC 10/10, Math 20/20
4. Any benchmark movement is upward (GSM8K, LHE, MMLU)
5. No new Python in the hot path — all computation sovereign (GPU kernel)
