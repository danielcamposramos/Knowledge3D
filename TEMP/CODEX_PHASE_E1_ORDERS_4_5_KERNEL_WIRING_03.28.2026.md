# Codex — Phase E.1 ORDERS 4-5: Kernel Wiring

**Date:** 2026-03-28
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER — NOT A DISCUSSION. BUILD THIS.
**Prerequisite:** Orders 1-3 are DONE. `vram_task_buffer.py`, `gpu_task_dispatch.cu`, `gpu_task_dispatch.py`, `run_gpu_benchmark.py` exist and pass tests.

---

## READ FIRST

The current `gpu_task_dispatch.cu` is a COSINE SIMILARITY PLACEHOLDER. It works, it's correct, but it does NOT use the composed-head pipeline. These two orders wire the REAL pipeline into the kernel.

**Do NOT:**
- Add Python code to the hot path
- Create new Python wrapper functions
- Add asyncio, threading, multiprocessing
- Write analysis documents
- Add JSON serialization between GPU stages
- Add Python fallbacks
- Modify `vram_task_buffer.py` slot layout (it's correct)

**DO:**
- Extract `__device__` functions from existing `__global__` kernels
- Wire them into `gpu_task_dispatch.cu`
- Keep data in registers/shared memory between stages
- Test with `run_gpu_benchmark.py --suite synthetic --count 10`

---

## ORDER 4: Extract `__device__` Functions from Pipeline Kernels

Each existing kernel is a `__global__` function. To call them FROM `gpu_task_dispatch.cu`, extract the core logic into a `__device__` function in a header. The `__global__` wrapper stays for standalone use.

### 4A: Create `knowledge3d/cranium/cuda/device_functions.cuh`

This header collects ALL device-callable pipeline stages. For each kernel below, extract core logic into a `__device__` function.

### 4B: Nine-Chain Swarm → `__device__`

**Source:** `knowledge3d/cranium/kernels/nine_chain_swarm_kernel.cu`
**Current signature:**
```c
extern "C" __global__ void nine_chain_swarm_kernel(
    const float* __restrict__ input_embedding,
    float* __restrict__ chain_states,
    float* __restrict__ output_embedding,
    float* __restrict__ resonance_scores,
    int num_iterations
)
```

**Extract to:**
```c
__device__ void nine_chain_swarm_device(
    const float* query_embedding,    // 32 floats from input slot
    float* chain_states,             // shared memory: 9 × 32 floats
    float* output_embedding,         // 32 floats result
    float* resonance_scores,         // 9 floats
    int num_iterations               // from ARB budget or default 3
)
```

**Key:** The swarm uses `blockIdx.x` for chain_id. In the device function, use a loop over 9 chains (or launch 9 cooperative groups from the thread block). The thread block assigned to each task in `gpu_task_dispatch` has 128 threads — plenty for 9 chains of 32 dims.

### 4C: Halting Gate → `__device__`

**Source:** `knowledge3d/cranium/kernels/gre_multimodal_halting_gate.cu`
**Current signature:**
```c
extern "C" __global__ void gre_multimodal_halting_gate(
    const float* __restrict__ scores_ptr,
    const unsigned int* __restrict__ candidate_hash_ptr,
    unsigned int* __restrict__ flags_ptr,
    float* __restrict__ metrics_ptr,
    unsigned int length,
    float minimum_threshold,
    float gap_threshold,
    float agreement_threshold
)
```

**Extract to:**
```c
__device__ int halting_gate_device(
    const float* scores,         // from swarm resonance_scores (9 floats)
    int num_scores,              // 9 (or option_count)
    float min_threshold,         // 0.1f default
    float gap_threshold,         // 0.15f default
    float agreement_threshold    // 0.7f default
)
// Returns: 1 if converged, 0 if not
```

### 4D: Defeasible Resolver → `__device__`

**Source:** `knowledge3d/cranium/kernels/gre_defeasible_resolver.cu`
**Already has `__device__` helpers:** `quantize_trit`, `clamp_trit_int`, `encode_trit`

**Extract the `__global__` body to:**
```c
__device__ void defeasible_resolve_device(
    const float* conclusions,        // candidate scores
    const int8_t* rule_strengths,    // +1/0/-1 per rule
    float* verdicts,                 // modified scores after defeasible logic
    int num_candidates,
    int num_workers
)
```

### 4E: Wire Into `gpu_task_dispatch.cu`

Replace the current cosine-similarity body with the composed pipeline:

```c
#include "device_functions.cuh"

extern "C" __global__ void gpu_task_dispatch(
    const unsigned char* __restrict__ input_buffer,
    unsigned char* __restrict__ output_buffer,
    unsigned int task_count
) {
    const unsigned int task_id = blockIdx.x;  // ONE block per task now
    if (task_id >= task_count) return;

    const unsigned int input_base = task_id * 1024u;
    const unsigned int output_base = task_id * 512u;

    // --- Read input slot (same offsets as before) ---
    const float* query_embedding = reinterpret_cast<const float*>(input_buffer + input_base + 0u);
    const unsigned int task_type = *reinterpret_cast<const unsigned int*>(input_buffer + input_base + 128u);
    const unsigned int option_count = *reinterpret_cast<const unsigned int*>(input_buffer + input_base + 132u);
    const float* option_embeddings = reinterpret_cast<const float*>(input_buffer + input_base + 136u);

    // --- Shared memory for swarm ---
    __shared__ float chain_states[9 * 32];
    __shared__ float swarm_output[32];
    __shared__ float resonance_scores[9];

    // --- Stage 1: Nine-Chain Swarm on query ---
    nine_chain_swarm_device(query_embedding, chain_states, swarm_output, resonance_scores, 3);
    __syncthreads();

    // --- Stage 2: Cosine similarity (swarm output vs options) ---
    unsigned int best_index = 0u;
    float best_score = -3.402823466e+38F;
    unsigned int bounded_options = option_count > 4u ? 4u : option_count;

    for (unsigned int oi = 0u; oi < bounded_options; ++oi) {
        const float* opt = option_embeddings + (oi * 32u);
        float score = 0.0f;
        for (unsigned int d = threadIdx.x; d < 32u; d += blockDim.x) {
            score += swarm_output[d] * opt[d];
        }
        // Warp reduce score (use __shfl_down_sync or atomicAdd to shared)
        // ... reduction code here ...
        if (score > best_score) {
            best_score = score;
            best_index = oi;
        }
    }

    // --- Stage 3: Halting gate ---
    int converged = halting_gate_device(resonance_scores, 9, 0.1f, 0.15f, 0.7f);

    // --- Stage 4: Write output slot ---
    if (threadIdx.x == 0u) {
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + 0u) = best_index;
        *reinterpret_cast<float*>(output_buffer + output_base + 4u) = best_score;
        *reinterpret_cast<signed char*>(output_buffer + output_base + 8u) = converged ? 1 : 0;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + 12u) = 1u;
        unsigned long long answer_hash =
            (static_cast<unsigned long long>(task_type) << 32) |
            static_cast<unsigned long long>(best_index);
        *reinterpret_cast<unsigned long long*>(output_buffer + output_base + 16u) = answer_hash;
    }
}
```

**IMPORTANT change:** Grid changes from `(ceil(task_count/128), 1, 1)` with `blockDim=(128,1,1)` to `(task_count, 1, 1)` with `blockDim=(128,1,1)`. ONE BLOCK PER TASK, because the swarm needs intra-block cooperation.

**Update `gpu_task_dispatch.py` launch accordingly:**
```python
# OLD: grid_x = (total + block_size - 1) // block_size
# NEW: grid_x = total  (one block per task)
loader.launch(self.kernel, (total, 1, 1), (128, 1, 1), [...])
```

**Update `cpu_reference_dispatch()` to match** — add a simple swarm simulation (mean of 9 perturbed copies of query, then cosine).

### 4F: Test

```bash
python scripts/run_gpu_benchmark.py --suite synthetic --count 10
```

Synthetic tasks must still get 10/10 (cosine between orthogonal one-hot vectors is unambiguous regardless of swarm). Confidence values will differ from pre-swarm baseline — that's expected.

---

## ORDER 5: Wire GRE Specialists by Task Type

### 5A: Add Specialist `__device__` Functions to `device_functions.cuh`

Extract from each kernel file in `knowledge3d/cranium/kernels/`:

| Kernel File | Device Function | Core Operation |
|---|---|---|
| `gre_arc_reasoner.cu` | `arc_reason_device()` | — |
| `gre_geometry_router.cu` | `geometry_route_device()` | Pairwise embedding relations |
| `gre_fractal_emitter.cu` | `fractal_emit_device()` | Self-similarity across scales |
| `gre_atomic_fission_fusion.cu` | `atomic_fission_fusion_device()` | Compound↔atom decomposition |
| `gre_temporal_reasoning.cu` | `temporal_reason_device()` | Sequence pattern detection |
| `gre_graph_crystallizer.cu` | `graph_crystallize_device()` | GNN-style neighbor aggregation |
| `gre_resonance_field.cu` | `resonance_field_device()` | Galaxy-aware score boosting |
| `gre_vector_resonator.cu` | `vector_resonate_device()` | Attention-weighted blending |
| `gre_cognitive_executive.cu` | `cognitive_executive_device()` | Trust weighting + coherence |

**Note:** `gre_oom_spill.cu` and `gre_world_model.cu` are NOT wired into the dispatch — they serve other purposes (memory management and ARC-AGI-3 world modeling respectively).

### 5B: Add Task-Type Dispatch Switch in `gpu_task_dispatch.cu`

Between Stage 1 (swarm) and Stage 2 (cosine), add specialist activation:

```c
// --- Stage 1.5: Specialist activation by task type ---
// Task type IDs from vram_task_buffer.py:
//   ARC=0, MATH=1, GSM8K=2, LHE=3, MMLU=4, CHAT=5, GENERAL=6, GRAMMAR=7

switch (task_type) {
    case 0u:  // ARC
        arc_reason_device(swarm_output, chain_states, 32);
        geometry_route_device(swarm_output, 32);
        fractal_emit_device(swarm_output, 32);
        break;
    case 1u:  // MATH
        atomic_fission_fusion_device(swarm_output, chain_states, 32);
        geometry_route_device(swarm_output, 32);
        break;
    case 2u:  // GSM8K
        atomic_fission_fusion_device(swarm_output, chain_states, 32);
        temporal_reason_device(swarm_output, 32);
        break;
    case 3u:  // LHE
        graph_crystallize_device(swarm_output, chain_states, 32);
        break;
    case 4u:  // MMLU
        resonance_field_device(swarm_output, 32);
        vector_resonate_device(swarm_output, chain_states, 32);
        break;
    default:  // CHAT, GENERAL, GRAMMAR — base TRM weights only
        break;
}
__syncthreads();

// Then proceed to Stage 2 (cosine similarity with specialist-modified swarm_output)
```

### 5C: Specialist Device Function Contract

Every specialist `__device__` function follows the SAME contract:

```c
__device__ void specialist_name_device(
    float* embedding,        // IN/OUT: the swarm output (32 floats in shared memory)
    const float* context,    // IN: chain_states or option_embeddings (read-only)
    int dim                  // embedding dimension (32)
)
```

The specialist MODIFIES `embedding` in-place. It reads `context` for additional signal. It uses `threadIdx.x` for parallelism within the block. It calls `__syncthreads()` before returning.

This uniform contract means:
- Specialists compose (output of one feeds input of next)
- Adding a new specialist = one `__device__` function + one line in the switch
- No data marshaling between specialists (all in shared memory)

### 5D: Cognitive Executive as Post-Specialist Gate

After the specialist switch and before cosine similarity, add the cognitive executive:

```c
// --- Stage 1.7: Cognitive executive (always runs, all task types) ---
cognitive_executive_device(resonance_scores, chain_states, swarm_output, 32);
__syncthreads();
```

This applies trust weighting and coherence scoring across all specialist outputs before the final cosine comparison.

### 5E: Test

```bash
# Synthetic (all task_type = MMLU since subject is "synthetic_subject")
python scripts/run_gpu_benchmark.py --suite synthetic --count 10

# MMLU (real embeddings, activates resonance_field + vector_resonator)
python scripts/run_gpu_benchmark.py --suite mmlu --count 50 --storage-root /K3D/Knowledge3D.local
```

Expected: Synthetic 10/10 (orthogonal vectors are unambiguous). MMLU accuracy should be ≥ D.3b baseline.

---

## FILE INVENTORY

Files you CREATE:
- `knowledge3d/cranium/cuda/device_functions.cuh` — all `__device__` extractions

Files you MODIFY:
- `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` — wire pipeline + specialists
- `knowledge3d/knowledgeverse/gpu_task_dispatch.py` — update grid launch (one block per task)
- `knowledge3d/knowledgeverse/gpu_task_dispatch.py:cpu_reference_dispatch()` — update to match new kernel semantics

Files you READ (extract `__device__` from):
- `knowledge3d/cranium/kernels/nine_chain_swarm_kernel.cu`
- `knowledge3d/cranium/kernels/gre_multimodal_halting_gate.cu`
- `knowledge3d/cranium/kernels/gre_defeasible_resolver.cu`
- `knowledge3d/cranium/kernels/gre_arc_reasoner.cu` (if exists, else create stub)
- `knowledge3d/cranium/kernels/gre_geometry_router.cu`
- `knowledge3d/cranium/kernels/gre_fractal_emitter.cu`
- `knowledge3d/cranium/kernels/gre_atomic_fission_fusion.cu`
- `knowledge3d/cranium/kernels/gre_temporal_reasoning.cu`
- `knowledge3d/cranium/kernels/gre_graph_crystallizer.cu`
- `knowledge3d/cranium/kernels/gre_resonance_field.cu`
- `knowledge3d/cranium/kernels/gre_vector_resonator.cu`
- `knowledge3d/cranium/kernels/gre_cognitive_executive.cu`

Files you DO NOT TOUCH:
- `knowledge3d/knowledgeverse/vram_task_buffer.py` — slot layout is correct
- `scripts/run_gpu_benchmark.py` — works as-is
- Any Python file in `knowledge3d/knowledgeverse/` (no Python additions to hot path)

---

## EXECUTION SEQUENCE

1. Create `device_functions.cuh` with nine_chain_swarm_device + halting_gate_device + defeasible_resolve_device
2. Wire into `gpu_task_dispatch.cu` (Order 4E)
3. Update `gpu_task_dispatch.py` launch grid
4. Test: `--suite synthetic --count 10` → 10/10
5. Add specialist `__device__` functions to `device_functions.cuh` (Order 5A)
6. Add task-type switch to `gpu_task_dispatch.cu` (Order 5B)
7. Add cognitive executive gate (Order 5D)
8. Test: `--suite synthetic --count 10` → 10/10, then `--suite mmlu --count 50`

---

## SUCCESS CRITERIA

- `gpu_task_dispatch.cu` calls nine_chain_swarm → specialist dispatch → cognitive executive → cosine similarity → halting gate
- ZERO Python in the hot path between `bulk_load()` and `read_results()`
- All specialist kernels wired by task_type switch
- Synthetic benchmark: 10/10
- MMLU benchmark: runs to completion, accuracy ≥ D.3b baseline
- `device_functions.cuh` contains ALL `__device__` extractions in one header

**Build it.**
