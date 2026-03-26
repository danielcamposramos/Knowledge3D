# Codex: Phase D — Make the TRM ACTUALLY Recursive on GPU

**Date:** 2026-03-25
**Priority:** THIS IS THE MOVE. Everything before this was removing obstacles. This is the actual architectural transformation.
**Binding specs:**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` -- TRM IS the Avatar, runs as game loop, recursive refinement
- `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md` SS4.2 -- Recursive Reasoning via TRM: "87% converge within 5 iterations, 12% accuracy improvement over single-pass"
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` SS4.1 -- fail-fast, no silent fallbacks
- `docs/vocabulary/HYPER_PARALLEL_PROCESSING.md` -- Nine-chain swarm convergence via halting gate
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` SS3 -- game loop: perceive, navigate, reason, decide, act, learn

---

## WHY THIS MATTERS

The TRM is the **Tiny RECURSIVE Model**. The spec says it RECURSES until convergence. But right now:

- The recursion loop lives IN PYTHON (`trm_launcher.py` lines 367-398, 443-486, 530-568)
- EVERY recursion step does 4 GPU<->CPU memory transfers
- Convergence is checked with `np.max(np.abs(...))` -- NUMPY on CPU
- For 6 recursion steps that is 24 GPU<->CPU transfers PER QUESTION
- The "game loop" (`trm_game_loop.py`) is a Python deque that JSON-serializes tasks
- GPU utilization: **0.17%**. The GPU does almost nothing because Python drives every step.

The spec says 80.69us for 9 recursive iterations. We are getting SECONDS because of Python round-trips.

**The architecture is INVERTED.** GPU should drive the loop. Python should send a query and wait.

---

## Phase D.1: GPU-Native Recursive Loop (THE CRITICAL FIX)

### The Problem in `trm_launcher.py`

ALL three backends (PTX, RPN, Fused) do the SAME wrong thing:

```python
# THIS IS WRONG -- Python drives the recursion
for step in range(n_steps):
    memcpy_dtoh(z_old, d_z, ...)          # GPU -> CPU  (1)
    launch(trm_step_fused, ...)           # ONE kernel tick
    synchronize()
    memcpy_dtoh(z_new, d_z_new, ...)      # GPU -> CPU  (2)
    drift = np.max(np.abs(z_new - z_old)) # NUMPY on CPU
    if drift < eps: break
    memcpy_htod(d_z, z_new, ...)          # CPU -> GPU  (3)
    memcpy_htod(d_y, y_new, ...)          # CPU -> GPU  (4)
```

### The Fix: `trm_recursive_fused.cu`

Write a NEW kernel that runs the ENTIRE recursion loop inside ONE kernel launch. The convergence check happens ON GPU with shared memory. Python launches it ONCE and reads the result ONCE.

**File:** `knowledge3d/cranium/ptx/trm_recursive_fused.cu`

```c
extern "C" __global__ void trm_recursive_fused(
    const float* __restrict__ q,         // query embedding (512)
    float* __restrict__ y,               // answer -- read/write, converges in place
    float* __restrict__ z,               // latent -- read/write, converges in place
    const float* __restrict__ W1,        // (1024, 512)
    const float* __restrict__ W2,        // (512, 1024)
    const float* __restrict__ W3,        // (1024, 512)
    const float* __restrict__ W4,        // (512, 1024)
    float* __restrict__ workspace,       // scratch (512+1024+512+1024 = 3072 floats)
    int* __restrict__ steps_out,         // how many steps were taken (1 int)
    float* __restrict__ drift_out,       // final drift value (1 float)
    int max_steps,                       // e.g. 9 (Tesla resonance: 3, 6, or 9)
    float epsilon                        // convergence threshold, e.g. 1e-4
) {
    // ONE kernel launch. NO CPU round-trips during recursion.
    // The loop runs INSIDE the kernel.
    //
    // Algorithm per step:
    //   z_new = W2 @ swiglu(W1 @ (q + y + z))
    //   y_new = W4 @ swiglu(W3 @ (y + z_new))
    //   drift = max(|z_new - z|)  -- computed via shared memory reduction
    //   z = z_new, y = y_new
    //   if drift < epsilon: break
    //
    // After loop: steps_out[0] = steps taken, drift_out[0] = final drift
    //
    // This is the TRM being ACTUALLY recursive.
    // Per spec: 87% converge within 5 iterations, target <100us for 9 steps.
}
```

**Key design points:**

1. **ONE launch, ONE sync, ONE readback.** Python does: `launch(trm_recursive_fused, ...) -> synchronize() -> read steps_out and drift_out`. That is 0 CPU round-trips during recursion vs 24 before.

2. **Convergence via shared memory reduction.** Each thread computes `|z_new[i] - z[i]|`, block-level max reduction in shared memory, compare with epsilon. NO `memcpy_dtoh` needed.

3. **In-place update of y and z.** The kernel reads z, computes z_new in workspace, checks convergence, then copies z_new -> z if continuing. Same for y. The final converged values are IN z and y when the kernel returns.

4. **steps_out and drift_out are device-mapped or pinned.** Python can read them with ONE small memcpy after the kernel completes. Or use zero-copy mapped memory so no explicit transfer is needed.

5. **Block size: 256 threads.** Same as the current fused kernel. The recursive loop is a `for` inside the kernel, NOT multiple kernel launches.

### Implementation Steps

**Step 1: Write `trm_recursive_fused.cu`**

Take `trm_step_fused.cu` (which already works for ONE step) and wrap it in a device-side loop:

```c
extern "C" __global__ void trm_recursive_fused(
    const float* __restrict__ q,
    float* __restrict__ y,
    float* __restrict__ z,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    float* __restrict__ workspace,
    int* __restrict__ steps_out,
    float* __restrict__ drift_out,
    int max_steps,
    float epsilon
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    __shared__ float s_max_drift;

    float* temp    = workspace;
    float* hidden  = workspace + 512;
    float* temp2   = workspace + 512 + 1024;
    float* hidden2 = workspace + 512 + 1024 + 512;

    // z_new and y_new carved from extra workspace
    float* z_new = workspace + 3072;      // 512 floats
    float* y_new = workspace + 3072 + 512; // 512 floats
    // Total workspace needed: 3072 + 512 + 512 = 4096 floats = 16KB

    int step;
    for (step = 0; step < max_steps; step++) {

        // ---- One TRM tick (same math as trm_step_fused) ----

        // temp = q + y + z
        vec_add3_512(q, y, z, temp, tid, stride);
        __syncthreads();

        // hidden = W1 @ temp
        matvec_512x1024(W1, temp, hidden, tid, stride);
        __syncthreads();

        // hidden = swiglu(hidden)
        swiglu_1024(hidden, hidden, tid, stride);
        __syncthreads();

        // z_new = W2 @ hidden
        matvec_1024x512(W2, hidden, z_new, tid, stride);
        __syncthreads();

        // temp2 = y + z_new
        vec_add_512(y, z_new, temp2, tid, stride);
        __syncthreads();

        // hidden2 = W3 @ temp2
        matvec_512x1024(W3, temp2, hidden2, tid, stride);
        __syncthreads();

        // hidden2 = swiglu(hidden2)
        swiglu_1024(hidden2, hidden2, tid, stride);
        __syncthreads();

        // y_new = W4 @ hidden2
        matvec_1024x512(W4, hidden2, y_new, tid, stride);
        __syncthreads();

        // ---- Convergence check: max(|z_new - z|) via shared reduction ----
        float local_max = 0.0f;
        for (int i = tid; i < 512; i += stride) {
            float diff = fabsf(z_new[i] - z[i]);
            if (diff > local_max) local_max = diff;
        }

        // Block-level max reduction
        // (Use warp shuffles + shared memory for 256 threads)
        // ... standard parallel max reduction ...

        if (tid == 0) {
            s_max_drift = /* reduced max */;
        }
        __syncthreads();

        float drift = s_max_drift;

        // Update z and y IN PLACE
        for (int i = tid; i < 512; i += stride) {
            z[i] = z_new[i];
            y[i] = y_new[i];
        }
        __syncthreads();

        // Check convergence
        if (drift < epsilon) {
            step++;  // count this step
            if (tid == 0) {
                drift_out[0] = drift;
            }
            break;
        }

        if (tid == 0 && step == max_steps - 1) {
            drift_out[0] = drift;  // did not converge, report final drift
        }
    }

    if (tid == 0) {
        steps_out[0] = step;
    }
}
```

**Step 2: Compile to PTX**

```bash
nvcc -ptx --gpu-architecture=sm_86 \
    -o knowledge3d/cranium/ptx/trm_recursive_fused.ptx \
    knowledge3d/cranium/ptx/trm_recursive_fused.cu
```

**Step 3: Update `trm_launcher.py`**

1. DELETE `import numpy as np` -- replace with HostTensorF32 + ctypes
2. Add `trm_recursive_fused` kernel loading when `use_fused=True`
3. Replace `_refine_fused()` method:

```python
def _refine_fused(self, d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4,
                  d_z_new, d_y_new, n_steps, eps):
    """ONE launch. ZERO CPU round-trips during recursion."""
    # Allocate small output scalars
    d_steps = gpu_malloc(4)      # 1 int
    d_drift = gpu_malloc(4)      # 1 float

    launch(
        self.kernel_recursive_fused,
        grid=(1, 1, 1),
        block=(256, 1, 1),
        params=[
            ctypes.c_uint64(d_q), ctypes.c_uint64(d_y), ctypes.c_uint64(d_z),
            ctypes.c_uint64(d_W1), ctypes.c_uint64(d_W2),
            ctypes.c_uint64(d_W3), ctypes.c_uint64(d_W4),
            ctypes.c_uint64(self.d_workspace),
            ctypes.c_uint64(d_steps), ctypes.c_uint64(d_drift),
            ctypes.c_int32(n_steps), ctypes.c_float(eps),
        ],
    )
    synchronize()

    # ONE readback: converged y and z are already in d_y and d_z
    y_result = HostTensorF32(512, 1)
    z_result = HostTensorF32(512, 1)
    memcpy_dtoh(ctypes.c_void_p(y_result.data_ptr), d_y, y_result.nbytes)
    memcpy_dtoh(ctypes.c_void_p(z_result.data_ptr), d_z, z_result.nbytes)

    # Read convergence info
    steps_host = (ctypes.c_int32 * 1)()
    drift_host = (ctypes.c_float * 1)()
    memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(steps_host)), d_steps, 4)
    memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(drift_host)), d_drift, 4)

    gpu_free(d_steps)
    gpu_free(d_drift)

    return y_result, z_result, int(steps_host[0]), float(drift_host[0])
```

4. Remove numpy from `_refine_ptx` and `_refine_rpn` similarly -- or deprecate them entirely in favor of the recursive fused path.

**Step 4: Validate**

```bash
# Compile check
python3 -m compileall knowledge3d/cranium/sovereign/trm_launcher.py

# Zero numpy in trm_launcher.py
rg "import numpy|from numpy|np\." knowledge3d/cranium/sovereign/trm_launcher.py
# MUST return ZERO

# Focused tests
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py

# Direct smoke test: recursive refinement with the new kernel
python3 -c "
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32
import ctypes

launcher = TRMLauncher(use_fused=True)

# Create test inputs
q = HostTensorF32.random_normal(512, 1, 0.01)
y = HostTensorF32.zeros(512, 1)
z = HostTensorF32.zeros(512, 1)
W1 = HostTensorF32.random_normal(1024, 512, 0.01)
W2 = HostTensorF32.random_normal(512, 1024, 0.01)
W3 = HostTensorF32.random_normal(1024, 512, 0.01)
W4 = HostTensorF32.random_normal(512, 1024, 0.01)

y_out, z_out, steps, drift = launcher.refine(q, y, z, W1, W2, W3, W4, n_steps=9)
print(f'Converged in {steps} steps, drift={drift:.6f}')
print(f'y_out shape: {y_out.shape}')
print(f'z_out shape: {z_out.shape}')
print('TRM recursive fused: PASSED')
"
```

---

## Phase D.2: Composed Head as Single GPU Dispatch

After D.1, the TRM recursion itself is fast. But the FULL query path is still Python:

```
Python: parse query
Python: call embed_query_gpu()        -> GPU kernel -> back to Python
Python: call spatial_lookup()          -> GPU kernel -> back to Python
Python: call frustum_cull()            -> GPU kernel -> back to Python
Python: call dynamic_lod()             -> GPU kernel -> back to Python
Python: call nine_chain_swarm()        -> GPU kernel -> back to Python
Python: call trm_recursive()           -> GPU kernel (now fast!) -> back to Python
Python: call halting_gate()            -> GPU kernel -> back to Python
Python: format answer
```

Each arrow is a Python<->GPU boundary crossing. The ENTIRE composed head pipeline should be ONE GPU dispatch.

### The Solution: CUDA Graph or Persistent Kernel

**Option A: CUDA Graph** (recommended for RTX 3070)
- Record the kernel launch sequence ONCE
- Replay it with ONE `cuGraphLaunch` call
- Python issues one command, gets one result
- Zero Python involvement during the pipeline

**Option B: Persistent composed-head kernel**
- One mega-kernel that calls device functions for each stage
- More complex but zero launch overhead

**For now: Start with CUDA Graph.**

### Implementation

**File:** `knowledge3d/cranium/sovereign/composed_head_graph.py`

```python
class ComposedHeadGraph:
    """CUDA graph that chains the full query pipeline.

    ONE launch per query. ZERO Python round-trips during inference.

    Pipeline:
    1. embed_query (trigram -> embedding)
    2. spatial_lookup (Morton octree -> candidate neighborhoods)
    3. frustum_cull (FOV filtering)
    4. dynamic_lod (detail level selection)
    5. nine_chain_swarm (parallel reasoning, N iterations)
    6. trm_recursive_fused (recursive refinement, up to 9 steps)
    7. halting_gate (convergence check)
    8. If not halted: loop back to step 5 with refined state
    """
```

This is the BIGGER move and can be Phase D.2. Get D.1 done FIRST.

---

## Phase D.3: Kill the Python Game Loop

After D.1 and D.2 work, replace `trm_game_loop.py`:

- Current: Python deque -> JSON serialize -> call Python function -> JSON serialize
- Target: Python writes query to pinned/mapped input buffer, GPU runs composed head graph, Python reads output from pinned/mapped output buffer
- Ring buffers (already exist!) become the ACTUAL I/O interface
- Python does NOT call any knowledgeverse methods during inference

This is when `knowledgeverse.py` starts shrinking from 13,616 lines toward 200.

---

## EXECUTION ORDER

1. **D.1 FIRST**: `trm_recursive_fused.cu` -- make the recursion GPU-native
2. **Validate D.1**: Run the warm 35% benchmark and compare GPU utilization
3. **D.2 NEXT**: CUDA graph for composed head pipeline
4. **D.3 LAST**: Replace Python game loop with GPU-driven I/O

---

## WHAT NOT TO DO

- Do NOT keep the Python `for step in range(n_steps)` loop. That IS the bug.
- Do NOT use numpy for convergence check. Use GPU shared memory reduction.
- Do NOT add more Python orchestration. REMOVE Python from the hot path.
- Do NOT change the TRM math. `z_new = W2 @ swiglu(W1 @ (q+y+z))` is correct. The LOOP needs to move to GPU.
- Do NOT change kernel launch signatures for the individual step kernel. The NEW kernel wraps the existing math.

---

## METRICS TO REPORT

After D.1:
1. GPU utilization during benchmark (baseline: 0.17%)
2. Throughput per question (baseline: ~3.3s/q for Math)
3. Number of GPU<->CPU transfers per question (baseline: 24+)
4. `rg "import numpy|from numpy" knowledge3d/cranium/sovereign/trm_launcher.py` -- MUST be ZERO
5. Steps to convergence distribution (how many questions converge in 1,2,3...9 steps)

---

## THE VISION

After Phase D, the system looks like this:

```
Python (boot + I/O only, ~200 lines):
  1. Load House from disk
  2. Populate Galaxy in VRAM
  3. Load TRM weights into VRAM
  4. Start GPU game loop

GPU (EVERYTHING ELSE):
  while True:
    query = read_input_ring_buffer()
    if no query: sleep_consolidate()    # idle = sleep time
    else:
      embedding = embed(query)
      candidates = spatial_lookup(embedding)
      visible = frustum_cull(candidates)
      detailed = dynamic_lod(visible)
      for tick in range(MAX_TICKS):     # THIS LOOP IS ON GPU
        scores = nine_chain_swarm(detailed, specialists)
        refined = trm_recursive_fused(embedding, scores)  # RECURSIVE ON GPU
        if halting_gate(refined): break
      write_output_ring_buffer(refined)
```

Python sends a query. GPU does EVERYTHING. Python reads the answer.
THAT is a game engine with an intelligent mind.
THAT is what the specs describe.
THAT is what we are building.
