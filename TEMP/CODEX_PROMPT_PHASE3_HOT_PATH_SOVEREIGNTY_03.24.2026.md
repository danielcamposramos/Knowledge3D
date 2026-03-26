# Codex Prompt: Phase 3 — Hot-Path Sovereignty (Device-Resident Inference)

**Date:** 2026-03-24
**Priority:** HIGHEST — This is where the 0% GPU utilization lives
**Binding specs (READ THESE FIRST):**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` — "TRM runs as continuous game loop via `trm_step_fused.ptx`." Game NPCs do NOT copy world state to CPU every frame. Their perception, navigation, decision all happen where the world lives — GPU memory.
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §3 — Seven VRAM regions: KERNELS (100MB), GALAXY_UNIVERSE (2GB), HOUSE_CONTEXT (2.5GB), WORLD_VIEW (2GB), TRM_WEIGHTS (400MB), AUDIT_JOURNAL (256MB), INGESTION_STARGATE (512MB). These are NOT Python dicts. They are device-resident memory regions.
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` §3 — "VRAM-native workspace. No CPU preprocessing." The spatial navigation pipeline (Morton, LED-A*, Frustum, LOD) MUST operate on device-resident data.
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §4.1 — "ptx_fallback_rate MUST be 0.0." Every numpy call in the hot path IS a sovereignty violation — it pulls data to host, processes on CPU, then pushes back.

---

## THE PROBLEM: 366 numpy calls in per-question hot path

Phase 1 and 2 fixed the SLEEP-TIME path (contrastive training). But sleep-time fires ONCE per benchmark. The INFERENCE path fires 5954 times. That is where GPU sits at 0%.

Current per-question flow (Python pseudocode of what actually happens):

```python
# 1. EMBED query — numpy stages trigram arrays on host
query_vec = np.array(trigrams)           # rpn_embedding_engine.py
d_query = copy_to_device(query_vec)      # host→device
kernel_trigram_embed(d_query)             # GPU: ~1μs
result = copy_to_host(d_query)           # device→host
embedding = np.array(result)             # back to numpy

# 2. NAVIGATE — numpy stages Morton codes on host
morton_codes = np.array(...)             # morton_octree.py
d_morton = copy_to_device(morton_codes)  # host→device
kernel_morton_query(d_morton)            # GPU: ~1μs
neighbors = copy_to_host(d_morton)       # device→host
neighbor_list = np.array(neighbors)      # back to numpy

# 3. FRUSTUM CULL — numpy stages frustum planes on host
planes = np.array(frustum_params)        # frustum.py
d_planes = copy_to_device(planes)        # host→device
kernel_frustum_cull(d_planes)            # GPU: ~1μs
visible = copy_to_host(d_planes)         # device→host
visible_ids = np.array(visible)          # back to numpy

# 4. LED-A* — numpy stages path graph on host
graph = np.array(edges)                  # led_pathfinder.py
d_graph = copy_to_device(graph)          # host→device
kernel_led_astar(d_graph)                # GPU: ~1μs
path = copy_to_host(d_graph)             # device→host
path_nodes = np.array(path)              # back to numpy

# 5. SWARM — numpy stages candidate features on host
features = np.array(candidates)          # nine_chain_specialized_bridge.py
d_features = copy_to_device(features)    # host→device
kernel_swarm_dispatch(d_features)        # GPU: ~1μs
scores = copy_to_host(d_features)        # device→host
score_array = np.array(scores)           # back to numpy

# 6. HALTING — numpy stages convergence check on host
states = np.array(chain_states)          # sovereign_bridges.py
d_states = copy_to_device(states)        # host→device
kernel_halting_gate(d_states)            # GPU: ~1μs
converged = copy_to_host(d_states)       # device→host
```

Each stage: numpy array creation + host→device + GPU kernel (~1μs) + device→host + numpy array creation. The GPU does ~6μs of work per question. The host does ~6ms of staging. **The GPU is idle 99.9% of the time.**

## THE TARGET: Device-Resident Pipeline

In a game engine, the NPC's world state lives in GPU memory. The NPC's brain reads from and writes to that memory. The CPU only touches it for I/O (display, input, save).

```
Boot:
    Galaxy table → VRAM (GALAXY_UNIVERSE region, 2GB)
    Morton octree → VRAM (HOUSE_CONTEXT region)
    Frustum params → VRAM (WORLD_VIEW region)
    LED-A* graph → VRAM (HOUSE_CONTEXT region)
    Swarm buffers → VRAM (TRM_WEIGHTS region)
    Query scratch → VRAM (WORLD_VIEW region)

Per question:
    ONLY INPUT: query string → trigram codes → ONE copy_to_device
    ALL REMAINING: GPU kernels reading/writing VRAM regions
    ONLY OUTPUT: ONE copy_to_host → answer string
```

ONE host→device (query in). ONE device→host (answer out). Everything between: GPU kernels operating on device-resident data. This is the game loop architecture from THREE_BRAIN_SYSTEM spec.

---

## Phase 3A: Device-Side Transpose (Finish Phase 2 Gap)

**Scope:** `trm_adapters.py` lines 599-602

**Current:** `_apply_gradient_device` transposes gradient ON HOST, uploads BOTH gradient and its transpose (2 `copy_to_device`).

**Target:** Upload gradient ONCE, transpose ON DEVICE.

**How:**

Option 1 — Simple transpose kernel. The AdvancedRPNEngine already has `OP_MATMUL_SMALL`. A transpose is `C = A^T` which is just a memory shuffle. Add `OP_TRANSPOSE_SMALL` to `rpn_opcodes.py` and implement in the RPN engine. For small matrices (128×128), this is trivial.

Option 2 — Use the existing `matmul` with identity. `transpose(G) = I_perm @ G` where `I_perm` is the permutation identity. This works but wastes compute on a matmul for a pure shuffle.

Option 3 — Keep a persistent `gradient_transposed` device buffer (already allocated in `AdapterDeviceBuffers`). After uploading gradient, launch a transpose kernel that reads from `buffers.gradient` and writes to `buffers.gradient_transposed`. Zero host involvement.

**Recommend Option 3.** The buffer already exists. Write a small `transpose_2d` RPN helper in `RPNMathCore` that operates on two `DeviceTensor` pointers. Then `_apply_gradient_device` becomes:

```python
RPNMathCore.copy_to_device(gradient_host, buffers.gradient.ptr)  # 1 upload
self._math_core.transpose_2d(buffers.gradient_transposed, buffers.gradient)  # device-only
```

This cuts steady-state adapter gradient from 2 uploads to 1. Small win, but it completes the Phase 2 contract.

---

## Phase 3B: Spatial Navigation — Device-Resident Buffers

**Scope:** The three spatial sovereign files (146 numpy uses total):
- `morton_octree.py` — 45 uses
- `led_pathfinder.py` — 62 uses
- `frustum.py` — 39 uses

**Why these three FIRST:** They form the PERCEPTION stage of the TRM game loop (THREE_BRAIN_SYSTEM spec §3.2). In a game engine, perception happens entirely in GPU memory — the NPC's view frustum, spatial index, and pathfinding graph are all GPU-resident structures. The NPC does NOT copy the octree to CPU every frame.

**Strategy: Same pattern as Phase 1/2**

1. **Audit each file.** Identify which numpy uses are:
   - **Staging:** Creating arrays to upload to GPU (replace with `HostTensorF32` + one-time upload)
   - **Computation:** Using numpy for math that should be RPN kernels (replace with `RPNMathCore` ops)
   - **Device-resident data:** Arrays that represent VRAM-resident structures (replace with `DeviceTensor` pointers allocated at boot)

2. **Make spatial structures device-resident at boot.** The Morton octree, frustum planes, and LED-A* graph should be allocated in VRAM ONCE during `Knowledgeverse.__init__()` and stay there. Per-question queries should pass a device pointer to the query position, not rebuild arrays.

3. **Replace numpy staging with `HostTensorF32`.** Same pattern as Phase 1. `HostTensorF32` is the sovereign host staging type. No numpy.

4. **Replace numpy computation with RPN opcodes.** Any `np.dot`, `np.linalg.norm`, `np.argsort`, `np.where` in these files should become RPN kernel calls via existing opcodes or new ones.

**Specific tips per file:**

### morton_octree.py

The Morton octree is a SPATIAL INDEX. In a game engine, spatial indices live in GPU memory. The octree structure (node positions, child pointers, Morton codes) should be a contiguous device buffer allocated at boot. Per-question: the query calls `kernel_morton_query` with a device pointer to the query position. The kernel traverses the octree IN VRAM and writes results to a device-resident output buffer.

Look for:
- `np.array` of Morton codes → `HostTensorF32` + one-time `copy_to_device` at boot
- `np.argsort` / `np.where` for neighbor selection → RPN sort/filter opcodes or a dedicated `morton_query.ptx` kernel
- `np.zeros` for result buffers → pre-allocated `DeviceTensor` scratch

### frustum.py

The frustum defines the avatar's FIELD OF VIEW. In a game engine, frustum planes are 6 float4 values that update when the camera moves. They live in GPU memory. Culling is a single kernel launch that reads all object positions and the frustum planes from VRAM.

Look for:
- `np.array` of frustum plane coefficients → `DeviceTensor` updated by the TRM navigation kernel (not by Python)
- `np.dot` for plane-point tests → already a PTX kernel (`kernel_frustum_cull`), but numpy is staging the inputs
- `np.where` for visible set → output mask should be a device-resident bit vector

### led_pathfinder.py

LED-A* is GRAPH NAVIGATION. In a game engine, the nav mesh lives in GPU memory. Pathfinding reads the nav mesh and writes a path — all in VRAM.

Look for:
- `np.array` of edge weights / adjacency → device-resident graph structure at boot
- `np.zeros` for distance/visited arrays → pre-allocated `DeviceTensor` scratch
- `np.argmin` for priority queue → RPN `OP_VEC_ARGMIN` or a dedicated min-heap kernel
- Entire A* open-set/closed-set management in numpy → this is the biggest sovereignty violation. A* should run as a PTX kernel that operates entirely in VRAM.

---

## Phase 3C: Query Pipeline — Device-Resident Embeddings

**Scope:** After spatial navigation is device-resident, the next bottleneck is the query embedding pipeline:
- `rpn_embedding_engine.py` — 46 uses
- `query_head_substrate.py` — 62 uses

**Same strategy:** Query embeddings should be computed on GPU and stay in VRAM. The composed head pipeline should read them from device memory, not from numpy arrays.

**This is Phase 3C — do NOT start until 3A and 3B are validated.**

---

## Execution Order

1. **Phase 3A:** Device-side transpose in `_apply_gradient_device` (trm_adapters.py)
   - Add `transpose_2d` to `RPNMathCore`
   - Modify `_apply_gradient_device` to use it
   - Validate: `pytest tests/test_trm_game_loop.py tests/test_rpn_sovereignty_phase2.py`

2. **Phase 3B:** Morton + Frustum + LED-A* numpy audit and migration
   - Start with `frustum.py` (smallest, 39 uses)
   - Then `morton_octree.py` (45 uses)
   - Then `led_pathfinder.py` (62 uses, most complex)
   - Validate each file independently before moving to the next

3. **Phase 3C:** Query embedding pipeline (AFTER 3B is green)

---

## RULES

1. Do NOT add CPU fallbacks. If a kernel is missing, report what opcode is needed.
2. Do NOT import numpy in any file you touch. Use `HostTensorF32` for host staging, `DeviceTensor` for VRAM.
3. Do NOT create new Python helper functions for math. Use `RPNMathCore` ops or request new RPN opcodes.
4. Do NOT change kernel launch signatures. Adapt the Python staging to match existing kernel contracts.
5. Every file you touch: run `rg "import numpy|from numpy|np\." <file>` and report the count. Target: ZERO.

---

## Spec Grounding

| Design Decision | Spec | Section |
|----------------|------|---------|
| Spatial index lives in VRAM | THREE_BRAIN_SYSTEM | §3.2: "NPC perception = Frustum culling + LOD" |
| Octree/graph device-resident | KNOWLEDGEVERSE | §3: HOUSE_CONTEXT region (2.5GB) |
| Query scratch in VRAM | KNOWLEDGEVERSE | §3: WORLD_VIEW region (2GB) |
| No numpy in hot path | KNOWLEDGEVERSE | §4.1: ptx_fallback_rate = 0.0 |
| Perception entirely on GPU | SGI_SPECIFICATION | §3: "No CPU preprocessing" |
| Game loop = perceive→navigate→reason→decide→act | THREE_BRAIN_SYSTEM | §3.5: TRM game tick |
| One input, one output, all between on GPU | THREE_BRAIN_SYSTEM | Abstract: "Python = boot + I/O only" |

---

## Success Metric

After Phase 3B, run the warm 35% benchmark with live monitor. The target:
- **GPU utilization > 0%** (current: 0%). Even 5% means the spatial pipeline is touching the GPU every question.
- **Process CPU drops** from 146% (current) toward lower — Python is doing less work per question.
- **VRAM usage rises** — more data is device-resident (good).
- **Benchmark score >= 18.66%** — no regression.
- **numpy count in touched files: ZERO.**
