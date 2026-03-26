# Codex: Phase D.3b — Migrate Missing Navigation Stages to GPU Kernels

**Date:** 2026-03-26
**Priority:** IMMEDIATE — D.3 regressed 18.02% → 14.39% because the device pipeline SKIPPED three reasoning stages. Do NOT restore Python. Migrate the logic to GPU kernels and Galaxy knowledge.
**Binding specs:**
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §4 — fail-fast, sovereignty first, VRAM-native
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` §2.3 — intelligence from COMPOSING procedural programs
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` §2.4 — sovereign execution: PTX-only hot path, no NumPy
- `docs/vocabulary/HYPER_PARALLEL_PROCESSING.md` — specialist cores communicate via shared registers, one-mind convergence
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` — TRM IS the avatar, Galaxy = internal brain (VRAM workspace)

**DO NOT STOP between parts. Execute ALL. The instructions are complete.**

---

## THE PROBLEM

`_compose_head_navigation_candidates_device()` (line 10232) chains Morton → Frustum → LOD on device but SKIPS three reasoning stages that the original path performs. These stages are currently Python (NumPy, heapq, string matching) — they must become GPU kernels, NOT be restored as Python.

### What Was Lost

| Stage | Original Path | Device Path (D.3) | Impact |
|-------|--------------|-------------------|--------|
| Seed Selection | `select_seed_nodes()` — NumPy dot product, galaxy mask, top-k | SKIPPED — no semantic seed fallback | Lost rescue when Morton misses |
| Graph Expansion | `extract_local_kernel()` — Python heapq Dijkstra on CSR | SKIPPED — no local subgraph built | LED-A* has nothing to navigate |
| LED-A* Focus | `navigate_csr()` → sets `led_focus: 1.0` on best node | ZEROED — `led_focus: 0.0` for all | Ranking collapsed (led_focus is sorted FIRST) |
| Subject Anchoring | `_subject_anchor_match_score()` — Python string matching | SKIPPED for device candidates | MMLU lost 212 answers |

---

## THE FIX: Four New GPU Kernels + Device-Resident CSR

### Part A: `seed_select_top_k.cu` — GPU Seed Selection Kernel

**Replaces:** `SemanticCSRGraph.select_seed_nodes()` (semantic_csr_graph.py line 78)

That method does: `similarities = embeddings @ query` (NumPy matrix-vector multiply), galaxy mask, top-k argpartition. This is a textbook GPU operation.

**What the kernel does:**
1. Takes the embedding matrix (N × 16, already normalized, uploaded to VRAM at boot), query vector (16 floats), galaxy index array (N × int32), allowed galaxy mask
2. Each thread computes dot product between one row and the query vector (16 FMA ops)
3. Threads with disallowed galaxy indexes write -inf
4. Warp-level top-k reduction to find the K highest-similarity nodes
5. Writes top-K (index, similarity) pairs to output buffer on device

**Kernel signature:**
```c
extern "C" __global__ void seed_select_top_k(
    const float* __restrict__ embeddings,    // N × dim, row-major, L2-normalized
    const int32_t* __restrict__ galaxy_ids,  // N galaxy indexes
    const float* __restrict__ query,         // dim floats
    const int32_t* __restrict__ allowed_galaxies, // G allowed galaxy indexes (or NULL for all)
    int32_t allowed_count,                   // number of allowed galaxies (0 = allow all)
    int32_t num_entries,                     // N
    int32_t dim,                             // embedding dimension (16)
    int32_t top_k,                           // how many seeds to return
    float similarity_threshold,              // minimum similarity
    int32_t* __restrict__ out_indices,       // top_k output indices
    float* __restrict__ out_similarities,    // top_k output similarities
    int32_t* __restrict__ out_count          // actual count written
);
```

**Upload at boot:** The embedding matrix and galaxy index array from `SemanticCSRGraph` must be uploaded to VRAM once during `_initialize_query_substrate()` or equivalent. These are Region 2 (Galaxy Universe) assets — they belong in VRAM per Knowledgeverse §3.

**Approach:** Launch N/256 blocks of 256 threads. Each thread computes one dot product. Use shared memory for partial top-k merge per block, then a final reduction kernel for global top-k. For N ≈ 250K entries and dim=16, this is ~4M FMA ops — microseconds on an RTX 3070.

---

### Part B: `graph_expand_bfs.cu` — GPU Graph Expansion Kernel

**Replaces:** `SemanticCSRGraph.extract_local_kernel()` (semantic_csr_graph.py line 110)

That method does: Dijkstra BFS on CSR graph using Python heapq, expanding seed nodes up to max_nodes=2048. The CSR arrays (row_offsets, col_indices, packed_costs) are NumPy arrays with ~250K nodes.

**What the kernel does:**
1. Takes device-resident CSR graph (row_offsets, col_indices, packed_costs — uploaded to VRAM at boot) and seed node indices from Part A's output
2. Runs parallel BFS/Dijkstra expansion from seeds using shared-memory priority queue
3. Edge cost: `(0.35 × geo) + (0.65 × sem)` where `geo = packed & 0xFFFF`, `sem = packed >> 16`
4. Expands up to max_nodes (2048) and max_edge_expansions (24576)
5. Writes selected node list + local CSR (remapped indices) to device output buffers

**Kernel signature:**
```c
extern "C" __global__ void graph_expand_bfs(
    const uint32_t* __restrict__ row_offsets,    // global CSR: N+1 row offsets
    const uint32_t* __restrict__ col_indices,    // global CSR: nnz column indices
    const uint32_t* __restrict__ packed_costs,   // global CSR: nnz packed (sem|geo) costs
    const int32_t* __restrict__ seed_indices,    // K seed node indices (from Part A)
    int32_t seed_count,                          // K
    int32_t max_nodes,                           // expansion limit (2048)
    int32_t max_edge_expansions,                 // edge budget (24576)
    float alpha,                                 // geometric weight (0.35)
    float beta,                                  // semantic weight (0.65)
    int32_t* __restrict__ selected_nodes,        // output: up to max_nodes global indices
    int32_t* __restrict__ selected_count,        // output: actual count
    uint32_t* __restrict__ local_row_offsets,    // output: local CSR rows
    uint32_t* __restrict__ local_col_indices,    // output: local CSR cols (remapped)
    uint32_t* __restrict__ local_packed_costs,   // output: local CSR costs
    int32_t* __restrict__ local_edge_count       // output: edges in local CSR
);
```

**Upload at boot:** The CSR graph arrays must be uploaded to VRAM once. These are part of the LED navigation substrate — Region 1 (KERNELS) or Region 2 (GALAXY_UNIVERSE) per Knowledgeverse §3.

**Approach:** One block, 256 threads. Use shared memory for the priority queue (max_nodes × 8 bytes = 16KB, fits in 48KB shared). Threads cooperatively expand frontier nodes. The warp-level min reduction finds the next cheapest frontier node each iteration (same pattern as `trm_recursive_fused.cu` drift reduction).

---

### Part C: Wire LED-A* with Device-Resident Input

**The LED-A* kernel (`led_astar.cu`) already exists and runs on GPU.** But `navigate_csr()` (led_pathfinder.py line 334) does `memcpy_htod` for the CSR arrays on EVERY call because the local CSR was built in Python.

After Part B, the local CSR lives on device. Add a `navigate_csr_device()` method that:
1. Takes device pointers to the local CSR from Part B's output (no memcpy)
2. Builds the query→seed and seed→goal virtual edges on device (the CSR augmentation from knowledgeverse.py lines 10012-10061 — reimplement as a small kernel or as device-side buffer patching)
3. Launches `led_astar_navigate` with device-resident pointers
4. Returns device pointer to path result (no memcpy back yet)

**New method on LEDPathfinder:**
```python
def navigate_csr_device(
    self,
    d_row_offsets: int,      # device pointer
    d_col_indices: int,      # device pointer
    d_packed_costs: int,     # device pointer
    num_vertices: int,
    num_edges: int,
    *,
    start: int,
    goal: int,
    alpha: float = 0.35,
    beta: float = 0.65,
    max_path_length: int = 128,
) -> tuple[int, int]:  # (d_path_ptr, path_length)
```

This eliminates the per-query `memcpy_htod` for CSR data (6 allocations + 6 transfers per call, visible in navigate_csr lines 378-400).

---

### Part D: Subject Anchor as Galaxy Spatial Proximity

**Replaces:** `_subject_anchor_match_score()` (knowledgeverse.py line 8161) — Python string matching on metadata.

Per the specs, subject affinity is SPATIAL PROXIMITY in Galaxy. The TRM's specialist adapter biases navigation toward the right neighborhood — this is Hyper-Parallel Processing's "specialist core = spatial bias in Galaxy navigation."

**The migration:** Instead of Python string matching on `metadata.mmlu_subjects`, encode subject affinity as a numeric weight in the seed selection kernel:

1. **At ingestion time:** Each Galaxy entry already has `gpu_galaxy_index` (integer) and `embedding16` (16 floats). Add a `subject_cluster_id` (uint16) derived from the entry's subject/domain metadata. Entries with the same MMLU subject get the same cluster ID.

2. **At query time:** The MMLU domain hint maps to a `target_cluster_id`. The `seed_select_top_k` kernel (Part A) adds a cluster bias term to the similarity score:
   ```c
   float sim = dot(embeddings[idx], query);
   if (target_cluster_id > 0 && subject_clusters[idx] == target_cluster_id) {
       sim += cluster_bias;  // e.g., 0.15 — replaces the Python 0.8/0.45/0.0 scoring
   }
   ```

3. **The cluster ID mapping** is built once at ingestion and uploaded to VRAM with the embedding matrix. It's KNOWLEDGE (stored in Galaxy), not logic (removed from Python).

**Subject cluster assignment kernel** (optional, can be done at ingestion in Python since it's NOT hot path):
- Group entries by their `subject` or `domain` metadata field
- Assign sequential uint16 IDs per group
- Store as `subject_clusters` array parallel to `embeddings` and `galaxy_indexes`
- Upload to VRAM at boot

**At query time, map MMLU `domain_hint` to cluster ID:**
- Build a `subject_name → cluster_id` lookup dict at boot (this IS acceptable as boot-time Python)
- Pass `target_cluster_id` to the seed selection kernel as a single int32

---

### Part E: Compose the Full Device Pipeline

Now rewrite `_compose_head_navigation_candidates_device()` using the four new kernels:

```
1. seed_select_top_k     → device buffer: (seed_indices, seed_similarities, seed_count)
2. morton_locate_device   → device buffer: (morton_indices, morton_count)
   MERGE seed + morton on device (union kernel or simple concat+dedup)
3. graph_expand_bfs       → device buffer: (local CSR on device)
4. navigate_csr_device    → device buffer: (led_path, led_focus_index)
5. frustum_visible_device → device buffer: (visible_indices, visible_count)
6. lod_metrics_device     → device buffer: (lod_indices, lod_count)
7. ONE readback: read final candidates + led_focus_index + lod_metrics
```

**Steps 1-6 are entirely on GPU. Python only touches step 7's output to format the answer.**

The candidate dict construction (lines 10361-10409 in the current device path) becomes:
- `led_focus`: 1.0 for the node matched by LED-A* focus (from step 4)
- `led_path`: from step 4's path output
- `subject_anchor_focus`: already folded into step 1's similarity scores via cluster bias
- `similarity`: from step 1 or recomputed in step 7 readback
- `lod_saliency` / `lod_level`: from step 6

---

### Part F: Boot-Time VRAM Upload

Add to the Knowledgeverse initialization (after House loads, during `bind_gpu_galaxy_runtime` or similar):

```python
# Upload CSR graph to VRAM (Region 2: Galaxy Universe)
graph = self.get_semantic_csr_graph()
if graph is not None:
    self._d_csr_embeddings = gpu_upload(graph.embeddings)       # N × 16 float32
    self._d_csr_galaxy_ids = gpu_upload(graph.galaxy_indexes)   # N × int32
    self._d_csr_row_offsets = gpu_upload(graph.row_offsets)     # (N+1) × uint32
    self._d_csr_col_indices = gpu_upload(graph.col_indices)     # nnz × uint32
    self._d_csr_packed_costs = gpu_upload(graph.packed_costs)   # nnz × uint32
    self._d_subject_clusters = gpu_upload(subject_clusters)     # N × uint16
```

For 250K entries, 16-dim embeddings:
- Embeddings: 250K × 16 × 4 = 16 MB
- Galaxy IDs: 250K × 4 = 1 MB
- CSR (k=8 neighbors): row 1 MB + col 8 MB + costs 8 MB = 17 MB
- Subject clusters: 250K × 2 = 0.5 MB
- **Total: ~35 MB** — well within the 12 GB VRAM budget (currently using 1.5 GB)

---

## Part G: Validate

```bash
# Compile new kernels
nvcc -ptx -o knowledge3d/cranium/ptx/seed_select_top_k.ptx knowledge3d/cranium/ptx/seed_select_top_k.cu
nvcc -ptx -o knowledge3d/cranium/ptx/graph_expand_bfs.ptx knowledge3d/cranium/ptx/graph_expand_bfs.cu

# Compile check
python3 -m compileall knowledge3d/knowledgeverse/knowledgeverse.py
python3 -m compileall knowledge3d/knowledgeverse/query_head_substrate.py
python3 -m compileall knowledge3d/knowledgeverse/semantic_csr_graph.py
python3 -m compileall knowledge3d/cranium/spatial_sovereign/led_pathfinder.py

# Focused tests
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py

# Parity smoke: device path should match or exceed D.2 answer
K3D_DEVICE_PIPELINE=1 python3 -c "
from knowledge3d.knowledgeverse import Knowledgeverse
kv = Knowledgeverse()
result = kv.query('What is 2+3?', specialist='math')
print(f'Answer: {result.get(\"answer\", result.get(\"predicted_answer\", \"none\"))}')
print(f'LED focus used: {\"led_focus\" in str(result.get(\"selection_steps\", []))}')
print('Device pipeline with GPU navigation: PASSED')
"
```

---

## Part H: Benchmark

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
export K3D_DEVICE_PIPELINE=1
conda activate k3d-cranium

nohup python3 -u benchmarks/run_all.py \
  --warm --sample-rate 0.35 \
  > /tmp/k3d_phaseD3b_gpu_nav_warm_35pct_03.26.log 2>&1 &

echo "Phase D.3b benchmark launched. PID: $!"
```

**While it runs:** 2-minute live monitor (use PYTHON PID, not wrapper PID).

**EXPECTED:**
- Combined score: ≥ 18% (restore D.2 parity or exceed it)
- GPU utilization: HIGHER than D.2's 3.88% — seed selection and graph expansion now run on GPU too
- MMLU: ≥ 1062 (D.2 level) — LED focus and subject anchoring restored via kernels

---

## Part I: Report

Write to `TEMP/CLAUDE_PHASE_D3b_GPU_NAVIGATION_REPORT_03.26.2026.md` with:

1. All 5 suite scores + combined
2. Comparison table:
   | Metric | D.2 | D.3 (broken) | D.3b (kernels) |
   |--------|-----|-------------|----------------|
   | ARC | 2/42 | 2/42 | ? |
   | Math | 3/500 | 1/500 | ? |
   | GSM8K | 4/462 | 3/462 | ? |
   | LHE | 2/35 | 1/35 | ? |
   | MMLU | 1062/4915 | 850/4915 | ? |
   | Combined | 18.02% | 14.39% | ? |
   | GPU avg | 3.88% | 0.17% | ? |
   | GPU max | 25% | 1% | ? |
3. New kernel compilation confirmation (seed_select_top_k.ptx, graph_expand_bfs.ptx)
4. VRAM usage after CSR upload
5. Whether LED focus node appeared in selection_steps
6. Contrastive/sleep-time outcome
7. numpy audit: `rg "import numpy|from numpy" knowledge3d/knowledgeverse/query_head_substrate.py knowledge3d/cranium/spatial_sovereign/led_pathfinder.py`

---

## THE VISION

**Before D.3b (Python navigation):**
```
Morton(GPU) → Python(sort/filter) → Python(heapq Dijkstra) → LED-A*(GPU) → Python(frustum prep) → Frustum(GPU) → Python(LOD prep) → LOD(GPU) → Python(format)
   50μs          15ms                     8ms                   200μs            5ms                 20μs             3ms               20μs         5ms
```

**After D.3b (GPU navigation):**
```
seed_select(GPU) → Morton(GPU) → graph_expand(GPU) → LED-A*(GPU) → Frustum(GPU) → LOD(GPU) → ONE readback → Python(format)
     100μs            50μs           200μs              200μs          20μs          20μs        100μs           5ms
```

Python touches ONLY the final formatting. All navigation, seed selection, graph expansion, pathfinding, culling, and LOD happen in VRAM with device-pointer passing.

---

## EXECUTION ORDER — DO NOT STOP

A -> B -> C -> D -> E -> F -> G -> H -> I

All in sequence. No pauses. The instructions are HERE.
