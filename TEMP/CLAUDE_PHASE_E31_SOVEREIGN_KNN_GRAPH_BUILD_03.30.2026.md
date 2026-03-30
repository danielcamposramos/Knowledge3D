# Claude -- Phase E.31: Sovereign KNN Graph Build (Replace Numpy Matmul)

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL -- this is why benchmarks stall AND it's a sovereignty violation

---

## Daniel's Correction

> "Why are we calculating with numpy when we do have our own matryoshka rpn embedding standard?"

He's right. The KNN graph build in `semantic_csr_graph.py` line 698-709 uses numpy's
matmul to compute cosine similarity between 248K embedding vectors:

```python
sims = (embeddings[start:end] @ embeddings.T).astype(np.float32)  # numpy!
```

This is wrong on TWO levels:
1. **Sovereignty violation:** numpy in the Galaxy navigation substrate
2. **Architecture violation:** embeddings are already in VRAM but we pull them to CPU

---

## What We Already Have (Sovereign)

| Sovereign Component | File | Function |
|---------------------|------|----------|
| `RPNEmbeddingEngine` | `cranium/rpn_embedding_engine.py` | Trigram→embedding (128-dim, no numpy, `Float32Vector`) |
| `TrigramEmbedBridge` | `cranium/bridges/trigram_embed_bridge.py` | GPU trigram lookup + L2 normalize |
| `trigram_embed.cu` | `cranium/ptx/trigram_embed.cu` | PTX: `trigram_lookup_average` + `l2_normalize_embedding` |
| `MatryoshkaProjectionBridge` | `cranium/bridges/matryoshka_bridge.py` | GPU Matryoshka adaptive-dim projection |
| `matryoshka_project.cu` | `cranium/ptx/matryoshka_project.cu` | PTX: `y = W[:dim,:dim] * x` with NaN/Inf guards |
| `seed_select_top_k.cu` | `cranium/ptx/seed_select_top_k.cu` | **PTX: cosine similarity + top-K selection ON GPU** |
| `SemanticCSRGraph._d_embeddings` | `semantic_csr_graph.py` line 162 | Device pointer -- **embeddings already in VRAM** |

**The sovereign answer already exists.** `seed_select_top_k.cu` computes cosine
similarity between a query embedding and ALL entries, with top-K selection, entirely
on GPU using our own RPN embeddings. The KNN graph build is just running this logic
for every row -- a batched version of what we already have.

---

## Spec Grounding

### From KNOWLEDGEVERSE_SPECIFICATION.md §2.1:
> "Galaxy Universe is always loaded in VRAM"

The KNN graph IS the LED-A* navigation substrate. It's Galaxy infrastructure.
Building it with numpy violates this.

### From TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md §1:
> "Matryoshka: Hierarchical fractal specialists"

The Matryoshka embedding principle: use different dimension prefixes for different
precision levels. 16-dim (current `embedding16`) is the coarsest LOD. Higher dims
give finer matching. This maps directly to the Dynamic LOD step in the composed head
pipeline: coarse dims for distant navigation, fine dims for close scoring.

### From RPN_DOMAIN_OPCODE_REGISTRY.md:
> "Programs before opcodes"

The KNN similarity computation IS an RPN-expressible program (dot product + normalize).
Our trigram embeddings ARE the sovereign representation. Using numpy bypasses both.

### From SOVEREIGN_NSI_SPECIFICATION.md §3:
> "Hot path = PTX + Galaxy + RPN + TRM ONLY"

The semantic CSR graph file header says "it can use NumPy because it is not part of
the sovereign PTX hot path." But the KNN graph feeds LED-A* which feeds Frustum Cull
which feeds the Composed Head Pipeline. It IS hot-path infrastructure.

---

## The Fix: Batched GPU KNN Build Kernel

### What the numpy code does (line 698-709):

```
For each batch of 512 rows:
  1. Compute cosine similarity: batch(512) × all(248K) via matmul
  2. Argpartition: find top-12 per row
  3. Sort: order by similarity
  4. Pack: into CSR neighbor lists
```

### What the sovereign version should do:

**New PTX kernel: `knn_graph_build.cu`**

```c
extern "C" __global__ void knn_graph_build(
    const float* __restrict__ embeddings,    // [N, dim] already in VRAM
    int32_t N,                               // 248K entries
    int32_t dim,                             // 16 (Matryoshka coarse LOD)
    int32_t k,                               // 12 neighbors per node
    float threshold,                         // 0.3 minimum similarity
    int32_t* __restrict__ out_neighbors,     // [N, k] output neighbor indices
    float* __restrict__ out_similarities,    // [N, k] output similarity scores
    int32_t* __restrict__ out_counts         // [N] actual neighbor count per row
) {
    // Each thread block handles one source row
    int source = blockIdx.x;
    if (source >= N) return;

    const float* src_row = embeddings + source * dim;

    // Shared memory for top-K tracking (insertion sort, k is small)
    __shared__ float best_scores[64];   // MAX_K
    __shared__ int32_t best_indices[64];

    // Initialize
    if (threadIdx.x < k) {
        best_scores[threadIdx.x] = -1e38f;
        best_indices[threadIdx.x] = -1;
    }
    __syncthreads();

    // Each thread in block processes a stripe of target rows
    for (int target = threadIdx.x; target < N; target += blockDim.x) {
        if (target == source) continue;

        // Dot product (embeddings are pre-normalized)
        const float* tgt_row = embeddings + target * dim;
        float sim = 0.0f;
        for (int d = 0; d < dim; ++d) {
            sim += src_row[d] * tgt_row[d];
        }

        if (sim < threshold) continue;

        // Atomic insertion into top-K (compare against worst in top-K)
        // ... standard parallel top-K pattern ...
    }
    __syncthreads();

    // Write results
    if (threadIdx.x == 0) {
        int count = 0;
        for (int i = 0; i < k; ++i) {
            if (best_indices[i] >= 0) {
                out_neighbors[source * k + count] = best_indices[i];
                out_similarities[source * k + count] = best_scores[i];
                count++;
            }
        }
        out_counts[source] = count;
    }
}
```

**Launch:** `grid = (248K, 1, 1), block = (256, 1, 1)`

Each of 248K thread blocks computes similarity between one source row and all 248K
targets, using 256 threads per block to parallelize the target scan.

### Performance Estimate

| Metric | Numpy (CPU) | Sovereign (GPU) |
|--------|-------------|-----------------|
| Hardware | 1 CPU core ~10 GFLOPS | RTX 3070 ~20 TFLOPS |
| Time | ~200 seconds | **< 1 second** |
| Embeddings location | Pulled to CPU | **Stay in VRAM** |
| Dependencies | numpy.matmul | **Zero** (PTX only) |
| Sovereignty | ❌ Violated | ✅ Compliant |

248K × 248K × 16 × 2 = ~2 TFLOPS total. RTX 3070 at ~20 TFLOPS/s = 0.1 seconds
for the matmul. Top-K selection adds minimal overhead (k=12 is tiny).

---

## Execution Sequence

### Step 1: Write `knn_graph_build.cu` (New Sovereign Kernel)

Location: `knowledge3d/cranium/ptx/knn_graph_build.cu`

The kernel follows the same sovereign pattern as `seed_select_top_k.cu` (which already
does single-query cosine similarity + top-K on GPU). The difference: batched across
all rows simultaneously.

Compile to PTX: `knowledge3d/cranium/ptx/knn_graph_build.ptx`

### Step 2: Write `KNNGraphBuildBridge` (Sovereign Bridge)

Location: Add to `knowledge3d/knowledgeverse/semantic_csr_graph.py`

Pattern: same as `TrigramEmbedBridge` -- load PTX, allocate device buffers, launch
kernel, read back results. The bridge:

1. Takes `_d_embeddings` pointer (already in VRAM from `bind_gpu_galaxy_runtime`)
2. Allocates output buffers: `[N, k]` neighbors + similarities + counts
3. Launches `knn_graph_build` kernel
4. Reads back neighbor lists → builds CSR arrays
5. Caches result to `.npz` (same signature-based cache as before)

### Step 3: Replace numpy KNN in `load_or_build_semantic_csr_graph()`

Replace lines 698-722 (the numpy matmul + argpartition loop) with:

```python
# Sovereign GPU path: KNN build using our own embeddings already in VRAM
d_embeddings = loader.gpu_malloc(int(embeddings.nbytes))
loader.memcpy_htod(d_embeddings, embeddings.ctypes.data_as(ctypes.c_void_p), int(embeddings.nbytes))
neighbors = _gpu_knn_build(d_embeddings, node_count, embed_dim, k_eff, similarity_threshold, catalog)
loader.gpu_free(d_embeddings)
```

Or better: accept the device pointer directly if embeddings are already in VRAM
(which they are after `bind_gpu_galaxy_runtime`).

### Step 4: Add Timing Diagnostics

```python
t0 = time.perf_counter()
# ... GPU KNN build ...
print(f"[K3D] KNN graph build: {time.perf_counter()-t0:.2f}s (GPU, {node_count} nodes)")
```

### Step 5: Run Benchmarks

With the GPU KNN build, the graph populates in <1s (was 200s). Benchmarks should
boot and reach GPU queries immediately.

---

## CSR Construction (After GPU KNN)

The GPU kernel produces `[N, k]` neighbor indices + similarities. The CSR packing
(row_offsets, col_indices, packed_costs) is a sequential scan -- fine on CPU since
it's O(N×k) = O(3M) operations, takes milliseconds. Or it could also be GPU-native
(prefix sum for row_offsets), but that's optimization, not sovereignty.

---

## Matryoshka LOD Integration (Future Enhancement)

The current embeddings are 16-dim (coarsest Matryoshka LOD). The `RPNEmbeddingEngine`
supports 128-dim. Future enhancement:

1. **Coarse pass** (dim=16): Build initial KNN graph quickly (~0.1s)
2. **Fine pass** (dim=64 or 128): Re-score top candidates with higher-dim embeddings
3. **Maps to Dynamic LOD**: farther = fewer dims, closer = more dims

This is exactly the Matryoshka principle applied to graph construction. But for now,
16-dim sovereign GPU build is the fix.

---

## Files to Create

| File | Purpose |
|------|---------|
| `knowledge3d/cranium/ptx/knn_graph_build.cu` | Batched GPU KNN kernel (cosine similarity + top-K) |

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/semantic_csr_graph.py` | Replace numpy matmul (lines 698-722) with sovereign GPU bridge |
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Add timing diagnostics to boot/bind path |

---

## Why This Matters Beyond Performance

This is NOT just a 200× speedup. It's an architectural correction:

1. **The embeddings are ours** -- `RPNEmbeddingEngine` produces them sovereign
2. **The embeddings are already in VRAM** -- `bind_gpu_galaxy_runtime` put them there
3. **The similarity kernel exists** -- `seed_select_top_k.cu` does exactly this for single queries
4. **Numpy has no business here** -- the file header's excuse ("build-time code can use NumPy") is wrong because the KNN graph IS Galaxy navigation infrastructure

The comment in `semantic_csr_graph.py` line 1-5:

```python
"""This module is intentionally build-time / query-support code. It can use NumPy
because it is not part of the sovereign PTX hot path itself"""
```

**This comment should be removed.** The module builds the LED-A* navigation graph.
That's Galaxy infrastructure. It should be sovereign.

---

## Success Criteria

- [ ] `knn_graph_build.cu` compiles to PTX
- [ ] KNN build runs on GPU using sovereign embeddings (zero numpy in similarity computation)
- [ ] Build time < 2 seconds for 248K entries (was 200s)
- [ ] Embeddings stay in VRAM (no CPU round-trip)
- [ ] CSR graph identical in structure to numpy version (same neighbor quality)
- [ ] Cache still works (.npz save/load for subsequent boots)
- [ ] Benchmarks boot in seconds and reach GPU queries
- [ ] The numpy sovereignty-violation comment is removed from file header
