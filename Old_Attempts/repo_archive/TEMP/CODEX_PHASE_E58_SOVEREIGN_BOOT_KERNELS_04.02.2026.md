# CODEX E.58 — Sovereign Boot Kernels: Star Materializer + Ref CSR Builder

**Date:** April 2, 2026
**Prerequisite:** E.57 implemented, sovereign_runtime_bundle.pkl created, manifest fresh
**Goal:** Move the artifact build path off Python onto the GPU — making it sovereign AND fast
**Sovereignty:** Hot path already sovereign. This closes the remaining host-side build debt.

---

## The Debt Codex Correctly Identified

After E.57, the runtime hot path is fully sovereign (gpu_task_dispatch + trm_recursive_core + lesson/gravity kernels). The debt is in artifact generation — specifically in `sovereign_hot_path.ensure_loaded()` when the bundle is missing or stale:

| Step | Current owner | Should be |
|------|--------------|-----------|
| Catalog → packed field structs | Python per-entry loop | `star_materializer` GPU kernel |
| Embedding16 → Embedding32 (doubling + normalize) | Python `_pad32()` per entry | `star_materializer` GPU kernel |
| Role/CSR offset+count arrays | Python loop over all stars | `ref_csr_builder` GPU kernel |
| Star table upload | Python `struct.pack` → HtoD | Streaming via pinned buffers |
| Artifact save | Python rebuild from host arrays | GPU D2H after device compaction |

Python's job becomes: file I/O + struct packing of scalar fields + kernel launch. Everything else runs on GPU.

**Why this matters beyond speed:** It makes boot itself sovereign. The knowledge corpus materializes INTO the GPU, not through Python lists. TRM's brain assembles itself on-device, not in the host process.

---

## Architecture: Double-Buffered Streaming Boot

```
Disk                    CPU (Python)            GPU

[chunk 1 pkl] ──read──► [pack input struct]
                         [pin to buffer A] ──HtoD stream A──► [star_materializer]
[chunk 2 pkl] ──read──► [pack input struct]                        ↓
                         [pin to buffer B] ──HtoD stream B──► [star_materializer]
                                                (overlap)          ↓
                                                               [ref_csr_builder]
                                                                    ↓
                                                               [artifact export D2H]
                                                                    ↓
                                                  Python: save sovereign_runtime_bundle.pkl
```

Three stages run concurrently:
- Stage 1 (CPU thread): reads catalog chunk from disk, packs into `CatalogInputEntry` struct array
- Stage 2 (CUDA stream A): HtoD transfer of packed chunk to device input buffer
- Stage 3 (CUDA stream B): `star_materializer` processes previous chunk on GPU

Double-buffer: while stream B materializes chunk N, stream A transfers chunk N+1, CPU packs chunk N+2.

---

## Kernel 1: `star_materializer`

**New file:** `knowledge3d/cranium/cuda/star_materializer.cu`
**Header:** include `device_functions.cuh` (for fnv1a64 and cosine32)
**Launch:** `star_materializer<<<(chunk_size + 127) / 128, 128, 0, stream>>>`

### Input Struct (packed by Python, HtoD transferred)

Python packs each catalog entry into this layout using `struct.pack`:

```c
// Offset  Size  Field
//   0     64    float embedding16[16]  (the precomputed 16-dim embedding)
//  64      4    uint32_t galaxy_id
//  68      4    uint32_t star_type
//  72      4    uint32_t role_id       (0=unknown,1=router,2=executor,3=validator,4=answer,5=anti_pattern)
//  76      4    uint32_t layer_id      (1-4)
//  80      4    uint32_t flags         (STAR_FLAG_ACTIVE=0x01, STAR_FLAG_LEARNABLE=0x02)
//  84      4    uint32_t answer_eligible (0 or 1)
//  88      4    int32_t  semantic_polarity (-1, 0, or 1)
//  92      4    float    semantic_focus
//  96      4    float    semantic_mass
// 100      4    float    attractive_prior
// 104      4    float    repulsive_prior
// 108      4    uint32_t route_policy_id
// 112      8    uint64_t star_hash     (fnv1a64 of star_id string, computed by Python)
// 120      4    float    position[3]   (domain_hash/norm, subject_hash/norm, layer_id/4.0)
// 132     20    uint8_t  padding
// Total: 152 bytes per input entry
#define CATALOG_INPUT_ENTRY_BYTES 152
```

### Kernel Logic (one thread per star)

```c
extern "C" __global__ void star_materializer(
    unsigned char* __restrict__ galaxy_table,   // STAR_RECORD_BYTES * max_stars
    const unsigned char* __restrict__ input,    // CATALOG_INPUT_ENTRY_BYTES * entry_count
    unsigned int entry_count,
    unsigned int star_offset                    // base index into galaxy_table
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= entry_count) return;

    const unsigned char* src = input + i * CATALOG_INPUT_ENTRY_BYTES;
    unsigned char* dst = galaxy_table + (star_offset + i) * STAR_RECORD_BYTES;

    // Expand embedding16 → embedding32: double + L2 normalize
    float e[32];
    float norm_sq = 0.0f;
    for (int d = 0; d < 16; ++d) {
        const float v = *reinterpret_cast<const float*>(src + d * 4);
        e[d]      = v;
        e[d + 16] = v;
        norm_sq  += v * v;
    }
    norm_sq *= 2.0f;  // doubling the vector doubles the dot product of each half
    const float inv_norm = (norm_sq > 1e-12f) ? rsqrtf(norm_sq) : 0.0f;
    for (int d = 0; d < 32; ++d) {
        *reinterpret_cast<float*>(dst + STAR_EMBEDDING_OFFSET + d * 4) = e[d] * inv_norm;
    }

    // Copy scalar fields directly from input struct to star record
    *reinterpret_cast<unsigned int*>(dst + STAR_GALAXY_ID_OFFSET)        = *reinterpret_cast<const unsigned int*>(src + 64);
    *reinterpret_cast<unsigned int*>(dst + STAR_TYPE_OFFSET)              = *reinterpret_cast<const unsigned int*>(src + 68);
    *reinterpret_cast<unsigned int*>(dst + STAR_SELECTION_ROLE_OFFSET)    = *reinterpret_cast<const unsigned int*>(src + 72);
    *reinterpret_cast<unsigned int*>(dst + STAR_LAYER_ID_OFFSET)          = *reinterpret_cast<const unsigned int*>(src + 76);
    *reinterpret_cast<unsigned int*>(dst + STAR_FLAGS_OFFSET)             = *reinterpret_cast<const unsigned int*>(src + 80);
    *reinterpret_cast<unsigned int*>(dst + STAR_ANSWER_ELIGIBLE_OFFSET)   = *reinterpret_cast<const unsigned int*>(src + 84);
    *reinterpret_cast<int*>(dst + STAR_SEMANTIC_POLARITY_OFFSET)          = *reinterpret_cast<const int*>(src + 88);
    *reinterpret_cast<float*>(dst + STAR_SEMANTIC_FOCUS_OFFSET)           = *reinterpret_cast<const float*>(src + 92);
    *reinterpret_cast<float*>(dst + STAR_SEMANTIC_MASS_OFFSET)            = *reinterpret_cast<const float*>(src + 96);
    *reinterpret_cast<float*>(dst + STAR_ATTRACTIVE_PRIOR_OFFSET)         = *reinterpret_cast<const float*>(src + 100);
    *reinterpret_cast<float*>(dst + STAR_REPULSIVE_PRIOR_OFFSET)          = *reinterpret_cast<const float*>(src + 104);
    *reinterpret_cast<unsigned int*>(dst + STAR_ROUTE_POLICY_OFFSET)      = *reinterpret_cast<const unsigned int*>(src + 108);
    *reinterpret_cast<unsigned long long*>(dst + STAR_STAR_HASH_OFFSET)   = *reinterpret_cast<const unsigned long long*>(src + 112);

    // Position (3 floats)
    for (int d = 0; d < 3; ++d) {
        *reinterpret_cast<float*>(dst + STAR_POSITION_OFFSET + d * 4) = *reinterpret_cast<const float*>(src + 120 + d * 4);
    }

    // Velocity: zero-initialize
    for (int d = 0; d < 3; ++d) {
        *reinterpret_cast<float*>(dst + STAR_VELOCITY_OFFSET + d * 4) = 0.0f;
    }

    // Ref counts: zero-initialize (ref_csr_builder fills these after)
    *reinterpret_cast<unsigned int*>(dst + STAR_ROUTER_REF_COUNT_OFFSET)       = 0u;
    *reinterpret_cast<unsigned int*>(dst + STAR_EXECUTOR_REF_COUNT_OFFSET)     = 0u;
    *reinterpret_cast<unsigned int*>(dst + STAR_VALIDATOR_REF_COUNT_OFFSET)    = 0u;
    *reinterpret_cast<unsigned int*>(dst + STAR_ANTI_PATTERN_REF_COUNT_OFFSET) = 0u;
    // Null-fill ref slots
    for (int slot = 0; slot < 2; ++slot) {
        *reinterpret_cast<unsigned int*>(dst + STAR_ROUTER_REFS_OFFSET       + slot * 4) = 0xFFFFFFFFu;
        *reinterpret_cast<unsigned int*>(dst + STAR_EXECUTOR_REFS_OFFSET     + slot * 4) = 0xFFFFFFFFu;
        *reinterpret_cast<unsigned int*>(dst + STAR_VALIDATOR_REFS_OFFSET    + slot * 4) = 0xFFFFFFFFu;
        *reinterpret_cast<unsigned int*>(dst + STAR_ANTI_PATTERN_REFS_OFFSET + slot * 4) = 0xFFFFFFFFu;
    }
}
```

**Constants needed in the header or top of file (must match galaxy_vram_table.py exactly):**
```c
#define STAR_RECORD_BYTES                  256
#define STAR_EMBEDDING_OFFSET                0
#define STAR_GALAXY_ID_OFFSET              128
#define STAR_TYPE_OFFSET                   132
#define STAR_SELECTION_ROLE_OFFSET         136
#define STAR_LAYER_ID_OFFSET               140
#define STAR_FLAGS_OFFSET                  144
#define STAR_ANSWER_ELIGIBLE_OFFSET        148
#define STAR_SEMANTIC_POLARITY_OFFSET      152
#define STAR_SEMANTIC_FOCUS_OFFSET         156
#define STAR_SEMANTIC_MASS_OFFSET          160
#define STAR_ATTRACTIVE_PRIOR_OFFSET       164
#define STAR_REPULSIVE_PRIOR_OFFSET        168
#define STAR_ROUTE_POLICY_OFFSET           172
#define STAR_STAR_HASH_OFFSET              176
#define STAR_ROUTER_REF_COUNT_OFFSET       184
#define STAR_ROUTER_REFS_OFFSET            188
#define STAR_EXECUTOR_REF_COUNT_OFFSET     196
#define STAR_EXECUTOR_REFS_OFFSET          200
#define STAR_VALIDATOR_REF_COUNT_OFFSET    208
#define STAR_VALIDATOR_REFS_OFFSET         212
#define STAR_ANTI_PATTERN_REF_COUNT_OFFSET 220
#define STAR_ANTI_PATTERN_REFS_OFFSET      224
#define STAR_POSITION_OFFSET               232
#define STAR_VELOCITY_OFFSET               244
```

---

## Kernel 2: `ref_csr_builder`

**New file:** `knowledge3d/cranium/cuda/ref_csr_builder.cu`
**Purpose:** After all stars are materialized, write role-typed refs directly into each star's record.

Python builds the ref tuples: `(star_index, role_type, ref_index, slot)` as a packed array. The kernel then writes them to the correct offsets in the star table.

This is an alternative to the host-side CSR offset/count array approach used by `GalaxyVRAMTable.load_stars()`. Instead, refs are written INLINE into the 256-byte star record (where the 2-slot ref arrays live at fixed offsets).

### Input Ref Tuple Struct (packed by Python)

```c
// Each ref tuple: 16 bytes
//  0   4  uint32_t star_index   (which star this ref belongs to)
//  4   4  uint32_t role_type    (0=router,1=executor,2=validator,3=anti_pattern)
//  8   4  uint32_t ref_index    (index of the referenced star)
// 12   4  uint32_t slot         (0 or 1, up to ROLE_REF_LIMIT=2)
#define REF_TUPLE_BYTES 16
```

### Kernel Logic

```c
extern "C" __global__ void ref_csr_builder(
    unsigned char* __restrict__ galaxy_table,
    const unsigned char* __restrict__ ref_tuples,
    unsigned int ref_count
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ref_count) return;

    const unsigned char* src = ref_tuples + i * REF_TUPLE_BYTES;
    const unsigned int star_index = *reinterpret_cast<const unsigned int*>(src + 0);
    const unsigned int role_type  = *reinterpret_cast<const unsigned int*>(src + 4);
    const unsigned int ref_index  = *reinterpret_cast<const unsigned int*>(src + 8);
    const unsigned int slot       = *reinterpret_cast<const unsigned int*>(src + 12);

    if (slot >= 2u) return;  // ROLE_REF_LIMIT = 2

    unsigned char* star_dst = galaxy_table + star_index * STAR_RECORD_BYTES;

    unsigned int count_offset, refs_offset;
    switch (role_type) {
        case 0: count_offset = STAR_ROUTER_REF_COUNT_OFFSET;       refs_offset = STAR_ROUTER_REFS_OFFSET;       break;
        case 1: count_offset = STAR_EXECUTOR_REF_COUNT_OFFSET;     refs_offset = STAR_EXECUTOR_REFS_OFFSET;     break;
        case 2: count_offset = STAR_VALIDATOR_REF_COUNT_OFFSET;    refs_offset = STAR_VALIDATOR_REFS_OFFSET;    break;
        case 3: count_offset = STAR_ANTI_PATTERN_REF_COUNT_OFFSET; refs_offset = STAR_ANTI_PATTERN_REFS_OFFSET; break;
        default: return;
    }

    // Write the ref index into the correct slot
    *reinterpret_cast<unsigned int*>(star_dst + refs_offset + slot * 4) = ref_index;

    // Atomically update the count (count = max occupied slot + 1)
    // Since slots are written once and slot < 2, use atomicMax on the count field
    atomicMax(reinterpret_cast<unsigned int*>(star_dst + count_offset), slot + 1u);
}
```

**Note on atomics:** `atomicMax` on the count field is safe here because each star's ref slots are written by distinct threads with distinct slot values. The final count = max(slot+1 for all written slots) = number of valid refs.

---

## Python Side: What Replaces `_build_stars_from_catalog()`

The new sovereign boot builder in `sovereign_hot_path.py` (`_build_stars_sovereign()`) does:

```python
CHUNK_SIZE = 4096
CATALOG_INPUT_ENTRY_BYTES = 152

def _build_stars_sovereign(self, catalog: list[dict]) -> None:
    """Stream catalog into GPU via star_materializer kernel."""
    import struct
    import ctypes

    n_stars = len(catalog)
    # Pre-resolve all string refs → integer indices (fast: pure Python dict)
    id_to_index = {str(row.get("id") or ""): i for i, row in enumerate(catalog)}

    # Collect ref tuples: (star_index, role_type[0-3], ref_index, slot)
    ref_tuples: list[tuple[int,int,int,int]] = []
    for i, row in enumerate(catalog):
        source = self.knowledgeverse._resolve_catalog_entry(row)
        for role_type, key_list in enumerate(
            [["router_refs","component_refs"], ["executor_refs","grammar_refs"],
             ["validator_refs","meta_refs"], ["anti_pattern_refs","contrastive_refs"]]
        ):
            slot = 0
            for key in key_list:
                for ref_id in list(source.get(key) or []):
                    if slot >= 2: break
                    ref_idx = id_to_index.get(str(ref_id))
                    if ref_idx is not None:
                        ref_tuples.append((i, role_type, ref_idx, slot))
                        slot += 1

    # Materialize stars in chunks via star_materializer kernel
    for chunk_start in range(0, n_stars, CHUNK_SIZE):
        chunk = catalog[chunk_start : chunk_start + CHUNK_SIZE]
        chunk_size = len(chunk)
        input_buf = bytearray(chunk_size * CATALOG_INPUT_ENTRY_BYTES)
        for j, row in enumerate(chunk):
            source = self.knowledgeverse._resolve_catalog_entry(row)
            embedding16 = (self.knowledgeverse._precomputed_entry_embedding16(source)
                           or [0.0] * 16)[:16]
            embedding16 += [0.0] * (16 - len(embedding16))
            offset = j * CATALOG_INPUT_ENTRY_BYTES
            struct.pack_into("<16f", input_buf, offset, *embedding16)
            struct.pack_into("<IIIIIIi", input_buf, offset + 64,
                galaxy_id, star_type, role_id, layer_id, flags, answer_eligible,
                semantic_polarity)
            struct.pack_into("<fffIfQ3f",  input_buf, offset + 92,
                semantic_focus, semantic_mass, attractive_prior, route_policy_id,
                star_hash, pos_x, pos_y, pos_z)
        # HtoD + launch
        dev_input = loader.alloc_gpu(len(input_buf))
        loader.memcpy_htod(dev_input, ..., len(input_buf))
        loader.launch(self._star_materializer_kernel, ...)
        loader.gpu_free(dev_input)

    # Upload ref tuples + launch ref_csr_builder
    if ref_tuples:
        ref_buf = bytearray(len(ref_tuples) * 16)
        for k, (si, rt, ri, sl) in enumerate(ref_tuples):
            struct.pack_into("<4I", ref_buf, k * 16, si, rt, ri, sl)
        dev_refs = loader.alloc_gpu(len(ref_buf))
        loader.memcpy_htod(dev_refs, ..., len(ref_buf))
        loader.launch(self._ref_csr_builder_kernel, ...)
        loader.gpu_free(dev_refs)
```

**What Python is NOT doing in this path:**
- No `star.get("embedding")` list manipulation for 259k stars
- No `struct.pack(...)` of the full 256-byte star record in Python
- No building of CSR offset/count arrays in Python
- No `GalaxyVRAMTable.load_stars()` (that method does all the above)

`GalaxyVRAMTable.load_stars()` remains as the legacy path for tests and small tables. The new `_build_stars_sovereign()` replaces it for the full corpus rebuild.

---

## Async Python I/O: Overlapping Disk and GPU

For the chunked loop above, overlap disk reads and GPU work using Python threading:

```python
from concurrent.futures import ThreadPoolExecutor
import queue

def _build_stars_sovereign_async(self, catalog: list[dict]) -> None:
    """Double-buffered streaming: disk → CPU pack → HtoD → kernel."""
    pack_queue = queue.Queue(maxsize=3)   # max 3 chunks in flight

    def reader_worker():
        for start in range(0, len(catalog), CHUNK_SIZE):
            chunk = catalog[start : start + CHUNK_SIZE]
            packed = self._pack_chunk(chunk, start)   # CPU packing
            pack_queue.put(packed)
        pack_queue.put(None)  # sentinel

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(reader_worker)
        while True:
            packed = pack_queue.get()
            if packed is None:
                break
            self._upload_and_launch_chunk(packed)   # HtoD + kernel
            # CUDA stream launches are async; Python continues immediately
```

This means: while the GPU materializes chunk N, Python is packing chunk N+1 AND reading chunk N+2 from disk. Three-stage pipeline with < 5ms Python overhead between chunks.

---

## Python Launcher Bridge

**New file:** `knowledge3d/knowledgeverse/star_materializer_bridge.py`

Follows the same pattern as `gpu_task_dispatch.py`:

```python
from pathlib import Path
from knowledge3d.cranium.sovereign import loader

CUDA_DIR = Path(__file__).resolve().parents[1] / "cranium" / "cuda"
PTX_DIR  = Path(__file__).resolve().parents[1] / "cranium" / "ptx"

class StarMaterializerBridge:
    """Sovereign kernel bridge for GPU-side star table construction."""
    def __init__(self) -> None:
        self.materializer_kernel = loader.load_ptx_file(
            str(self._ensure_ptx("star_materializer")), "star_materializer")
        self.csr_kernel = loader.load_ptx_file(
            str(self._ensure_ptx("ref_csr_builder")), "ref_csr_builder")

    def materialize_chunk(self, galaxy_table_ptr, input_ptr, entry_count, star_offset):
        loader.launch(self.materializer_kernel,
            ((entry_count + 127) // 128, 1, 1), (128, 1, 1),
            galaxy_table_ptr, input_ptr,
            ctypes.c_uint(entry_count), ctypes.c_uint(star_offset))

    def build_csr(self, galaxy_table_ptr, ref_tuples_ptr, ref_count):
        loader.launch(self.csr_kernel,
            ((ref_count + 127) // 128, 1, 1), (128, 1, 1),
            galaxy_table_ptr, ref_tuples_ptr, ctypes.c_uint(ref_count))
```

---

## Integration in `sovereign_hot_path.py`

In `ensure_loaded()`, when the artifact is stale/missing:

```python
# OLD: Python-heavy path
stars = self._build_stars_from_catalog(catalog)
self.star_table.load_stars(stars)

# NEW: sovereign GPU path
self._build_stars_sovereign_async(catalog)
# star_table already has all stars materialized on GPU
# (star_count set after materialize_chunk calls complete)
```

`_build_stars_from_catalog()` and `GalaxyVRAMTable.load_stars()` are kept for tests, small tables, and fallback — but are NOT called in the production boot path.

---

## Expected Performance Improvement

| Step | Before | After |
|------|--------|-------|
| Python field extraction × 259k | ~60s | ~30s (CPU-bound, limited by dict access) |
| Embedding expansion × 259k | ~5s Python | ~50ms GPU kernel |
| Star record packing × 259k | ~20s Python `struct.pack` | ~30ms GPU kernel |
| CSR build × 259k | ~15s Python | ~20ms GPU kernel |
| HtoD transfer | sequential | overlapped with CPU pack |
| Total rebuild | 5-15 min | Target: < 2 min |

The remaining ~30s is Python catalog reading + field extraction per entry (unavoidable until entries are stored in pre-packed binary form). That is E.59 work: a **canonical binary star format** for the House, so entries are stored as pre-packed `CATALOG_INPUT_ENTRY_BYTES` blobs and loaded directly without per-entry Python extraction.

---

## Success Criteria

1. `star_materializer.cu` compiles cleanly: `nvcc -ptx -arch=sm_86 star_materializer.cu`
2. `ref_csr_builder.cu` compiles cleanly
3. After rebuild with new path: `sovereign_runtime_manifest.json` shows updated `default_knowledge_signature`
4. Rebuilt bundle loads correctly: `ensure_loaded()` returns `mode=artifact`, `star_count >= 259_943`
5. Route depth test: MATH family query achieves `route_depth >= 2` (regression from E.56 must not break)
6. All existing tests pass: `pytest tests/test_spine_routing.py tests/test_gpu_task_dispatch.py tests/test_galaxy_vram_table.py` — 35 passed

---

## Files

| File | Action |
|------|--------|
| `knowledge3d/cranium/cuda/star_materializer.cu` | NEW — GPU star packing kernel |
| `knowledge3d/cranium/cuda/ref_csr_builder.cu` | NEW — GPU ref slot writer |
| `knowledge3d/cranium/ptx/star_materializer.ptx` | Auto-generated by nvcc |
| `knowledge3d/cranium/ptx/ref_csr_builder.ptx` | Auto-generated by nvcc |
| `knowledge3d/knowledgeverse/star_materializer_bridge.py` | NEW — Python launcher bridge |
| `knowledge3d/knowledgeverse/sovereign_hot_path.py` | Add `_build_stars_sovereign_async()`, update `ensure_loaded()` |

No changes to hot-path kernels (gpu_task_dispatch, trm_recursive_core, trm_step_fused). No sovereignty regressions.
