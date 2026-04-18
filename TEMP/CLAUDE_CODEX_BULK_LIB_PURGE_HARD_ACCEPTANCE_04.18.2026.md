# Claude → Codex Spec: Bulk-Lib Purge — Hard Acceptance Gate

**Date**: 2026-04-18
**Author**: Claude (architecture)
**Implementer**: Codex
**Daniel's Ruling (verbatim)**: "Hard acceptance always — we do not use standard libraries, we code our Kernels to have freedom, as this is a new architecture and at the same time Knowledge Representation paradigm, we are just leveraging 3D file standards and aligning as much as possible with the W3C and current standards and protocols."

---

## 1. Principle

K3D is simultaneously a new compute architecture (sovereign GPU-native execution) and a new knowledge-representation paradigm (procedural RPN programs encoding meaning, not surface forms). Standard numerical libraries (NumPy, SciPy, SymPy, sklearn, PyTorch at inference time, SentenceTransformers) are foreign objects in this body. They were never intended for a sovereign system; they carry assumptions (CPU arrays, Python GIL, trained model weights) that are incompatible with the K3D execution model.

The audit (`bulk_lib_audit_04.18.2026.md`) found **3,881 violations** across **127 sovereign files**. These are not technical debt to schedule later. They are architectural violations that prevent the system from running its hot path on GPU.

**Prior treatment of this rule:** Daniel has requested zero numpy in sovereign code fourteen times since October 2025. The count is irrelevant. The rule is now a hard acceptance gate: CI fails if any banned library reference exists in a sovereign path outside of `Old_Attempts/`.

**What we DO align with (external standards):**
- **glTF 2.0 / GLB** (Khronos Group): K3D's House file format. 3D assets, node graph, PBR materials.
- **glTF extensions** (`extras.k3d`): K3D-specific dual-client metadata, embeddings, RDF links.
- **USD / OpenUSD** (Pixar / Khronos): 3D scene description for complex House scenes.
- **W3C RDF / OWL**: Semantic metadata on House nodes; ontology references in star schema.
- **W3C HTML5 / DOM**: `domOps` alongside `meshOps` — K3D projects to DOM elements as first-class output (Christoph Dorn directive, March 2026).
- **W3C ARIA / WCAG 2.2**: Accessibility metadata exposed through public interfaces.
- **WebGPU compute shaders** (W3C): Future browser-side compute alignment; PTX semantics map to WebGPU compute pipelines.
- **WebXR** (W3C): Avatar presence in the House; synthetic user API.
- **Unicode**: Full character space in Character Galaxy.

**What we do NOT align with:**
- Any ML training framework (PyTorch, TensorFlow, JAX) in the inference hot path.
- Any linear-algebra runtime library (NumPy, SciPy, CuPy) in sovereign code.
- Any vector-search framework (FAISS, Annoy) — replaced by `matryoshka_prefix_dot.cu`.
- Any symbolic-math library (SymPy) — replaced by Tier 3 opcodes.
- Any pre-trained embedding service (SentenceTransformers, FastEmbed, HuggingFace Transformers at inference) — replaced by `rpn_meaning_project.ptx`.

---

## 2. Sovereign Replacement Table

For every banned pattern, the sovereign replacement is defined here. If the replacement opcode does not yet exist, this spec authorizes it.

### 2.1 NumPy Array Construction

| Banned Pattern | Sovereign Replacement | Status |
|---|---|---|
| `np.array([...])` | `(ctypes.c_float * N)(*values)` or `StackValue.from_literal(...)` + `YARD_PUSH` | Replacement exists |
| `np.zeros(N, dtype=np.float32)` | `(ctypes.c_float * N)()` (zero-initialized by ctypes) | Replacement exists |
| `np.ones(N, dtype=np.float32)` | ctypes buffer + PTX fill kernel | Replacement exists |
| `np.empty(N)` | `loader.gpu_malloc(N * 4)` directly | Replacement exists |

### 2.2 NumPy Math Operations

| Banned Pattern | Sovereign Replacement | Opcode | Status |
|---|---|---|---|
| `np.dot(a, b)` | Tier 2 `OP_MATVEC_F32` or Tier 1 scalar chain | 0x180+ range | Exists (verify exact opcode in `rpn_opcodes.py`) |
| `np.matmul(A, B)` | Tier 2 `MATMUL_SMALL` | Exists in `rpn_opcodes.py` as `OP_MATMUL_SMALL` | Verify |
| `np.linalg.inv(M)` (4×4) | Tier 3 `MAT4_INV` | Existing WINE range | Exists |
| `np.linalg.inv(M)` (N×N, N≠4) | New: `LU_SOLVE_GENERAL` | **0x1A2** (new) | Not yet — design below |
| `np.linalg.norm(v)` | Tier 1 `OP_SQRT` + `OP_DOT` chain or dedicated `VEC_NORM` | Verify in opcode registry | Exists in part |
| `np.linalg.norm(v, axis=k)` | New: `REDUCE_NORM_AXIS` → use `REDUCE_SUM_AXIS` + `SQRT` | **0x1A5** (new) | Not yet — design below |
| `np.linspace(a, b, n)` | New: `GRID_LINSPACE_F32` | **0x1A4** (new) | Not yet — design below |
| `np.arange(n)` | PTX loop with accumulator; or `GRID_LINSPACE_F32` with step=1 | Same as above | Not yet |
| `np.sum(arr, axis=k)` | New: `REDUCE_SUM_AXIS` | **0x1A5** (new) | Not yet |
| `np.mean(arr, axis=k)` | New: `REDUCE_MEAN_AXIS` | **0x1A6** (new) | Not yet |

### 2.3 NumPy Random

| Banned Pattern | Sovereign Replacement | Opcode | Status |
|---|---|---|---|
| `np.random.randn(N)` | Deterministic Matryoshka seed via `matryoshka_prefix_dot.cu`, OR PTX philox RNG + Box-Muller | `RNG_NORMAL_BOXMULLER` **0x1A3** | Philox exists; Box-Muller wrapper not yet |
| `np.random.rand(N)` | PTX xoroshiro / philox uniform RNG (already exists in `kernels/`) | Existing | Verify path |
| `np.random.randint(a, b)` | PTX philox + modulo | Existing | Verify |

### 2.4 NumPy Indexing and Strides

| Banned Pattern | Sovereign Replacement | Opcode | Status |
|---|---|---|---|
| `arr[mask]` boolean indexing | PTX scatter/gather — **`STRIDED_GATHER`** | **0x1AD** (new) | Not yet |
| `arr[:, indices]` fancy indexing | Same: `STRIDED_GATHER` | **0x1AD** | Not yet |
| `np.ix_(a, b)` outer-product index | Decompose to two `STRIDED_GATHER` calls | Same | Not yet |
| `np.frombuffer(ctypes_ptr, ...)` | Eliminated — ctypes pointer IS the buffer; no numpy wrapping needed | N/A | Remove calls |

### 2.5 CuPy

| Banned Pattern | Sovereign Replacement | Status |
|---|---|---|
| `cp.asarray(host_array)` | Direct `cuMemAlloc` + `cuMemcpyHtoD` via `loader.gpu_malloc()` + `loader.memcpy_htod()` | Replacement exists in `loader.py` |
| `cp.zeros(N)` | `loader.gpu_malloc(N*4)` (CUDA zero-initializes allocation) | Replacement exists |
| `cp.RawModule(path=...)` kernel loading | Formalize `cupy_env.py` as **fallback-only** — non-hot-path kernel loading during dev; replace with `loader.load_ptx_file()` in production | `cupy_env.py` stays as dev tool ONLY |
| `cp.asarray(...)` in hot loop | Remove entirely — `dynamic_lod.py:45` — use pre-allocated GPU buffer | Remove |

**CuPy policy:** `knowledge3d/cranium/utils/cupy_env.py` is NOT a sovereignty violation as a development tool for NVRTC header management. It is NEVER called from a hot-path code path. Any `cp.` call outside of `cupy_env.py` in sovereign paths is a violation.

### 2.6 PyTorch (Inference Path)

| Banned Pattern | Sovereign Replacement | Status |
|---|---|---|
| `torch.as_tensor(arr, device='cuda')` | `loader.gpu_malloc()` + `loader.memcpy_htod()` | Replacement exists |
| `torch.tensor(arr)` | Same | Replacement exists |
| `torch.no_grad()` context | Remove — K3D inference never computes gradients | Remove, no replacement needed |
| `torch.nn.functional.interpolate()` | New: `TENSOR_INTERPOLATE` | **0x1B6** (new) | Not yet |
| `F.grid_sample(...)` | Same: `TENSOR_INTERPOLATE` | **0x1B6** | Not yet |

**PyTorch in training paths (`models/rlwhf_*.py`, `models/spatial_memory_trainer.py`) remains LEGAL** — training is an ingestion-path activity. The prohibition is on inference-time usage.

### 2.7 SentenceTransformers / Embedding Models

| Banned Pattern | Sovereign Replacement | Status |
|---|---|---|
| `SentenceTransformer('all-MiniLM-L6-v2').encode(text)` | `rpn_meaning_projector.project(star.meaning_rpn)` via `rpn_meaning_project.ptx` (Phase B spec) | Phase B not yet written — file shim NOW, complete Phase B next |
| `SentenceTransformer` in `answer_ranker.py:126` | Pre-cache embeddings in Galaxy during load; game loop indexes only via `matryoshka_prefix_dot.cu` | Pending Phase B |

**Qdrant ingestion path:** `qwen_matryoshka_client.py` (Phase A spec) is the only permitted external embedding call, and it runs on the Phenom host (RTX 970), not on the sovereign hot path.

### 2.8 Sklearn

| Banned Pattern | Sovereign Replacement | Status |
|---|---|---|
| `TfidfVectorizer().fit_transform(corpus)` | Pre-compute offline during ingestion phase; runtime is lookup-only from Galaxy | Pre-compute script needed |
| `KMeans(n_clusters=k).fit(X)` | Tier 3 `KMEANS_STEP` opcode (Lloyd iteration exists at estimated 0x186) + new `KMEANS_PLUS_INIT` | **0x1B7** `KMEANS_PLUS_INIT` (new) |
| `sklearn` in `bridge/live_server.py:128` | Move TfIDF to offline indexing; server uses pre-built lookup tables | Split file |

### 2.9 SciPy

| Banned Pattern | Sovereign Replacement | Status |
|---|---|---|
| `scipy.linalg.solve(A, b)` | Tier 3 `LU_SOLVE_GENERAL` (0x1A2 — new below) | Not yet |
| `scipy.sparse.csr_matrix` | Custom PTX sparse CSR format + `SPARSE_MATMUL` opcode | **0x1AE** (new) |

### 2.10 Architectural Violations (Remove, No Replacement Opcode)

The Kimi swarm identified three library usages that are architectural violations — they bypass K3D's spatial indexing and embedding systems entirely:

| Library | Violation | Action |
|---|---|---|
| **NetworkX** (`nx.shortest_path`, `nx.Graph`) | Replaces Morton Octree + LED-A* spatial pathfinding | Remove. Morton Octree IS the graph. LED-A* IS the path search. |
| **SymPy** (`sympy.lambdify`, symbolic diff) | Replaces Tier 3 RPN opcodes for symbolic computation | Remove. Compile offline to Tier 3 RPN programs. No runtime SymPy. |
| **FAISS** (`faiss.IndexFlatIP`, `faiss.Kmeans`) | Replaces `matryoshka_prefix_dot.cu` for vector search | Remove. `matryoshka_prefix_dot.cu` is the vector search. |

---

## 3. New Opcode Inventory — Proposed Range 0x1A0-0x1BF

Opcodes 0x170-0x177 are reserved for Transfer Yard operations (existing spec). Opcodes 0x178-0x17A are the three queue ops (core isolation spec). WINE opcodes at 0x180+. This spec proposes the range **0x1A0-0x1BF** for new math and utility operations required by the bulk-lib purge.

Before assigning, Codex must grep `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` for any existing assignments in this range. If collisions exist, adjust upward.

### Proposed New Opcodes

| Opcode | Mnemonic | Semantics | Priority | Replaces |
|---|---|---|---|---|
| `0x1A0` | `MATMUL_2D_F32` | General (M,K) @ (K,N) matrix multiply on yard banks; supports batch dim | P0 | `np.matmul`, `np.dot` on 2D |
| `0x1A1` | `BATCHED_MATMUL_F32` | (B,M,K) @ (B,K,N) batched variant | P1 | `torch.bmm` patterns |
| `0x1A2` | `LU_SOLVE_GENERAL` | LU decomposition + solve for general NxN; N read from yard bank, result written to bank | P0 | `np.linalg.inv(NxN)`, `scipy.linalg.solve` |
| `0x1A3` | `RNG_NORMAL_BOXMULLER` | Pop count N, pop seed, push N normal-distribution float32 values using philox + Box-Muller | P0 | `np.random.randn` |
| `0x1A4` | `GRID_LINSPACE_F32` | Pop start, stop, n_steps; push n_steps evenly-spaced float32 values onto active bank | P1 | `np.linspace`, `np.arange` |
| `0x1A5` | `REDUCE_SUM_AXIS` | Pop axis_id, pop tensor from bank; push per-slice sums; supports 2D tensors in yard | P0 | `np.sum(arr, axis=k)` |
| `0x1A6` | `REDUCE_MEAN_AXIS` | Pop axis_id, pop tensor; push per-slice means | P0 | `np.mean(arr, axis=k)` |
| `0x1A7` | `ATTENTION_FWD` | Single-head attention: pop Q, K, V from three banks; push output to active bank; scale by 1/√d | P0 | `SentenceTransformer` attention layer |
| `0x1A8` | `LAYER_NORM_FWD` | Pop input from bank, pop gamma/beta scalars; push layer-normalized output | P0 | `SentenceTransformer` layer norm |
| `0x1A9` | `GELU_FWD` | Pop float, push GELU(x) = x·Φ(x) approximation via tanh | P0 | `SentenceTransformer` GELU |
| `0x1AA` | `IMAGE_DECODE_JPEG` | Pop encoded JPEG byte buffer, push decoded float32 RGB tensor (via nvjpeg) | P0 | `cv2.imdecode`, `PIL.Image.open` |
| `0x1AB` | `RESIZE_BILINEAR_F32` | Pop image tensor + (H_out, W_out); push resized float32 tensor | P0 | `cv2.resize`, `PIL.Image.resize` |
| `0x1AC` | `NORMALIZE_IMAGE` | Pop image tensor, pop mean[3], pop std[3]; push per-channel normalized tensor | P0 | cv2 / PIL normalize |
| `0x1AD` | `STRIDED_GATHER` | Pop indices buffer, pop source tensor; push gathered elements (fancy indexing) | P0 | `arr[mask]`, `arr[:, indices]` |
| `0x1AE` | `SPARSE_MATMUL` | Pop CSR format (data, indices, indptr) from two banks; push dense result | P1 | `scipy.sparse.csr_matrix.dot` |
| `0x1AF` | `SPARSE_EIGSH` | Pop CSR matrix, pop n_eigenpairs; push top-n eigenvalues + vectors | P2 | `scipy.sparse.linalg.eigsh` |
| `0x1B6` | `TENSOR_INTERPOLATE` | Pop input tensor, pop (H_out, W_out), pop mode (0=bilinear, 1=nearest); push interpolated tensor | P0 | `F.interpolate`, `F.grid_sample` |
| `0x1B7` | `KMEANS_PLUS_INIT` | Pop data matrix, pop k; push k initialized centroids using k-means++ seeding | P1 | `sklearn.KMeans` init |
| `0x1B8` | `CTYPES_VIEW_AS_PTX` | Registers a ctypes pointer as a PTX-accessible buffer descriptor; no copy | P0 | `np.ctypeslib.as_array`, `np.frombuffer` patterns |
| `0x1B9` | `CUDA_MALLOC_ASYNC` | Stream-ordered async allocation; pop size_bytes; push device pointer | P1 | `torch.tensor.to('cuda')` stream patterns |

Opcodes `0x1B0-0x1B5`, `0x1BA-0x1BF` are reserved for future use.

**Registration requirement**: Every new opcode above MUST be added to `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` before Codex implements it. No opcode is implemented without registry entry first.

---

## 4. W3C and 3D-Standard Alignment Checkpoint

Per Daniel's ruling, K3D's only external alignment is W3C standards and 3D file formats. This table makes the boundary explicit.

### Aligned (these are correct external dependencies)

| Standard | Body | K3D Use | Where |
|---|---|---|---|
| glTF 2.0 | Khronos | House file format; star persistence | `GLB` files in House (Region 3) |
| glTF extensions (`extras.k3d`) | K3D / Khronos track | Dual-client metadata, embeddings, RDF links | Node spec |
| USD / OpenUSD | Pixar / ASWF | Complex House scene description (if needed) | House construction |
| RDF / OWL | W3C | Semantic metadata on House nodes | Star schema |
| HTML5 / DOM | W3C | `domOps` output target alongside meshOps | Tablet WINE output |
| WCAG 2.2 | W3C WAI | Accessibility metadata | House spec |
| ARIA | W3C | Semantic markup for dual-client output | Tablet |
| WebGPU compute shaders | W3C | Future browser-side PTX mapping | PTX ↔ WebGPU alignment |
| WebXR | W3C | Avatar presence in House; synthetic user API | Viewer |
| Unicode | Unicode Consortium | Full character space in Character Galaxy | Galaxy |

### Not Aligned (these are prohibited in sovereign paths)

| Library / Framework | Why Not |
|---|---|
| NumPy | CPU array abstraction; not GPU-native; alien to PTX execution model |
| SciPy | Depends on BLAS/LAPACK; Python callback-based; not composable with RPN programs |
| SymPy | Runtime symbolic computation; Python-based; replaced by Tier 3 PTX opcodes |
| PyTorch (inference) | Autograd + graph construction overhead; inference uses PTX kernels directly |
| SentenceTransformers | Trained model weights; external data dependency; replaced by `rpn_meaning_project.ptx` |
| sklearn | Python-level ML algorithms; replaced by sovereign PTX equivalents |
| FastEmbed | Same class as SentenceTransformers |
| HuggingFace Transformers | Same class |
| FAISS | Vector search framework; replaced by `matryoshka_prefix_dot.cu` |
| NetworkX | Graph algorithms; replaced by Morton Octree + LED-A* |
| pandas | Data manipulation library; CPU-only; no sovereign path use |

---

## 5. Migration Phasing — 8 Phases

These 8 phases are ordered by dependency. Phase N cannot begin until Phase N-1 acceptance gates pass. The core isolation spec (`CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md`) runs in parallel with Phase 0.

### Phase 0: Archive and Isolate (Pre-requisite — run first)

**Entry gate**: None — this is the starting condition.
**Actions**:
- Complete Old_Attempts migration per `CLAUDE_CODEX_OLD_ATTEMPTS_MIGRATION_04.18.2026.md`.
- Place shims at all archived paths.
- Complete core isolation per `CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md`.
- All three queue opcodes (0x178-0x17A) registered and implemented.
- `Old_Attempts/` excluded from all grep gates.

**Exit gate**: All 5 gates in the Old_Attempts spec pass. All 5 gates in the core isolation spec pass.

---

### Phase 1: Transfer Yard Default on All 3 Tiers

**Entry gate**: Phase 0 exit gates pass.
**Actions** (per Transfer Yard spec `CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md`):
- Tier 1: delete variant flag in `lightweight_rpn.py` — yard is the only path.
- Tier 2: `modular_rpn_kernel_transfer_yard.cu` + `.ptx` compiled; `TransferYardStack` Python sidecar deleted.
- Tier 3: `advanced_rpn_kernel_transfer_yard.cu` + `.ptx` compiled; `advanced_rpn.py` points to it.
- `MAX_INSTANCES` hard-codes replaced with `query_sm_count() * 9`.
- `RPN_STACK_DEPTH` set to 69 in `rpn_execute_device.cuh`.

**Exit gate**: Transfer Yard spec §9 acceptance gates pass (all 7).

---

### Phase 2: ActionBuffer and Scalar Dtype Cleanup

**Entry gate**: Phase 1 exit gates pass.
**Target**: `bridge/headless_tablet.py`, `cranium/actions/action_types.py`
**Actions**:
- Convert `ActionBuffer` `np.dtype(...)` to `ctypes.Structure` with native C99 types.
- Replace all `np.uint32`, `np.float32` scalar casts with native ctypes scalars.
- Verify zero-copy semantics in bridge marshalling.
- Remove any `np.zeros`, `np.array` in `bridge/headless_tablet.py:564-595`.

**Exit gate**:
```bash
grep -rn "import numpy\|from numpy" knowledge3d/bridge/ --exclude-dir=Old_Attempts  # → 0
grep -rn "import numpy\|from numpy" knowledge3d/cranium/actions/ --exclude-dir=Old_Attempts  # → 0
```

---

### Phase 3: CuPy Formalization and Confidence Path

**Entry gate**: Phase 2 exit gates pass.
**Target**: `cranium/utils/cupy_env.py`, `cranium/actions/confidence_propagation.py`, `cranium/dynamic_lod.py`
**Actions**:
- Formalize `cupy_env.py` with module-level comment: "DEV TOOL ONLY — not permitted in hot path."
- Replace `torch.as_tensor(...device='cuda')` in `confidence_propagation.py:56-90` with `loader.gpu_malloc()` + `loader.memcpy_htod()`.
- Replace `cp.asarray(...)` in `dynamic_lod.py:45` with pre-allocated GPU buffer.
- Audit `cranium/actions/confidence_propagation.py:78` — remove `cp.zeros` and `cp.asarray`.

**Exit gate**:
```bash
grep -rn "cp\." knowledge3d/cranium/ --include="*.py" --exclude-dir=Old_Attempts | grep -v cupy_env.py  # → 0
grep -rn "torch\.as_tensor\|torch\.tensor" knowledge3d/cranium/actions/ --exclude-dir=Old_Attempts  # → 0
```

---

### Phase 4: Knowledgeverse Galaxy State GPU Migration

**Entry gate**: Phase 3 exit gates pass.
**Target**: `knowledgeverse/knowledgeverse.py:50-80` (136 numpy violations)
**Actions**:
- Migrate `np.zeros(n_nodes, dtype=...)` galaxy state arrays to GPU-resident ctypes buffers + `cuMemAlloc`.
- Migrate all `np.` operations in `knowledgeverse/semantic_csr_graph.py` to PTX-based sparse ops or pre-allocated VRAM.
- Remove `torch` inference-time imports from `knowledgeverse.py`.
- Target: `knowledgeverse/` numpy violation count 197 → 0.

**Exit gate**:
```bash
grep -rn "import numpy\|from numpy\|np\." knowledge3d/knowledgeverse/ --include="*.py" --exclude-dir=Old_Attempts  # → 0
grep -rn "import torch\|from torch" knowledge3d/knowledgeverse/ --include="*.py" --exclude-dir=Old_Attempts | grep -v "rlwhf\|training"  # → 0
```

---

### Phase 5: Phase B Native Embedding — Retire Surface-Form Path

**Entry gate**: Phase 1 exit gates pass (Transfer Yard needed for `rpn_meaning_project.cu`).
**Actions** (per Phase B spec `CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING_04.18.2026.md`):
- Write `rpn_meaning_project.cu` + compile to PTX.
- Write `rpn_meaning_project_bridge.py` (ctypes, no numpy).
- Write `rpn_meaning_projector.py` (sovereign wrapper).
- Update `star_crafter.py` to use `rpn_meaning_projector.project(star.meaning_rpn)`.
- Retire `sovereign_multi_modal_embedder.py` surface-form path (shims already placed in Phase 0).
- Retire `rpn_embedding_engine.py` and `sovereign_matryoshka_embedder.py` surface-form callers in hot path.

**Exit gate**: Phase B spec §9 acceptance gates pass (all 8).

---

### Phase 6: Cranium Specialists — numpy Purge

**Entry gate**: Phases 1 and 5 exit gates pass.
**Target**: `cranium/procedural_compiler.py` (198), `cranium/specialists/procedural_drawing_specialist.py` (127), `cranium/specialists/batch_optimizer.py` (14)
**Actions**:
- Replace `np.random.randn(256, dim)` in `procedural_compiler.py:42` and `procedural_drawing_specialist.py:167` with PTX philox + Box-Muller (`RNG_NORMAL_BOXMULLER`, 0x1A3) or deterministic Matryoshka seed.
- Replace `np.zeros`, `np.array(codes)` in `procedural_drawing_specialist.py:214-220` with ctypes byte-packed struct.
- Replace `ord(c)` text-to-embedding path with Character Galaxy star_id lookup (per Phase B spec §5.1).
- Replace all numpy in `batch_optimizer.py` with `TransferYardTier3Engine` operations.
- Implement `GRID_LINSPACE_F32` (0x1A4), `REDUCE_SUM_AXIS` (0x1A5), `REDUCE_MEAN_AXIS` (0x1A6) opcodes as needed.

**Exit gate**:
```bash
grep -rn "import numpy\|from numpy\|np\." knowledge3d/cranium/specialists/ --include="*.py" --exclude-dir=Old_Attempts  # → 0
grep -rn "np\.random" knowledge3d/cranium/ --include="*.py" --exclude-dir=Old_Attempts  # → 0
```

---

### Phase 7: Bridges and Grey-Area Splits

**Entry gate**: Phase 6 exit gates pass.
**Target**: `bridge/live_server.py`, `cranium/memory.py`, `scripts/benchmark_audio_minimal.py`
**Actions**:
- `bridge/live_server.py:128` — move TfIDF computation to offline ingestion script; server endpoint uses pre-built lookup table from Galaxy.
- `cranium/memory.py:245` — move sklearn KMeans call to offline preprocessing; runtime does lookup only. Implement `KMEANS_PLUS_INIT` (0x1B7) for runtime clustering if needed.
- `scripts/benchmark_audio_minimal.py` — split: extract test harness (Python/numpy OK for test infrastructure) from sovereign kernel invocation (clean path).
- `glb_weights.py:42` — audit `torch.tensor(arr)` — if on inference path, replace with `loader.memcpy_htod()`.
- `cranium/ptx_runtime/thinking_tag_embedder.py:28` — remove `torch.no_grad()` context manager.

**Exit gate**:
```bash
grep -rn "sklearn\|TfidfVectorizer\|KMeans" knowledge3d/cranium/ knowledge3d/bridge/ --include="*.py" --exclude-dir=Old_Attempts  # → 0
grep -rn "torch\.no_grad" knowledge3d/cranium/ --include="*.py" --exclude-dir=Old_Attempts  # → 0
```

---

### Phase 8: Full Sweep and Hard Gate Activation

**Entry gate**: Phases 2-7 all pass their exit gates.
**Actions**:
- Run the final hard gate grep battery below.
- Fix any remaining violations found.
- Activate CI hard gate (fail on any hit).
- Write completion report in `docs/reports/bulk_lib_purge_completion.md`.

---

## 6. Acceptance Gates (HARD — CI Fails on Any Hit)

These grep commands return exit code 0 when clean, nonzero when violations exist. CI pipeline runs them as a required check.

```bash
# Gate 1: No numpy in sovereign paths
grep -rn "import numpy\|from numpy" knowledge3d/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts \
    --exclude-dir=tests \
    --exclude-dir=scripts
# Expected: 0 lines

# Gate 2: No cupy in hot path (cupy_env.py is the only permitted location)
grep -rn "import cupy\|from cupy\|cp\." knowledge3d/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts \
    --exclude="cupy_env.py" \
    --exclude-dir=tests
# Expected: 0 lines

# Gate 3: No scipy in sovereign paths
grep -rn "import scipy\|from scipy" knowledge3d/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts \
    --exclude-dir=tests
# Expected: 0 lines

# Gate 4: No sklearn in sovereign paths
grep -rn "import sklearn\|from sklearn" knowledge3d/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts \
    --exclude-dir=tests \
    --exclude-dir=scripts
# Expected: 0 lines

# Gate 5: No SentenceTransformers in sovereign paths
grep -rn "sentence_transformers\|SentenceTransformer\|sentence-transformers" knowledge3d/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts \
    --exclude-dir=tests
# Expected: 0 lines

# Gate 6: No torch in inference paths
# NOTE: torch is permitted in knowledge3d/models/ (training). Prohibited in cranium/, knowledgeverse/, bridge/, tablet/.
grep -rn "import torch\|from torch" \
    knowledge3d/cranium/ \
    knowledge3d/knowledgeverse/ \
    knowledge3d/bridge/ \
    knowledge3d/tablet/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts \
    --exclude-dir=tests
# Expected: 0 lines

# Gate 7: No pandas in sovereign paths
grep -rn "import pandas\|from pandas" knowledge3d/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts \
    --exclude-dir=tests \
    --exclude-dir=scripts
# Expected: 0 lines

# Gate 8: Matryoshka Weight-Matrix Pack-Order Verification
# Every ternary-packed weight matrix (1.6-bit / 5-trits-per-byte) MUST store rows in ascending row-index order
# Tier-prefix truncation at Matryoshka boundaries is only valid if row order is stable
# CI check: static assert at kernel-registration time (first-row == 0, last-row == N-1, strictly increasing)
# + one-time runtime check on weight upload
grep -rn "ASCENDING_ROW_ORDER\|pack_order_header" knowledge3d/cranium/ptx/ \
    --include="*.cu" \
    --include="*.cuh"
# Expected: ≥1 hits (kernel registration verifies header byte 0x01)

# Gate 9: No FastEmbed or HuggingFace transformers at inference
grep -rn "fastembed\|FastEmbed\|transformers\.AutoModel\|from transformers import" \
    knowledge3d/cranium/ knowledge3d/knowledgeverse/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts
# Expected: 0 lines

# Gate 10: No NetworkX, SymPy, or FAISS in sovereign paths
grep -rn "import networkx\|import sympy\|import faiss\|from networkx\|from sympy\|from faiss" \
    knowledge3d/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts \
    --exclude-dir=tests
# Expected: 0 lines

# Gate 11: No np.frombuffer or np.ctypeslib hidden bridges
grep -rn "np\.frombuffer\|np\.ctypeslib\|numpy\.frombuffer" knowledge3d/ \
    --include="*.py" \
    --exclude-dir=Old_Attempts
# Expected: 0 lines
```

---

## 7. Codex Handoff Checklist (Ordered by Phase)

**Phase 0 (Pre-requisites)**
1. Complete Old_Attempts migration spec — run all 5 gates, report pass/fail.
2. Complete core isolation spec — run all 5 gates, report pass/fail.
3. Confirm `Old_Attempts/` excluded from all grep gates (add `--exclude-dir=Old_Attempts` everywhere).

**Phase 1 (Transfer Yard)**
4. Write `modular_rpn_kernel_transfer_yard.cu` — compile to PTX, flip Tier 1/2/3 bridges.
5. Write `advanced_rpn_kernel_transfer_yard.cu` — compile to PTX, wire Tier 3 bridge.
6. Run Transfer Yard spec §9 gates — all 7 must pass.

**Phase 2 (ActionBuffer)**
7. Rewrite `action_types.py` `ActionBuffer` as `ctypes.Structure`. Run Phase 2 exit gate.

**Phase 3 (CuPy)**
8. Replace `cp.asarray` / `torch.as_tensor` in `confidence_propagation.py`, `dynamic_lod.py`. Run Phase 3 exit gate.

**Phase 4 (Knowledgeverse)**
9. Migrate `knowledgeverse.py:50-80` galaxy state to GPU-resident ctypes. This is the largest single task (1-2 days). Run Phase 4 exit gate.

**Phase 5 (Native Embedding)**
10. Write `rpn_meaning_project.cu` per Phase B spec §3. Compile. Write bridge + projector. Run Phase B §9 gates.

**Phase 6 (Specialists)**
11. Replace `np.random.randn` in `procedural_drawing_specialist.py:167` with `RNG_NORMAL_BOXMULLER` (0x1A3) or Matryoshka seed.
12. Replace `np.zeros/np.array(codes)` in `procedural_drawing_specialist.py:214-220` with ctypes struct.
13. Purge numpy from `batch_optimizer.py`. Run Phase 6 exit gate.

**Phase 7 (Bridges)**
14. Split `live_server.py` — move TfIDF offline. Run Phase 7 exit gate.
15. Remove `torch.no_grad()` from `thinking_tag_embedder.py:28`.

**Phase 8 (Hard Gate)**
16. Run all 10 hard gates (§6). Fix any remaining hits.
17. Add all 10 gates to CI pipeline.
18. Extend `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` with all new opcodes in §3 (0x1A0-0x1B9 as applicable).
19. Write `docs/reports/bulk_lib_purge_completion.md` — reference audit date, final grep counts, gate results.
20. Confirm `Old_Attempts/` contains the expected archived files and their README_WHY_ARCHIVED.md files.

---

## 8. Must-NOT-Do List

- ❌ Do NOT start Phase 6 (specialists) before Phase 1 (Transfer Yard) is complete. The yard substrate IS the replacement for numpy arrays in specialists.
- ❌ Do NOT start Phase 5 (native embedding) before Phase 1 (Transfer Yard) is complete. `rpn_meaning_project.cu` uses the yard for program execution.
- ❌ Do NOT run Phase 8 hard gate until Phases 2-7 all pass. Running the gate on a partially migrated tree just creates noise.
- ❌ Do NOT add `try/except ImportError: import numpy as np` or any similar "try sovereign, fall back to numpy" pattern. We fail and fix. There are no fallbacks.
- ❌ Do NOT move files from Old_Attempts back into sovereign paths. Archive is one-way until a new spec explicitly reintegrates them.
- ❌ Do NOT add new numpy imports to fix a temporary gap while the sovereign replacement is being written. Raise `NotImplementedError` or leave a TODO comment and file it as a gap. The gap is better than a regression.
- ❌ Do NOT keep silent: every acceptance gate pass or fail must be reported with evidence (the grep output or test result), not just "passed."
