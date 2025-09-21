# PTX Fused Head Expansion Plan

This log captures the outstanding work required to make the fused head read and generate all modalities directly through PTX kernels. Use it to track progress; mark each step as it gets completed.

## Milestone A — Geometry + Galaxy Memory
- [x] A1. Define GPU memory layout for Galaxy nodes, embeddings, and bufferViews.
- [x] A2. Implement PTX kernels to load existing mesh buffers from GLB extras into GPU memory.
- [x] A3. Implement PTX kernels for mesh transforms (scale/rotate/blend) and embedding updates.
- [x] A4. Implement PTX write-back path that emits updated bufferViews/embeddings back into GLB/JSON.
- [x] A5. Extend `PTXGeometryOps` to expose `load_mesh`, `transform_mesh`, `save_mesh` APIs.
- [x] A6. Update `SleepTimeCompute`/trainer to use PTX geometry read/write instead of CPU GLTF edits.

## Milestone B — Text Embedding Operations
- [ ] B1. Implement PTX kernels for token embedding lookup and cosine similarity search.
- [ ] B2. Add optional small PTX attention block for short generative spans (numeric reasoning).
- [ ] B3. Wire fused head text routing so PTX handles numeric/math queries before fallback logits.
- [ ] B4. Update training loop to log PTX text ops and use them in RLWHF scoring.

## Milestone C — Audio/Image/Video Pipelines
- [ ] C1. Prototype PTX audio kernels (e.g., STFT, filtering) and expose via `PTXAudioOps`.
- [ ] C2. Prototype PTX image kernels (texture load/edit) and expose via `PTXImageOps`.
- [ ] C3. Prototype PTX video frame kernels (basic temporal ops) and expose via `PTXVideoOps`.
- [ ] C4. Add fused head routing for audio/image/video queries.

## Milestone D — persistence & tooling
- [ ] D1. Build GPU-aware serializer to commit Galaxy deltas produced by PTX ops.
- [ ] D2. Add regression tests (unit + integration) that execute PTX pipelines end-to-end.
- [ ] D3. Document build/run instructions for new PTX kernels and note minimum GPU requirements.

---

**Progress Log**

| Date (UTC) | Item | Status | Notes |
|------------|------|--------|-------|
| 2025-09-21 | A1 | Done | Drafted GPU Galaxy buffer layout (see below) and encoded offsets in loader metadata. |
| 2025-09-21 | A2 | Done | Added GPU upload scaffold (`galaxy_buffer.py`) to map GLB meshes into device buffers. |
| 2025-09-21 | A3 | Done | PTX geometry kernels + wrappers handle transforms, normal recompute, and embedding blending. |
| 2025-09-21 | A4 | Done | Dirty tracking + GLB/JSON write-back helpers (`save_meshes_to_glb`, `save_embeddings_to_json`). |
| 2025-09-21 | A5 | Done | `PTXGeometrySession` + PTXOps wiring provide load/transform/save API surface. |
| 2025-09-21 | A6 | Done | SleepTimeCompute now loads/saves House GLB via PTX and updates zone meshes/metadata on GPU. |

### GPU Galaxy Buffer Layout (A1 draft)

| Component | Structure | Notes |
|-----------|-----------|-------|
| Node table | `struct Node { uint32 id; uint16 type; uint16 flags; uint32 embedding_offset; uint32 data_offset; }` packed and stored in a single device buffer. | Supports star/ray/resource types; `flags` encode modality. |
| Embedding arena | Contiguous FP32 buffer (`float*`) with per-node offset tracked in node table. | Allows direct PTX access to embeddings for similarity search or updates. |
| Geometry descriptors | `struct MeshView { uint32 vertex_offset; uint32 vertex_count; uint32 index_offset; uint32 index_count; uint32 material_id; }`. | Points into shared vertex/index pools; compatible with GLB bufferView indexing. |
| Vertex pool | `float3` array for all vertices. | Keep SoA option open for PTX kernels that prefer xyz arrays. |
| Index pool | `uint32` array for triangle indices. | Aligns with GLB accessor requirements. |
| Media blobs | For audio/image/video store pointers (`uint64 gpu_ptr`) + metadata (sample rate, resolution). | Initially just references; PTX kernels can stream from pinned staging buffers. |
| Dirty map | Bitset marking modified nodes/meshes. | Used by serializer to emit minimal diffs back to GLB/JSON. |

Next steps (A1 → A2): formalise SoA vs AoS decision for vertices, write helper that maps GLB bufferViews into this layout, and prototype CUDA memcpy wrappers.

Update the status column with `Pending`, `In Progress`, or `Done`, plus short notes (commit hash, branch, etc.).
