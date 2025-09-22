# PTX Fused Head Expansion Plan

This log captures the outstanding work required to make the fused head read and generate all modalities directly through PTX kernels. Use it to track progress; mark each step as it gets completed.

## Milestone A — Geometry + Galaxy Memory
- [x] A1. Define GPU memory layout for Galaxy nodes, embeddings, and bufferViews.
- [x] A2. Implement PTX kernels to load existing mesh buffers from GLB extras into GPU memory.
- [x] A3. Implement PTX kernels for mesh transforms (scale/rotate/blend) and embedding updates.
- [x] A4. Implement PTX write-back path that emits updated bufferViews/embeddings back into GLB/JSON.
- [x] A5. Extend `PTXGeometryOps` to expose `load_mesh`, `transform_mesh`, `save_mesh` APIs.
- [x] A6. Update `SleepTimeCompute`/trainer to use PTX geometry read/write instead of CPU GLTF edits.
- [x] A7. Generate PTX-ready base language galaxies and integrate them into default training flows.
- [x] A8. Ensure House exports reference `memory_house.glb` (no JSON fallback) and all PTX loaders/writers assume binary GLBs.
- [ ] A9. Build PTX learning memory for teacher tags + generated insights (GLB + PTX cosine).

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
| 2025-09-21 | A7 | Done | Language galaxies generated (EN/PT/ES/ZH) with PTX cosine retrieval wired into fused head. |
| 2025-09-21 | A8 | Done | House asset rebuilt as binary GLB; codebase now references `memory_house.glb` exclusively. |
| 2025-09-21 | A9 | In Progress | Plan drafted: log teacher tags into JSONL, convert to GLB, load during fused head init. |

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

### Language Galaxy Builder
- Tool: `python -m knowledge3d.tools.language_galaxy_builder --input viewer/public/galaxy/working/lexicon_pt_br_kaikki.jsonl --input viewer/public/galaxy/working/lexicon_audio_pt_br_librispeech9h.jsonl --language-id pt-BR --label "Portuguese (BR) Language Galaxy" --out viewer/public/galaxy/pt_br_language.glb --manifest viewer/public/galaxy/pt_br_language.json`
- Emits binary GLBs with `vectorsView`/`embeddingsView` so PTX loads them directly; manifests capture bounding boxes + modality counts for placement in the language quadrant.
- Use `python -m knowledge3d.tools.convert_gltf_to_glb src.gltf dst.glb` for generic GLTFs with valid bufferViews, otherwise rebuild Houses via the tool below before PTX-only pipelines run.
- Rebuild legacy Houses via `python -m knowledge3d.tools.rebuild_house_glb viewer/public/houses/<id>/memory_house.gltf viewer/public/houses/<id>/memory_house.glb` and delete the JSON `.gltf` to avoid CPU fallbacks.
## Milestone L — Learning Memory + PTX Generation
- [ ] L1. Log teacher tags, feedback, and generated insights into standard JSONL (learning_memory.jsonl).
- [ ] L2. Build `learning_memory_builder` to convert the JSONL into a PTX-ready GLB with metadata.
- [ ] L3. Update fused head to log teacher tags into learning memory and consult it via PTX cosine before neural fallback.
- [ ] L4. Extend trainer to rebuild/reload learning GLB between sessions.
- [ ] L5. Add sanity check ensuring new memories can be appended and reloaded without restart.

