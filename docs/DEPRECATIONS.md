# Deprecations — Cranium Core v3.0

This project is transitioning to the Cranium Core v3.0 pipeline where all knowledge and runtime bindings live inside embedded glTF/GLB with `meshes[*].primitives[*].extras.k3d` and direct buffer views.

As part of this transition, the following modules and examples are deprecated. They will remain available during the migration window, but new work should target the embedded glTF path only.

- k3dgen/house.py: Sidecar `.k3d` storage and access helpers.
  - Replacement: use embedded glTF with `extras.k3d` and bufferViews; see `spec/glTF_K3D_extension.md` and `knowledge3d/tools/phase0_export_glb.py`.

- k3dgen/ai_native.py: AI-native extras helpers using `embedding_b64` payloads.
  - Replacement: attach embeddings in binary `BufferView` referenced by `extras.k3d["embeddingsView"]` with `embeddingDims`.

- examples referencing `.k3d` (e.g., `examples/my_house_generator.py`, `examples/*.gltf` with sidecar URIs):
  - Replacement: convert to embedded glTF. If needed, add a `vectorsView` and `embeddingsView` in the same `Buffer` as positions.

Notes
- The sidecar `.k3d` format remains documented for historical reference but is no longer a supported output in generators. See `k3dgen/__main__.py` for the embedded-only CLI.
- New exporters must ensure direct buffer access for the AI client with `extras.k3d.direct_buffer_access = true` and provide `embeddingDims`.

