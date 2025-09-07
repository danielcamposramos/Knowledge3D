TRELLIS Integration (Asset Generation)

Intent
- Use TRELLIS to generate 3D assets (rooms, books, shelves, leaves, trees) and convert them into K3D’s permanent memory (glTF/GLB + extras.k3d).
- Keep the generator decoupled: K3D remains memory‑first; TRELLIS is a pluggable producer.

Adapter
- Tool: `knowledge3d/tools/trellis_adapter.py`
  - `to-k3d`: CSV (+metadata) → K3D GLB using k3dgen internals.
  - `from-mesh`: inject minimal `extras.k3d` into an existing glTF so the viewer can load it immediately (uses POSITION as `vectorsView` and creates ids `v0..`).
  - `gen` (stub): shows how to run TRELLIS and where to place outputs for conversion.

Suggested Workflow
1) Clone TRELLIS under `ext/TRELLIS` and follow its README to set up inference.
2) Generate meshes from prompts or video/image sources into `/k3dlocal/datasets/trellis/`.
3) Convert:
   - If you have embeddings + meta: `trellis_adapter to-k3d --csv ... --metadata ... --out ...`
   - If you only have mesh glTF: `trellis_adapter from-mesh --gltf asset.gltf --out asset.k3d.gltf`
4) Add to viewer via `viewer/public/condo.json` or serve from datasets server and reference by URL.

Notes
- For richer K3D payloads, attach CLIP/CLAP embeddings and thumbnails via our multimodal ingesters, then call `to-k3d`.
- Doors are intra‑house; use metadata to mark rooms/objects and Knowledge Garden hierarchy.

Licensing
- Respect TRELLIS’s license for generated assets and derivative use.

