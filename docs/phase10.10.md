# PHASE 10.10: GPU EXECUTION + THREE.JS RENDERING

## GOAL
Run PTX‑style vertex generation on GPU (mocked via torch.jit in this phase), write GLBs with `extras.k3d`, and render generated shapes in the Knowledge Garden with honesty filtering.

## COMPONENTS
- `knowledge3d/cranium/ptx/generate_shape_kernel.ptx` — PTX template for vertex generation
- `knowledge3d/cranium/phase10/ptx_kernel_loader.py` — PTX loader (torch.jit mock for now)
- `knowledge3d/cranium/phase10/text_to_3d_generator.py` — attempts GPU vertex gen; writes actual GLB with `extras.k3d`
- `knowledge3d/bridge/live_server.py` — writes materialized shapes manifest; boot‑loads counts
- `viewer/src/main.ts` — loads and renders shapes from `/house/materialized_objects/manifest.json` with honesty filter

## USAGE
- Generate: `/generate_3d <prompt>` in live chat
- Sleep ritual: `/sleep materialize` — also refreshes manifest
- Viewer loads shapes automatically on start; filter defaults to `honesty >= 0.7`.

## NOTES
- PTX execution is mocked via `torch.jit` in this phase. Full driver API to follow.
- GLBs store embedding, honesty_score, shape_type in `node.extras.k3d`.

