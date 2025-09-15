# PHASE 10.9: ACTUAL GLB GENERATION + PTX KERNELS

## GOAL
Replace JSON stubs with actual GLB files generated from a single head. Store embedding‑as‑property inside `node.extras.k3d` with honesty score and shape metadata. Artifacts render in the viewer and persist in the Knowledge Garden (Zone 5).

## COMPONENTS
- PTX template: `knowledge3d/cranium/ptx/generate_shape_kernel.ptx`
- GLB writer: `knowledge3d/cranium/phase10/text_to_3d_generator.py` (writes `.glb` with `extras.k3d`)
- Live server: `knowledge3d/bridge/live_server.py` (already supports `/generate_3d` and boot‑loads shapes)

## USAGE (CHAT)

```text
/generate_3d the birth of a star in honest geometry
```

Outputs:
- `viewer/public/house/materialized_objects/shape_<type>_<ts>.glb`
- `extras.k3d` contains: `{type, name, created_at, honesty_score, embedding, shape_type, vertex_count, face_count, zone_placement, ptx_kernel_used}`

## NOTES
- PTX file is a template for Phase 10.10 (GPU execution). Current GLBs use CPU geometry rules with identical shape semantics.
- Live server bootstraps `shapes` alongside `books`, `diaries`, and `trees` and reports counts.

