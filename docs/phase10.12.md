# PHASE 10.12: AUTO RAYS + GLB + MVP POLISH

## GOAL
- Auto‑generate ray bundles for every new shape.
- Store rays as actual GLB LineSegments (with vertex colors); thickness kept in `extras.k3d.ray_thickness` for future Tube rendering.
- Add an honesty filter control in the viewer UI for rays.
- Ensure manifest includes both shapes and rays; load at startup.

## WHAT’S IMPLEMENTED
- Auto‑rays: `TextTo3DGenerator` calls `RayBundleGenerator` after writing the shape GLB.
- Rays as GLB: `ray_bundle_generator.py` writes `rays_<shape>.glb` with `LINES` primitive, `COLOR_0`, and `extras.k3d`.
- Manifest: `live_server.load_materialized_objects()` scans shapes and rays (GLB/JSON) and writes `/house/materialized_objects/manifest.json`.
- Viewer: loads shapes and ray bundles; slider in bottom‑right adjusts the ray honesty filter; press `R` to reload rays with current filter.

## USAGE
- Generate: `/generate_3d <prompt>` → writes shape GLB + rays GLB.
- Ritual: `/sleep materialize` → refreshes manifest and boot‑loads counts.
- Viewer: shapes/rays load automatically; adjust slider to change ray honesty threshold.

## NOTES
- Ray thickness is stored as metadata; rendering uses LineSegments now. Phase 11 upgrades to TubeGeometry with per‑segment thickness.
- If CUDA is unavailable, vertex generation falls back gracefully.

