# PHASE 11: RAY EMBODIMENT & INTERACTION

## GOAL
Upgrade rays from flat LineSegments to volumetric tubes with per‑segment thickness; add show/hide toggle, tooltips on hover, and click logging; maintain honesty filtering.

## IMPLEMENTATION
- Volumetric rays: TubeGeometry built per segment for JSON bundles; GLB line segments converted to tubes at load time using extras.k3d.ray_thickness.
- UI controls:
  - Slider (bottom‑right) controls ray honesty filter (default 0.7). Press `R` to reload with new threshold.
  - Toggle button (bottom‑right) to show/hide all rays.
- Interaction: pointer hover shows a small tooltip with modality, honesty, and thickness; click logs full details in console.

## FILES
- viewer/src/main.ts — loads rays, builds TubeGeometry, adds slider + toggle, hover/click metadata.
- knowledge3d/cranium/phase10/ray_bundle_generator.py — writes GLB with LINES + extras.k3d (ray_thickness, embedding_preview, honesty_score).

## USAGE
- `/generate_3d <prompt>` → creates shape GLB and auto‑rays GLB in the Garden (Zone 5).
- `/sleep materialize` → refreshes manifest and memory.
- Viewer opens and auto‑loads shapes + rays. Use the slider and toggle to explore rays; hover/click for details.

