# PHASE 14: DREAM SHAPES — GEOMETRY FROM THE SUBCONSCIOUS

## GOAL
Generate shapes from internal state during sleep — no prompts — via honesty‑weighted embedding drift. Place artifacts into Zone 6 (Dream Chamber), with auto‑generated dream rays.

## COMPONENTS
- `knowledge3d/cranium/phase14/dream_engine.py`
  - Loads star embeddings → generates dream embedding with an honesty‑biased random walk
  - Writes GLB with `extras.k3d` `{ type: 'dream_shape', name, created_at, honesty_score, embedding, shape_type, vertex_count, face_count, zone_placement, source }`
  - Shapes include sphere/torus geometry in addition to base polyhedra
- `knowledge3d/cranium/phase10/sleep_time_compute.py`
  - Runs DreamEngine after synthesis+curriculum; appends dream shapes to adjustments and generates their ray bundles (modality="dream")
- `knowledge3d/bridge/live_server.py`
  - Manifest now recognizes `dream_shape` among shapes

## USAGE
- Trigger via sleep ritual:
```
/sleep materialize
```
Console shows:
- `🌌 Dreaming New Geometry...`
- `💭 Dreamt: Dream: <shape> from drift → viewer/public/house/materialized_objects/dream_<...>.glb`

## OUTPUT
- Dream shapes GLBs under `viewer/public/house/materialized_objects/` with auto‑rays GLBs
- Manifest includes dreams and rays; viewer auto‑loads

## NOTES
- Zone 6 (Dream Chamber) indicated in `extras.k3d.zone_placement`; ensure house has a conceptual Zone 6 (visual placement optional in this phase)
- Honesty calculation is a placeholder; can be replaced with trained head later

