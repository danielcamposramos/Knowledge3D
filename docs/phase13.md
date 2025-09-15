# PHASE 13: AUTO‑CURATED MEANING SYNTHESIS

## GOAL
Enable autonomous meaning synthesis and self‑curated training during sleep‑time compute so the AI invents new connections, materializes synthesized shapes, and generates its own training prompts.

## COMPONENTS
- `knowledge3d/cranium/phase13/auto_synthesis_engine.py`
  - Loads stars from Galaxy GLB (`extras.k3d`) with embeddings + honesty.
  - Finds high‑similarity, high‑honesty pairs (cosine ≥ 0.7, honesty ≥ 0.6).
  - Fuses embeddings, chooses a shape type, and writes a synthesized shape GLB with `extras.k3d`:
    `{ type: 'synthesized_shape', name, created_at, honesty_score, embedding, shape_type, vertex_count, face_count, source_stars[], similarity, zone_placement }`.
  - Auto‑generates ray bundles (GLB) for the synthesized shape.
- `knowledge3d/cranium/phase13/self_curriculum_engine.py`
  - Scans Galaxy stars for low honesty (< 0.5) and generates self‑curated training queries.
- `knowledge3d/cranium/phase10/sleep_time_compute.py`
  - Integrates synthesis + curriculum after materialization; logs results into `sleep_time_adjustments.json`.
- `knowledge3d/bridge/live_server.py`
  - Manifest now includes synthesized shapes.

## USAGE
- Run the sleep ritual:
```text
/sleep materialize
```
Expected console:
- `🧠 Running Autonomous Meaning Synthesis...`
- `✨ Synthesized: Synthesis: <starA> + <starB> → viewer/public/house/materialized_objects/synth_... .glb`
- `📚 Generating Self‑Curated Curriculum...`
- `🎓 Self‑Training on: <query>`

## OUTPUT
- New GLBs under `viewer/public/house/materialized_objects/` for synthesized shapes and their ray bundles.
- Manifest (`/house/materialized_objects/manifest.json`) includes them; viewer auto‑loads.

## NEXT
Phase 14: Dream shapes — generate geometry purely from internal state during sleep (Zone 6: Dream Chamber) with honesty‑weighted embedding drift.

