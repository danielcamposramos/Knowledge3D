# PHASE 10.8: TEXT‑TO‑3D GENERATION FROM SINGLE HEAD

## GOAL
Generate 3D shape artifacts from text prompts using a single cognitive head (mocked). Shapes are persisted as JSON metadata now (Phase 10.9 replaces with GLBs), placed in Zone 5 (Knowledge Garden), and auto‑loaded at session start.

## COMPONENTS
- `knowledge3d/cranium/phase10/text_to_3d_generator.py` — generator (PTX‑style geometry rules)
- `knowledge3d/bridge/live_server.py` — `/generate_3d` chat command

## USAGE (CHAT)

```text
/generate_3d quantum entanglement as a fractal tree
```

Output:
- `viewer/public/house/materialized_objects/shape_<type>_<ts>.json`
- Console chat: path to saved artifact and zone placement
- Auto‑loaded into `server.materialized_memory['shapes']`

## NOTES
- Honesty gate: prompts with low mock honesty score (< 0.7) are rejected.
- Shape types (mock): tetrahedron, cube, octahedron, icosahedron, dodecahedron.
- Geometry uses parametric rules; PTX kernel emission and GLB writing land in Phase 10.9.

