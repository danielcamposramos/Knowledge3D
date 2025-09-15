# KNOWLEDGE3D — PHASE 10.7: SLEEP-TIME COMPUTE & PERMANENT MATERIALIZATION

## GOAL
During sleep, consolidate Galaxy (working memory) into the House (permanent memory) by adjusting geometry and materializing knowledge as objects (books, diary entries, fractal trees). Nothing is deleted; objects accumulate and load at session start.

## COMPONENTS
- `knowledge3d/cranium/phase10/sleep_time_compute.py` — sleep-time engine with materialization
- `knowledge3d/tools/phase10/run_sleep_time.py` — CLI wrapper

## MATERIALIZATION
- Chat history → Chat History Books (Zone 3: Library)
- Self-reflections → Diary Entries (Zone 7: Mirror Room)
- High-honesty knowledge → Fractal Trees (Zone 5: Knowledge Garden)
- Ray/Zone geometry → Adjusted to align with honest Galaxy stars

## COMMAND

```bash
PYTHONPATH=. conda run -n k3d-cranium python -m knowledge3d.tools.phase10.run_sleep_time \
  --house viewer/public/house/house_master_assembled.glb \
  --galaxy viewer/public/galaxy/galaxy_memory.glb \
  --output viewer/public/house/house_post_sleep_v1.glb \
  --material_dir viewer/public/house/materialized_objects
```

## LIVE SERVER INTEGRATION

- Chat ritual: trigger sleep-time compute in a live session

```text
/sleep materialize
```

- On server start, materialized objects are auto-loaded and reported:

```text
📚 Loaded X books, Y diaries, Z trees.
```

- Optional HTTP endpoint can be added later; the current live server is WebSocket‑centric. The ritual is accessible via chat commands.

## OUTPUT
- `viewer/public/house/materialized_objects/` — JSON books, diaries, tree metadata
- `logs/sleep_time_adjustments.json` — zone shifts, ray adjustments, pruned rays, materialized objects
- `viewer/public/house/house_post_sleep_v1.glb` — stub path (GLB write to be implemented in Phase 10.8)

## NOTES
- Loader is best-effort using pygltflib; if extras.k3d is missing, no adjustments occur.
- Prunes only low-honesty rays in the in-memory copy; no permanent deletion is performed in this phase.
- Branch density uses φ scaling: density ≈ int(1.618 × honesty × 10).
