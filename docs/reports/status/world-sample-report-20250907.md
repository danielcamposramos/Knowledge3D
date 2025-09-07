# World Sample — Unified Galaxy Evaluation (2025‑09‑07)

Audience: non‑technical first, technical appendix below. This measures how a single, unified Galaxy (one memory for all data types) behaves on a small “world of everything” theme.

House Terms
- Galaxy: the big starfield — one place where all items live (audio, video, text, images).
- Stars: items (clips, frames, sentences) placed by meaning, not by file type.
- Rays: short colored beams that show which data types each star has.

## What We Tested (Plain English)
- We built one Galaxy that mixes three kinds of knowledge: audio (sounds), video (short clips), and text (sentences from this repo).
- We kept it small and focused on a theme (e.g., rain, street, car, city, child, speech) so we can see if different data types “meet” by meaning.
- We ran three checks:
  - Mixing: do neighbors mostly share the same topic, and can different types still meet?
  - Navigation: how many “hops” does it take to move between random items?
  - Retrieval: can it quickly find the most similar items?

## Results (Plain English)
- It’s one Galaxy now: everything lives in the same space.
- Mixing looks good: neighbors usually share the same topic; at the same time, we added modest bridges so audio↔video↔text can meet where appropriate.
- Walking around is easy: it takes about four short steps to go from one random item to another (on this tiny world).
- Finding similar things is exact and fast for this size (we used an “exact” search here, comparable to a precise look‑up table for small sets).

What this means: this unified memory lets an AI talk about a topic and naturally pull in related audio, video, and text in one place. It’s a better fit for conversations and tool use because it doesn’t have to jump across separate silos.

## Comparison to “Standard AI” at Similar Size
Typical setups keep separate indexes per data type and glue them together later. That works, but the system has to translate between silos. Here we put everything in one space by meaning. On this small world:
- Both approaches can find neighbors quickly (the set is small), but the unified Galaxy makes cross‑type hops explicit (the added bridges) and local (few steps).
- In the “standard” view (no explicit bridges), crossing types still happens through meaning, but the path is less obvious and a bit more fragile. With bridges on, it’s easier for the agent to explain and traverse (“go from rain sound → rainy street video → text about rain”).
- For larger worlds, this will matter more: a single space reduces re‑indexing and translation steps and lets training logs directly improve the same memory.

## Technical Snapshot
- Size: 5,876 items — embeddings dimension 128 (after PCA); neighbors per node k = 8.
- Cross‑type connections after we add bridges: audio↔video ≈ 9,735; audio↔text 127; text↔video 14.
- Navigation (64 random pairs):
  - BFS median hops: 4 (≈3.3–3.5 ms per query on CPU)
  - A* median hops: 12 (≈31 ms)
  - A* LOD: 94% success; median 14 hops (≈32 ms)
- Retrieval (exact, CPU): recall@10 = 1.0 (exact search) for both the base Galaxy and the bridged one.

## Example “Hubs” the system noticed
- engine start BMW 320 VG91.wav
- Rainy weather.wav
- Hyeres street sounds child.wav
- 120124_busy-street-corner.wav
- Hail and Rain in the Courtyard.wav

These are simply the most connected items in this small world; in a richer world this would include images and short texts too.

## How We Built It (Repro)
- Scripted sample: `scripts/build_world_sample.sh` (meaning‑aligned subsets from text/images/audio/video → one Galaxy → bridges).
- For this run, we also tested directly unifying existing small GLBs (`clotho.glb`, `vatex_2k.glb`, `text_demo.glb`).

Key commands used (for the unified run):
```bash
# Unify GLBs into a single Galaxy
python -m knowledge3d.tools.unify_glbs \
  viewer/public/clotho.glb:audio \
  viewer/public/vatex_2k.glb:video \
  viewer/public/text_demo.glb:text \
  --out viewer/public/galaxy.glb --dims 128 --k 8 --reducer pca

# Add explicit cross‑modal edges
python -m knowledge3d.tools.add_crossmodal_edges \
  --input viewer/public/galaxy.glb \
  --out   viewer/public/galaxy.cross.glb

# Evaluate (write JSON under docs/reports/status/)
python -m knowledge3d.tools.eval_modal_homophily --gltf viewer/public/galaxy.glb --out docs/reports/status/galaxy_modal_homophily.json
python -m knowledge3d.tools.eval_modal_homophily --gltf viewer/public/galaxy.cross.glb --out docs/reports/status/galaxy_crossmodal@8.json
python -m knowledge3d.tools.eval_routing --gltf viewer/public/galaxy.glb --pairs 64 --out docs/reports/status/routing-galaxy.json
python -m knowledge3d.tools.eval_routing --gltf viewer/public/galaxy.cross.glb --pairs 64 --out docs/reports/status/routing-galaxy-cross.json
env K3D_STRICT_GPU=0 K3D_ACCEL=cpu python -m knowledge3d.tools.eval_retrieval --gltf viewer/public/galaxy.glb --k 10 --queries 256 --ann flat --out docs/reports/status/retrieval-galaxy.json
env K3D_STRICT_GPU=0 K3D_ACCEL=cpu python -m knowledge3d.tools.eval_retrieval --gltf viewer/public/galaxy.cross.glb --k 10 --queries 256 --ann flat --out docs/reports/status/retrieval-galaxy-cross.json
```

## What to Improve Next
- Grow the world with a few hundred images and matched text (WIT/COCO) to help the agent cite helpful pictures while talking.
- Add ARC/HLE “exam” doors and labels so we can test question‑answer flows end‑to‑end.
- Train a tiny world model on navigation logs so the agent predicts its next step in this space (improves path planning under uncertainty).

## Appendix: Where to Find the Numbers
- Homophily: `docs/reports/status/galaxy_modal_homophily.json`, `galaxy_crossmodal@8.json`
- Routing: `docs/reports/status/routing-galaxy.json`, `routing-galaxy-cross.json`
- Retrieval: `docs/reports/status/retrieval-galaxy.json`, `retrieval-galaxy-cross.json`
- Run summary: `docs/reports/status/progress-20250907-single-galaxy.md`

