# Single Galaxy Build — Progress (2025-09-07)

- Unified Galaxy: `viewer/public/galaxy.glb` (audio + video + text)
- Cross‑modal variant: `viewer/public/galaxy.cross.glb` (added edges)
- Source GLBs:
  - `viewer/public/clotho.glb` (audio)
  - `viewer/public/vatex_2k.glb` (video)

Commands
```bash
# Build unified Galaxy (CPU PCA; small dataset)
python3 -m knowledge3d.tools.unify_glbs \
  viewer/public/clotho.glb:audio \
  viewer/public/vatex_2k.glb:video \
  viewer/public/text_demo.glb:text \
  --out viewer/public/galaxy.glb --dims 128 --k 8 --reducer pca

# Add cross‑modal edges (unlike‑modality links from kNN)
python3 -m knowledge3d.tools.add_crossmodal_edges \
  --input viewer/public/galaxy.glb \
  --out   viewer/public/galaxy.cross.glb

# Evaluate
python3 -m knowledge3d.tools.eval_modal_homophily --gltf viewer/public/galaxy.glb --out docs/reports/status/galaxy_modal_homophily.json
python3 -m knowledge3d.tools.eval_modal_homophily --gltf viewer/public/galaxy.cross.glb --out docs/reports/status/galaxy_crossmodal@8.json
python3 -m knowledge3d.tools.eval_routing --gltf viewer/public/galaxy.glb --pairs 64 --out docs/reports/status/routing-galaxy.json
python3 -m knowledge3d.tools.eval_routing --gltf viewer/public/galaxy.cross.glb --pairs 64 --out docs/reports/status/routing-galaxy-cross.json
python3 -m knowledge3d.tools.reflect_glb --gltf viewer/public/galaxy.glb --out docs/reports/status
```

Results
- Nodes: 5,876 — Dims: 128
- Neighbors (k): 8
- Homophily (modality):
  - Mean: 0.7899 — Median: 0.8750
  - Cross edges (after cross‑modal linking): audio↔video ≈ 9,735; audio↔text 127; text↔video 14
- Routing (64 random pairs):
  - BFS median hops: 4 — avg ~3.3–3.5 ms
  - A* median hops: 12 — avg ~31 ms
  - A* LOD success: 94% — median hops: 14 — avg ~32 ms
- Reflection: see latest `docs/reports/status/k3d_reflection-*.md`

Viewer
- Load default: `/galaxy.glb`
- Cross‑modal edges: `?gltf=/galaxy.cross.glb`
- Near‑field shapes + rays enabled with spacing‑aware scales/lengths.

Notes
- For larger builds, include images (COCO) and text (compendium) in `unify_glbs` inputs; PCA will pad/truncate dims and merge.
- Keep large GLBs local per `docs/LARGE_ASSETS.md`.
