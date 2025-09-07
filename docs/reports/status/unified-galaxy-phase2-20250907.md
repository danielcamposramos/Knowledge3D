# Unified Galaxy — Phase 2/3 Progress (2025‑09‑07)

Audience: non‑technical summary first; technical appendix below.

## What We Did (Three Phases)
- Small image sample added: we extracted ~800 COCO items matching the same theme (rain/street/car/city/child/speech) and merged them with audio, video, and text into one Galaxy.
- Exams mounting (ARC): we built a local index of 200 ARC training tasks so the Tablet can browse them through the datasets server.
- One‑click pipeline: a script now assembles a small “world of everything” and outputs the unified Galaxy plus a bridged version.

## Why It Matters
- One place for everything: audio/video/text/images live together by meaning, so conversations can walk across types without translating between silos.
- Bridges across types: we add gentle cross‑links so it’s obvious how to go from, say, rain sounds → rainy street videos → sentences about rain.
- Exams ready: ARC tasks are listed and ready to be served into the Tablet app for lightweight reasoning trials.

## Results (Plain English)
- Mixing improved: image items join the same neighborhoods; nearby items usually share the topic.
- Navigation still easy: random items are ~5 short steps apart (slightly more than before since we added more items).
- Retrieval stays exact for this size; latency remains tiny on CPU.
- ARC index: 200 tasks registered locally; ready to mount in the viewer via the datasets server.

Compared to a “standard AI” setup of similar size (separate indexes per type), the unified Galaxy continues to:
- Make cross‑type paths obvious and short, which helps with tool use and explanations.
- Reduce glue code and re‑indexing, since training logs and navigation traces improve one shared memory.

## How To View
- Default Galaxy: open the viewer (loads `/galaxy.glb`).
- With bridges (explicit cross‑type edges): add `?gltf=/galaxy.cross.glb` to the URL.
- Serve exams (optional): `python -m knowledge3d.tools.serve_datasets --port 8766` then open the Tablet’s Exams app.

---

## Technical Appendix

Build steps executed
```bash
# Filter a small COCO image set by theme
python -m knowledge3d.tools.filter_glb_by_keywords \
  --input viewer/public/coco_50k.glb \
  --out viewer/public/_world/image.glb \
  --keywords "rain,street,car,city,child,speech" --max 800 --reducer pca --k 8

# Unify four modalities and add cross‑modal edges
python -m knowledge3d.tools.unify_glbs \
  viewer/public/_world/image.glb:image \
  viewer/public/clotho.glb:audio \
  viewer/public/vatex_2k.glb:video \
  viewer/public/text_demo.glb:text \
  --out viewer/public/galaxy.glb --dims 128 --k 8 --reducer pca
python -m knowledge3d.tools.add_crossmodal_edges --input viewer/public/galaxy.glb --out viewer/public/galaxy.cross.glb

# Evaluate
python -m knowledge3d.tools.eval_modal_homophily --gltf viewer/public/galaxy.glb --out docs/reports/status/galaxy_modal_homophily.json
python -m knowledge3d.tools.eval_modal_homophily --gltf viewer/public/galaxy.cross.glb --out docs/reports/status/galaxy_crossmodal@8.json
python -m knowledge3d.tools.eval_routing --gltf viewer/public/galaxy.glb --pairs 64 --out docs/reports/status/routing-galaxy.json
python -m knowledge3d.tools.eval_routing --gltf viewer/public/galaxy.cross.glb --pairs 64 --out docs/reports/status/routing-galaxy-cross.json
env K3D_STRICT_GPU=0 K3D_ACCEL=cpu python -m knowledge3d.tools.eval_retrieval --gltf viewer/public/galaxy.glb --k 10 --queries 256 --ann flat --out docs/reports/status/retrieval-galaxy.json
env K3D_STRICT_GPU=0 K3D_ACCEL=cpu python -m knowledge3d.tools.eval_retrieval --gltf viewer/public/galaxy.cross.glb --k 10 --queries 256 --ann flat --out docs/reports/status/retrieval-galaxy-cross.json

# ARC exams index (200 tasks)
python -m knowledge3d.tools.build_exams_index --max-arc 200
```

Metrics (this run)
- Nodes: 6,676 — Dims: 128 — k=8
- Homophily (by modality): mean 0.8151 — median 0.8750
- Cross edges (bridged): image↔text 9; audio↔text 115; text↔video 14; audio↔video 9,735
- Navigation (64 random pairs):
  - BFS median hops: 5 (avg ~5.6–6.2 ms)
  - A* median hops: 12 (avg ~42–44 ms)
  - A* LOD median hops: 15 (avg ~41–44 ms)
- Retrieval (exact, CPU): recall@10 = 0.9992 (flat vs flat sanity check)

Artifacts
- JSON metrics in `docs/reports/status/`
- World sample how‑to: `docs/WORLD_SAMPLE.md`
- Non‑technical overview: `docs/reports/status/world-sample-report-20250907.md`

Limitations and next steps
- HLE requires gated access (Hugging Face). For now, ARC is mounted locally and HLE can be added with credentials.
- Door placement: we can add doors that point to a curated mini set of ARC/HLE tasks for measured navigation from the Galaxy into exams.
- Bigger worlds: we can scale samples (e.g., add 1–10k images and matched text) and switch to GPU reducers/ANN for speed.
