# World of Everything — Small Unified Galaxy

Goal: build a tiny, meaning‑aligned sample across modalities (text, images, audio, video) and unify it into one Galaxy for AI+human tests. This provides a compact “world of everything” to probe cross‑modal reasoning, ARC‑style tasks, and conversational grounding.

Inputs (if present; each step skips gracefully when missing)
- Images (COCO): `coco.train.clip.csv`, `coco.train.meta.json`
- Audio (Clotho): `clotho.clap.csv`, `clotho.meta.json`
- Video (VATEX): `vatex.clip.csv`, `vatex.meta.json`
- Text: sampled from repo docs (filtered by keywords)

Keywords (theme)
- Default: `rain,street,car,city,child,speech`
- Override by setting `KEYWORDS="keyword1,keyword2,..."`

One‑liner (CPU‑friendly)
```bash
# Env is optional but recommended; see docs/ENV_POLICY.md
BASE=../Knowledge3D.local/datasets \
KEYWORDS="rain,street,car,city,child,speech" \
scripts/build_world_sample.sh
```

What it does
1) Text: collects lines from repo docs matching `KEYWORDS`, vectorizes to `DIMS` (default 128), builds `viewer/public/_world/text.glb`.
2) Images/Audio/Video: filters vectors+metadata by keywords to ≤ `SAMPLE_MAX` items each, builds per‑modality GLBs under `viewer/public/_world/`.
3) Unify: merges available per‑modality GLBs into a single Galaxy: `viewer/public/galaxy.glb`.
4) Cross‑modal edges: writes `viewer/public/galaxy.cross.glb` linking nearest unlike‑modality neighbors.

Tune knobs
- `KEYWORDS`: meaning space (comma‑separated). Example themes: `weather`, `transport`, `school`, `kitchen`, `animals`.
- `SAMPLE_MAX` (default 800): per‑modality cap after filtering.
- `DIMS` (default 128): common embedding dimension pre‑reduction.
- `K` (default 8): neighbors per node.
- `REDUCER` (default pca): 3D layout (`pca|umap|tsne`).

Viewer
- Default: open the viewer; it loads `/galaxy.glb` automatically if present.
- Edge view: `?gltf=/galaxy.cross.glb` to display modality‑bridging edges.

Under the hood
- Filtering tool: `knowledge3d/tools/filter_modal_csv.py`
- Unifier: `knowledge3d/tools/unify_glbs.py` (pads dims → PCA → neighbors → GLB)
- Cross‑edges: `knowledge3d/tools/add_crossmodal_edges.py`
- Visuals: near‑field instanced stars + rays with spacing‑aware scales/lengths and per‑format thickness/gradients.

Notes
- Keep large datasets under `../Knowledge3D.local/datasets`. See `docs/LARGE_ASSETS.md` for ingestion.
- The sample is intentionally small to be interactive; scale up by raising `SAMPLE_MAX` and adding WIT or other sources.

