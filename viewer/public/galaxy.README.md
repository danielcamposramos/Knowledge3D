This file documents how to (re)create `viewer/public/galaxy.glb` locally.

Why missing?
- `galaxy.glb` typically exceeds Git hosting limits (>99MB), so it is intentionally excluded from Git.
- Keep heavy artifacts under `../Knowledge3D.local/datasets/` as per `docs/LARGE_ASSETS.md`.

Quick options
- Minimal demo (few points; CPU OK):
  1) Create a tiny corpus `data/demo.txt` with a few lines of text.
  2) Build vectors: `python3 -m knowledge3d.tools.text_to_vectors --text data/demo.txt --out ../Knowledge3D.local/datasets/demo_vectors.csv --dims 128`
  3) Build GLB: `python3 -m k3dgen ../Knowledge3D.local/datasets/demo_vectors.csv --gltf viewer/public/galaxy.glb --k 5 --reducer pca`

- Unified Galaxy from multiple modalities (recommended):
  Use the merger tool to combine text/image/audio/video CSV+metadata into one GLB:
  
  ```bash
  # Example: combine three local CSVs under ../Knowledge3D.local/datasets/
  python3 -m knowledge3d.tools.build_galaxy \
    --out viewer/public/galaxy.glb --dims 256 --k 10 --reducer pca \
    text:../Knowledge3D.local/datasets/ai_compendium_80k_vectors.csv \
    image:../Knowledge3D.local/datasets/coco.train.clip.csv:../Knowledge3D.local/datasets/coco.train.meta.json \
    audio:../Knowledge3D.local/datasets/clotho.clap.csv:../Knowledge3D.local/datasets/clotho.meta.json
  ```

End‑to‑end (80k compendium)
- Follow `docs/LARGE_ASSETS.md` → “80k AI Compendium (local)” to build the text corpus and vectors.
- Then either:
  - Build directly with `k3dgen` from the vectors CSV, or
  - Use `knowledge3d.tools.build_galaxy` (above) to merge with other modalities.

Environment
- Prefer running inside the managed env described in `docs/ENV_POLICY.md` (e.g., `conda run -n k3dml env PYTHONPATH=. python -m ...`).
- For large builds, keep outputs in `../Knowledge3D.local/datasets/` and only copy/link `viewer/public/galaxy.glb` when you need to view it locally.

Notes
- The viewer auto-loads `/galaxy.glb` if present; otherwise it tries other samples and remains usable.
- Do not commit `galaxy.glb`. It is ignored by `.gitignore` by design.

