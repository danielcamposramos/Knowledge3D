Balanced Galaxy v7 — Modality-Balanced Sample (GPU)
===================================================

Summary
-------
- Built a small, balanced Galaxy with equal counts per modality:
  - text (exaone3.5 via Ollama): 55 lines → `viewer/public/text_exaone_v1.glb`
  - 3D (glTF samples subset): 55 assets → `viewer/public/shapes_index_small.glb`
- Unified into `viewer/public/galaxy.v7.glb` (n=110; effective dims=109), and added cross‑modal edges → `viewer/public/galaxy.v7.cross.glb`.
- Intent: validate cross‑modal navigation and density using low‑dimension, high‑density embeddings under GPU acceleration.

Why dims=109?
- Unify projects embeddings to a common dimensionality via PCA across the stacked embeddings. For very small n, the effective rank is capped by n‑1. Here n=110 → max effective dims = 109 even if `--dims 256`.

Commands
--------
```bash
# Generate topic‑coherent text via local Ollama exaone3.5
scripts/k3d_env.sh run python -m knowledge3d.tools.gen_text_ollama \
  --ollama http://192.168.0.4:11434 --model exaone3.5:latest \
  --topics "animals,sports,vehicles,gardens,tools" --n 80 \
  --out ../Knowledge3D.local/datasets/exaone_text_v1.txt

# Text → GLB (GPU ST + UMAP)
scripts/k3d_env.sh run python -m k3dgen \
  --text ../Knowledge3D.local/datasets/exaone_text_v1.txt \
  --gltf viewer/public/text_exaone_v1.glb --k 10 --reducer umap \
  --model sentence-transformers/all-MiniLM-L6-v2 --emb-precision f16

# 3D subset (55) → index GLB
mkdir -p ../Knowledge3D.local/datasets/gltf_samples_small
(cd ../Knowledge3D.local/datasets/gltf_samples && ls *.glb | head -n 55 | \
  xargs -I{} ln -s "$PWD/{}" ../gltf_samples_small/{})
scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_open3d \
  --root ../Knowledge3D.local/datasets/gltf_samples_small \
  --out viewer/public/shapes_index_small.glb --pattern ".glb" --reducer umap

# Unify + cross‑modal edges
scripts/k3d_env.sh run python -m knowledge3d.tools.unify_glbs \
  viewer/public/text_exaone_v1.glb:text \
  viewer/public/shapes_index_small.glb:3d \
  --out viewer/public/galaxy.v7.glb --dims 256 --k 10 --reducer umap
scripts/k3d_env.sh run python -m knowledge3d.tools.add_crossmodal_edges \
  --input viewer/public/galaxy.v7.glb --out viewer/public/galaxy.v7.cross.glb
```

Next Steps
----------
- Extend balance to audio/video/images:
  - Fetch small slices with `knowledge3d.tools.hf_fetch_multimodal` (audio/video); for images, prepare a CSV of paths+captions.
  - Ingest with `ingest_audio` / `ingest_video` and convert to GLB via `knowledge3d.tools.trellis_adapter to-k3d`.
  - Re‑unify with matched counts per modality.
- Increase per‑topic coherence across modalities (e.g., vehicles, sports, gardens) to stress‑test cross‑modal retrieval and navigation.

