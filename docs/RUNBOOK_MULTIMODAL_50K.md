# Multimodal 50k Runbook (Operator Guide)

Purpose
- Produce small but representative GLBs for images (COCO), audio (Clotho), and video (VATEX), plus cross‑modal matching to curate a tri‑modal pool.
- Keep commands deterministic, logged, and easy to re‑run.

Outputs
- Per‑modality GLBs: `viewer/public/{coco_50k,clotho,vatex_2k}.glb`
- Unified Galaxy: `viewer/public/galaxy.glb` (one virtual space)
- Cross‑modal: `../Knowledge3D.local/datasets/matched/{matches.jsonl,pool.txt}`

Pre‑Requisites
- GPU with recent NVIDIA driver (CUDA ≥ 12)
- Conda environment with PyTorch CUDA available
- Datasets mounted locally
  - COCO: `/home/daniel/K3D_llama_cpp/datasets/coco_raw/train2017/train2017/*.jpg`
  - COCO captions: `/home/daniel/K3D_llama_cpp/datasets/coco_raw/annotations/annotations/captions_train2017.json`
  - Clotho WAVs:
    - `/home/daniel/K3D_llama_cpp/datasets/clotho_raw/clotho_audio_development/development/*.wav`
    - `/home/daniel/K3D_llama_cpp/datasets/clotho_raw/clotho_audio_validation/validation/*.wav`
  - VATEX videos: `/home/daniel/K3D_llama_cpp/datasets/vatex_raw/media/*.{mp4,mkv,webm}`

Environment
- Pick the env explicitly to avoid surprises from auto‑activation:
  - `export K3D_CONDA_ENV=k3dml` (GPU env)
  - Optional (RAPIDS): `export K3D_CONDA_ENV=k3d-rapids`
- Validate GPU:
  - `nvidia-smi`
  - `scripts/k3d_env.sh run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`

Paths
- `RAW=/home/daniel/K3D_llama_cpp/datasets`
- `BASE=../Knowledge3D.local/datasets`
- `LOGS=/home/daniel/K3D_llama_cpp/logs`

Ingestion (GPU)
- COCO (OpenCLIP, max 50k)
  - `scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_coco --images-dir "$RAW/coco_raw/train2017/train2017" --captions "$RAW/coco_raw/annotations/annotations/captions_train2017.json" --out-csv "$BASE/coco.train.clip.csv" --out-meta "$BASE/coco.train.meta.json" --max 50000 > "$LOGS/coco_ingest.log" 2>&1 & echo $! > "$LOGS/coco_ingest.pid"`
- Clotho (LAION‑CLAP)
  - `scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_audio --audio "$RAW/clotho_raw/clotho_audio_development/development/*.wav" "$RAW/clotho_raw/clotho_audio_validation/validation/*.wav" --out-csv "$BASE/clotho.clap.csv" --out-meta "$BASE/clotho.meta.json" > "$LOGS/clotho_ingest.log" 2>&1 & echo $! > "$LOGS/clotho_ingest.pid"`
- VATEX (OpenCLIP, max 2k)
  - `scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_video --videos "$RAW/vatex_raw/media/*.mp4" "$RAW/vatex_raw/media/*.mkv" "$RAW/vatex_raw/media/*.webm" --out-csv "$BASE/vatex.clip.csv" --out-meta "$BASE/vatex.meta.json" --thumbs-dir "$BASE/vatex/thumbs" --base-url "" --fps 0.5 --max 2000 > "$LOGS/vatex_ingest.log" 2>&1 & echo $! > "$LOGS/vatex_ingest.pid"`

Build GLBs
- Clotho: `scripts/k3d_env.sh run python -m k3dgen "$BASE/clotho.clap.csv" --gltf viewer/public/clotho.glb --k 8 --reducer umap --metadata "$BASE/clotho.meta.json" --emb-precision f16`
- VATEX: `scripts/k3d_env.sh run python -m k3dgen "$BASE/vatex.clip.csv" --gltf viewer/public/vatex_2k.glb --k 10 --reducer umap --metadata "$BASE/vatex.meta.json" --emb-precision f16`
- COCO: `scripts/k3d_env.sh run python -m k3dgen "$BASE/coco.train.clip.csv" --gltf viewer/public/coco_50k.glb --k 10 --reducer umap --metadata "$BASE/coco.train.meta.json" --emb-precision f16`

Build Unified Galaxy (one space)
- Merge what’s available (skips missing gracefully):
```bash
scripts/k3d_env.sh run python -m knowledge3d.tools.build_galaxy \
  --out viewer/public/galaxy.glb --dims 256 --k 10 --reducer pca \
  image:$BASE/coco.train.clip.csv:$BASE/coco.train.meta.json \
  audio:$BASE/clotho.clap.csv:$BASE/clotho.meta.json \
  video:$BASE/vatex.clip.csv:$BASE/vatex.meta.json
```

Cross‑Modal Matching
- `scripts/k3d_env.sh run python -m knowledge3d.tools.match_crossmodal --audio ../Knowledge3D.local/datasets/clotho.meta.json --video ../Knowledge3D.local/datasets/vatex.meta.json --out ../Knowledge3D.local/datasets/matched --top 30000`
- Produces: `../Knowledge3D.local/datasets/matched/matches.jsonl` and `pool.txt`

Monitoring & Recovery
- Quick status: `scripts/k3d_env.sh run python -m knowledge3d.tools.tlab_bridge status`
- Tail logs: `tail -f "$LOGS/coco_ingest.log" "$LOGS/clotho_ingest.log" "$LOGS/vatex_ingest.log"`
- PIDs: `cat "$LOGS/coco_ingest.pid"` etc. Kill/restart if needed:
  - `kill $(cat "$LOGS/coco_ingest.pid") 2>/dev/null || true`
  - rerun the launch command

Viewer & Live
- Viewer: `cd viewer && npm run dev`
- Live WS bridge (custom port if 8765 is busy): `scripts/k3d_env.sh run python -m knowledge3d.bridge.live_server --port 8787`
- Viewer connects automatically (tries 8765 → 8787). To force: `http://localhost:5173/?ws=ws://localhost:8787`
- Press Enter to chat (HUD overlay). Dev controls hidden; add `?dev=1` for the old selector panel.
- Load GLBs: place `viewer/public/coco_50k.glb` / `clotho.glb` / `vatex_2k.glb`. The viewer auto‑loads the first available.

Known Pitfalls
- Wrong COCO path: images belong under `.../train2017/train2017`. If you see “no images processed”, fix this path.
- Python 2 trap: don’t nest `bash -lc` inside `conda run` (can drop to system Python 2.7). Always use `scripts/k3d_env.sh run python -m ...`.
- CLAP per‑file failures: the tool falls back to hashed vectors so the pipeline can continue; re‑run later with a more complete audio stack if needed.
- UMAP acceleration: if you have RAPIDS, set `export K3D_CONDA_ENV=k3d-rapids` before building to prefer GPU reducers/ANN (optional).

References
- `docs/ENV_POLICY.md`
- `knowledge3d/tools/ingest_{coco,audio,video}.py`
- `knowledge3d/tools/match_crossmodal.py`
- `docs/DATASETS_50K.md`
