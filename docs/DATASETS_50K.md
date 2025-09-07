# Multimodal 50k Datasets — Bootstrapping

Goal
- Build small (≤50k) K3D GLBs per modality for rapid iteration and end‑to‑end tests.

Pipeline (tools)
- Text → GLB: `k3dgen` on a text file (one line per entry).
- Image+Text (WIT) → GLB: `knowledge3d/tools/ingest_wit.py` then `k3dgen`.
- Audio → GLB: `knowledge3d/tools/ingest_audio.py` (LAION‑CLAP) then `k3dgen`.
- Video → GLB: `knowledge3d/tools/ingest_video.py` (OpenCLIP) then `k3dgen`.
- Orchestrator: `knowledge3d/tools/build_multimodal_50k.py` automates per‑modality builds.
 - HF Fetch: `knowledge3d/tools/hf_fetch_multimodal.py` downloads audio/video + captions from HF datasets.
 - Cross‑modal match: `knowledge3d/tools/match_crossmodal.py` ranks audio↔video pairs by caption similarity.

Examples
```bash
# Text (trim to ≤50k lines)
python -m knowledge3d.tools.build_multimodal_50k \
  --text data/ai_books_basic.txt \
  --text-out viewer/public/text_50k.glb

# WIT (sample TSV)
python -m knowledge3d.tools.build_multimodal_50k \
  --wit-tsv /k3dlocal/wit/wit_v1.train.sample.tsv.gz \
  --wit-out viewer/public/wit_50k.glb

# Audio (requires LAION-CLAP and local audio files)
python -m knowledge3d.tools.build_multimodal_50k \
  --audio "/k3dlocal/audio/*.wav" \
  --audio-out viewer/public/audio_50k.glb

# Video (requires OpenCLIP and local videos)
python -m knowledge3d.tools.build_multimodal_50k \
  --video "/k3dlocal/video/*.mp4" \
  --video-out viewer/public/video_50k.glb

# Fetch from HF (audio/video) with captions
python -m knowledge3d.tools.hf_fetch_multimodal \
  --dataset confit/audiocaps --split train --kind audio --limit 30000 \
  --out-dir ../Knowledge3D.local/datasets/audiocaps
python -m knowledge3d.tools.hf_fetch_multimodal \
  --dataset CLAPv2/clotho_full --split train --kind audio --limit 20000 \
  --out-dir ../Knowledge3D.local/datasets/clotho
python -m knowledge3d.tools.hf_fetch_multimodal \
  --dataset friedrichor/MSR-VTT --split train --kind video --limit 20000 \
  --out-dir ../Knowledge3D.local/datasets/msrvtt
python -m knowledge3d.tools.hf_fetch_multimodal \
  --dataset HuggingFaceM4/vatex --split train --kind video --limit 15000 \
  --out-dir ../Knowledge3D.local/datasets/vatex
python -m knowledge3d.tools.hf_fetch_multimodal \
  --dataset gigant/webvid-mini --split train --kind video --limit 15000 \
  --out-dir ../Knowledge3D.local/datasets/webvid_mini

# Rank cross‑modal pairs (audio↔video) to maximize connectivity
python -m knowledge3d.tools.match_crossmodal \
  --audio ../Knowledge3D.local/datasets/audiocaps/meta.jsonl \
  --video ../Knowledge3D.local/datasets/msrvtt/meta.jsonl \
  --out   ../Knowledge3D.local/datasets/matched --top 30000
```

Notes
- For large public datasets, start with official samples (e.g., WIT sample TSV) to keep bandwidth and storage reasonable.
- Set `K3D_ACCEL=gpu` when available to speed up OpenCLIP/CLAP.
- GLBs can be placed under `viewer/public/houses/<K3D_HOUSE_ID>/` for per‑avatar access.
