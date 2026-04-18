#!/usr/bin/env bash
# Multimodal 50k orchestrator (COCO, Clotho, VATEX) with logs and optional auto-build.
#
# Usage:
#   RAW=/custom/raw BASE=/K3D/Knowledge3D.local/datasets LOGS=/custom/logs \
#     scripts/run_ingest_build.sh [--autobuild-coco]
#
# Notes:
# - Respects K3D_CONDA_ENV (e.g., k3dml or k3d-rapids)
# - Uses scripts/k3d_env.sh run to ensure correct Python + PYTHONPATH
set -euo pipefail

RAW=${RAW:-/home/daniel/K3D_llama_cpp/datasets}
BASE=${BASE:-/K3D/Knowledge3D.local/datasets}
LOGS=${LOGS:-/home/daniel/K3D_llama_cpp/logs}
AUTOBUILD=0
if [[ "${1:-}" == "--autobuild-coco" ]]; then AUTOBUILD=1; fi

mkdir -p "$BASE" "$LOGS" "$BASE/vatex/thumbs"

echo "[RUN] Env: K3D_CONDA_ENV=${K3D_CONDA_ENV:-k3dml} RAW=$RAW BASE=$BASE LOGS=$LOGS"

# Launch COCO ingest (OpenCLIP)
COCO_IMG="$RAW/coco_raw/train2017/train2017"
COCO_CAP="$RAW/coco_raw/annotations/annotations/captions_train2017.json"
COCO_CSV="$BASE/coco.train.clip.csv"
COCO_META="$BASE/coco.train.meta.json"
if [[ -d "$COCO_IMG" && -f "$COCO_CAP" ]]; then
  nohup scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_coco \
    --images-dir "$COCO_IMG" \
    --captions "$COCO_CAP" \
    --out-csv "$COCO_CSV" \
    --out-meta "$COCO_META" \
    --max 50000 > "$LOGS/coco_ingest.log" 2>&1 & echo $! > "$LOGS/coco_ingest.pid"
  echo "[RUN] COCO ingest PID $(cat "$LOGS/coco_ingest.pid")"
else
  echo "[SKIP] COCO paths missing: $COCO_IMG or $COCO_CAP"
fi

# Launch Clotho ingest (CLAP)
CLO_CSV="$BASE/clotho.clap.csv"
CLO_META="$BASE/clotho.meta.json"
nohup scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_audio \
  --audio "$RAW/clotho_raw/clotho_audio_development/development/*.wav" \
          "$RAW/clotho_raw/clotho_audio_validation/validation/*.wav" \
  --out-csv "$CLO_CSV" \
  --out-meta "$CLO_META" > "$LOGS/clotho_ingest.log" 2>&1 & echo $! > "$LOGS/clotho_ingest.pid"
echo "[RUN] Clotho ingest PID $(cat "$LOGS/clotho_ingest.pid")"

# Launch VATEX ingest (OpenCLIP)
VATEX_CSV="$BASE/vatex.clip.csv"
VATEX_META="$BASE/vatex.meta.json"
nohup scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_video \
  --videos "$RAW/vatex_raw/media/*.mp4" "$RAW/vatex_raw/media/*.mkv" "$RAW/vatex_raw/media/*.webm" \
  --out-csv "$VATEX_CSV" \
  --out-meta "$VATEX_META" \
  --thumbs-dir "$BASE/vatex/thumbs" \
  --base-url "" \
  --fps 0.5 \
  --max 2000 > "$LOGS/vatex_ingest.log" 2>&1 & echo $! > "$LOGS/vatex_ingest.pid"
echo "[RUN] VATEX ingest PID $(cat "$LOGS/vatex_ingest.pid")"

# Build GLBs for audio/video immediately if CSVs exist (fast)
if [[ -f "$CLO_CSV" && -f "$CLO_META" ]]; then
  scripts/k3d_env.sh run python -m k3dgen "$CLO_CSV" --gltf viewer/public/clotho.glb --k 8 --reducer umap --metadata "$CLO_META" --emb-precision f16 || true
fi
if [[ -f "$VATEX_CSV" && -f "$VATEX_META" ]]; then
  scripts/k3d_env.sh run python -m k3dgen "$VATEX_CSV" --gltf viewer/public/vatex_2k.glb --k 10 --reducer umap --metadata "$VATEX_META" --emb-precision f16 || true
fi

# Optionally wait for COCO CSV to appear and auto‑build the GLB
if [[ "$AUTOBUILD" == "1" ]]; then
  echo "[WAIT] for COCO CSV: $COCO_CSV"
  for i in {1..7200}; do # up to ~2h, adjust as needed
    if [[ -s "$COCO_CSV" && -f "$COCO_META" ]]; then
      echo "[BUILD] COCO GLB"
      scripts/k3d_env.sh run python -m k3dgen "$COCO_CSV" --gltf viewer/public/coco_50k.glb --k 10 --reducer umap --metadata "$COCO_META" --emb-precision f16 || true
      break
    fi
    sleep 1
  done
fi

echo "[DONE] Launched ingests. Logs under $LOGS."

