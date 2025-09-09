#!/usr/bin/env bash
# Wikipedia → Embeddings (sharded, GPU) → Galaxy build helper
#
# Usage examples:
#   scripts/wiki_galaxy_pipeline.sh verify
#   scripts/wiki_galaxy_pipeline.sh embed-next 500000   # next 500k lines
#   scripts/wiki_galaxy_pipeline.sh build 500000        # build using head 500k rows
#   scripts/wiki_galaxy_pipeline.sh full-cycle 500000   # embed-next then build
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
DATA_DIR=${K3D_DATA_DIR:-"$ROOT_DIR/../Knowledge3D.local/datasets"}
TEXT="$DATA_DIR/wikipedia.en.txt"
CSV="$DATA_DIR/wikipedia.en.embed.csv"
META="$DATA_DIR/wikipedia.en.embed.meta.json"
SAMPLE_N=${2:-500000}
SAMPLE_CSV="$DATA_DIR/wikipedia.en.embed.head${SAMPLE_N}.csv"
OUT_GLTF="$ROOT_DIR/viewer/public/galaxy.cross.glb"

run_py() { "$ROOT_DIR/scripts/k3d_env.sh" run "$@"; }

verify() {
  echo "[verify] DATA_DIR=$DATA_DIR";
  [[ -f "$TEXT" ]] && echo "[ok] text: $(du -h "$TEXT" | awk '{print $1}')" || echo "[warn] missing: $TEXT";
  if [[ -f "$CSV" ]]; then
    local lines; lines=$(wc -l "$CSV" | awk '{print $1}')
    echo "[ok] embed csv: $(du -h "$CSV" | awk '{print $1}')  rows=$((lines-1))";
  else
    echo "[warn] missing: $CSV";
  fi
  [[ -f "$META" ]] && echo "[ok] meta: $(du -h "$META" | awk '{print $1}')" || echo "[warn] missing: $META";
}

embed_next() {
  if [[ ! -f "$TEXT" ]]; then echo "[err] missing $TEXT"; exit 1; fi
  local start=0
  if [[ -f "$CSV" ]]; then
    local lines; lines=$(wc -l "$CSV" | awk '{print $1}')
    start=$((lines>0?lines-1:0))
  fi
  echo "[embed] start=$start limit=$SAMPLE_N → $CSV";
  run_py python -m knowledge3d.tools.embed_text_sharded \
    --in "$TEXT" --out-csv "$CSV" --out-meta "$META" \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --batch 8192 --start "$start" --limit "$SAMPLE_N"
}

build() {
  if [[ ! -f "$CSV" ]]; then echo "[err] missing $CSV"; exit 1; fi
  echo "[build] sampling head $SAMPLE_N rows → $SAMPLE_CSV";
  head -n $((SAMPLE_N+1)) "$CSV" > "$SAMPLE_CSV"
  echo "[build] assembling galaxy to $OUT_GLTF";
  SPECS=("text:$SAMPLE_CSV")
  # Optional modalities if present
  [[ -f "$DATA_DIR/coco.train.clip.csv" ]] && SPECS+=("image:$DATA_DIR/coco.train.clip.csv:$DATA_DIR/coco.train.meta.json")
  [[ -f "$DATA_DIR/clotho.clap.csv" ]] && SPECS+=("audio:$DATA_DIR/clotho.clap.csv:$DATA_DIR/clotho.meta.json")
  [[ -f "$DATA_DIR/vatex.clip.csv" ]] && SPECS+=("video:$DATA_DIR/vatex.clip.csv:$DATA_DIR/vatex.meta.json")
  run_py python -m knowledge3d.tools.build_galaxy \
    --out "$OUT_GLTF" --dims 256 --k 10 --reducer pca "${SPECS[@]}"
  echo "[ok] built $OUT_GLTF"
}

case "${1:-verify}" in
  verify) verify ;;
  embed-next) verify; embed_next ;;
  build) verify; build ;;
  full-cycle) verify; embed_next; build ;;
  *) echo "Usage: $0 verify|embed-next [N]|build [N]|full-cycle [N]" >&2; exit 2 ;;
esac
