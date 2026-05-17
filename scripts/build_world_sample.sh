#!/usr/bin/env bash
# Build a tiny "World of Everything" single‑galaxy sample with aligned meanings across modalities.
#
# Goal
# - Assemble small, meaning‑aligned subsets from text, images, audio, and video
# - Build per‑modality GLBs, then unify into one Galaxy (meaning‑first)
# - Add explicit cross‑modal edges for smooth traversal
#
# Inputs (expected if present; skips gracefully otherwise)
# - $BASE/coco.train.clip.csv + coco.train.meta.json   (images)
# - $BASE/clotho.clap.csv + clotho.meta.json           (audio)
# - $BASE/vatex.clip.csv + vatex.meta.json             (video)
# - Local text (from repo docs)                        (text)
#
# Usage
#   KEYWORDS="rain,street,car,child,city,speech" \
#   BASE=/K3D/Knowledge3D.local/datasets \
#   scripts/build_world_sample.sh
#
# Viewer
#   Default loads /galaxy.glb. To visualize cross‑modal edges: ?gltf=/galaxy.cross.glb
set -euo pipefail

BASE=${BASE:-/K3D/Knowledge3D.local/datasets}
OUT_DIR=${OUT_DIR:-viewer/public/_world}
KEYWORDS=${KEYWORDS:-rain,street,car,city,child,speech}
TEXT_MAX=${TEXT_MAX:-1200}
SAMPLE_MAX=${SAMPLE_MAX:-800}
K=${K:-8}
DIMS=${DIMS:-128}
REDUCER=${REDUCER:-pca}

echo "[WORLD] base=$BASE out=$OUT_DIR keywords=$KEYWORDS dims=$DIMS k=$K"
mkdir -p "$OUT_DIR"

# 1) Build a small text set from local docs, filtered by keywords
TEXT_SRC="$OUT_DIR/world_text.txt"
echo "[WORLD] collecting text → $TEXT_SRC"
python - "$TEXT_SRC" "$TEXT_MAX" "$KEYWORDS" <<'PY'
import sys, re
from pathlib import Path
out, max_lines, kws = Path(sys.argv[1]), int(sys.argv[2]), [k.strip().lower() for k in sys.argv[3].split(',') if k.strip()]
paths = [
    Path('README.md'),
    Path('docs/VISION.md'),
    Path('docs/CRANIUM_SKILLS.md'),
    Path('docs/RUNBOOK_MULTIMODAL_50K.md'),
]
lines=[]
for p in paths:
    if not p.exists():
        continue
    for ln in p.read_text(encoding='utf-8', errors='ignore').splitlines():
        s=ln.strip()
        if not s or s.startswith('![') or s.startswith('```') or s.startswith('<'):
            continue
        low=s.lower()
        if any(k in low for k in kws):
            lines.append(s)
out.write_text('\n'.join(lines[:max_lines])+'\n',encoding='utf-8')
print('lines:',len(lines[:max_lines]))
PY

echo "[WORLD] text→vectors"
python -m knowledge3d.tools.text_to_vectors --text "$TEXT_SRC" --out "$OUT_DIR/world_text.vectors.csv" --dims "$DIMS"
echo "[WORLD] text GLB"
python -m k3dgen "$OUT_DIR/world_text.vectors.csv" --gltf "$OUT_DIR/text.glb" --k "$K" --reducer "$REDUCER" --emb-precision f16 || true

# 2) Filter each modality by KEYWORDS and build small GLBs
build_modality() {
  local name="$1"; shift
  local csv_in="$1"; shift
  local meta_in="$1"; shift
  local prefix="$OUT_DIR/${name}.sample"
  if [[ -f "$csv_in" && -f "$meta_in" ]]; then
    echo "[WORLD] filtering $name by [$KEYWORDS]"
    python -m knowledge3d.tools.filter_modal_csv \
      --csv "$csv_in" --meta "$meta_in" \
      --out-csv "$prefix.csv" --out-meta "$prefix.meta.json" \
      --keywords "$KEYWORDS" --max "$SAMPLE_MAX"
    echo "[WORLD] GLB $name"
    python -m k3dgen "$prefix.csv" --gltf "$OUT_DIR/${name}.glb" --k "$K" --reducer "$REDUCER" --metadata "$prefix.meta.json" --emb-precision f16 || true
  else
    echo "[SKIP] $name source missing: $csv_in or $meta_in"
  fi
}

build_modality image "$BASE/coco.train.clip.csv" "$BASE/coco.train.meta.json"
build_modality audio "$BASE/clotho.clap.csv" "$BASE/clotho.meta.json"
build_modality video "$BASE/vatex.clip.csv" "$BASE/vatex.meta.json"

# 3) Unify into one Galaxy
inputs=()
for pair in text "$OUT_DIR/text.glb" image "$OUT_DIR/image.glb" audio "$OUT_DIR/audio.glb" video "$OUT_DIR/video.glb"; do inputs+=("$pair"); done
declare -a glb_args=()
for ((i=0;i<${#inputs[@]};i+=2)); do
  kind="${inputs[i]}"; path="${inputs[i+1]}"
  if [[ -f "$path" ]]; then glb_args+=("$path:$kind"); fi
done
if [[ ${#glb_args[@]} -eq 0 ]]; then
  echo "[ERR] No modality GLBs found to unify." >&2; exit 1
fi
echo "[WORLD] unify → viewer/public/galaxy.glb"
python -m knowledge3d.tools.unify_glbs "${glb_args[@]}" --out viewer/public/galaxy.glb --dims "$DIMS" --k "$K" --reducer "$REDUCER"

# 4) Add cross‑modal edges
echo "[WORLD] cross‑modal edges → viewer/public/galaxy.cross.glb"
python -m knowledge3d.tools.add_crossmodal_edges --input viewer/public/galaxy.glb --out viewer/public/galaxy.cross.glb

echo "[WORLD] done. Open viewer and use ?gltf=/galaxy.cross.glb for edge view."

