#!/usr/bin/env bash
set -euo pipefail

# Orchestrate 1M dataset build -> vectors -> GLB -> doors.
# Runs inside the GPU container (k3d-gpu) with the repo mounted at /workspace

ROOT=/workspace
LOCAL=/k3dlocal
TXT1M="$LOCAL/datasets/ai_compendium_1m.txt"
SHARD_A="$LOCAL/datasets/ai_compendium_1m.partA.txt"
SHARD_B="$LOCAL/datasets/ai_compendium_1m.partB.txt"
CSV_A="$LOCAL/datasets/ai_compendium_1m.partA.512d.csv"
CSV_B="$LOCAL/datasets/ai_compendium_1m.partB.512d.csv"
CSV_M="$LOCAL/datasets/ai_compendium_1m.512d.csv"
GLB_RAW="$LOCAL/datasets/ai_compendium.1m.umap.ivfpq.glb"
GLB_DOORS="$LOCAL/datasets/ai_compendium.1m.umap.ivfpq.doors.glb"

export K3D_ACCEL=${K3D_ACCEL:-gpu}
export K3D_FAISS_DEVICE=${K3D_FAISS_DEVICE:-gpu}
# Optional verbose accel logs (FAISS/UMAP/PCA path hints)
export K3D_ACCEL_LOG=${K3D_ACCEL_LOG:-1}

echo "[1/6] Build 1M text corpus -> $TXT1M"
python -m knowledge3d.tools.build_mega_corpus --target 1000000 --out "$TXT1M"

echo "[2/6] Split into two 500k shards"
head -n 500000 "$TXT1M" > "$SHARD_A"
tail -n +500001 "$TXT1M" > "$SHARD_B"

echo "[3/6] Vectorize shards (512 dims)"
python -m knowledge3d.tools.text_to_vectors --text "$SHARD_A" --out "$CSV_A" --dims 512
python -m knowledge3d.tools.text_to_vectors --text "$SHARD_B" --out "$CSV_B" --dims 512

echo "[4/6] Merge CSVs -> $CSV_M"
python -m knowledge3d.tools.merge_vectors --inputs "$CSV_A" "$CSV_B" --out "$CSV_M"
# Cleanup shard CSVs and raw text shards to save space
rm -f "$CSV_A" "$CSV_B" "$SHARD_A" "$SHARD_B"
# Optionally drop the 1M text (reproducible via build_mega_corpus)
rm -f "$TXT1M"

echo "[5/6] Generate GLB with PCA + FAISS-IVFPQ -> $GLB_RAW"
# Strict GPU policy: omit UMAP LOD variants unless RAPIDS cuML is present (set up in container).
# To restore UMAP LODs later, add:  --lod-levels umap_fast,umap_high
python -m k3dgen "$CSV_M" --gltf "$GLB_RAW" --k 10 --reducer pca --ann ivfpq --emb-precision f16

echo "[6/6] Skip door marking (House-as-rooms model)"
mv -f "$GLB_RAW" "$GLB_DOORS" || cp -f "$GLB_RAW" "$GLB_DOORS"

# Final cleanup: remove merged CSV to reclaim space
rm -f "$CSV_M"

echo "Done. Output GLB: $GLB_DOORS"
