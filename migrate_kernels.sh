#!/bin/bash
# migrate_kernels.sh — Move non-canonical kernels to Old_Attempts

set -euo pipefail

KERNELS_DIR="knowledge3d/cranium/kernels"
OLD_DIR="$KERNELS_DIR/Old_Attempts"

mkdir -p "$OLD_DIR"

# Files to keep in kernels/ (canonical list)
CANONICAL=(
    "codec_ops.cu"
    "drawing_transform_ops.cu"
    "color_convert.cu"
    "filter_convolution.cu"
    "gradient_rasterizer.cu"
    "vectordotmap_encoder.cu"
    "trm_ops.cu"
    "ternary_ops.cu"
)

is_canonical() {
    local fname="$1"
    for canonical in "${CANONICAL[@]}"; do
        if [[ "$fname" == "$canonical" ]]; then
            return 0
        fi
    done
    return 1
}

shopt -s nullglob
for f in "$KERNELS_DIR"/*.cu; do
    basename="$(basename "$f")"
    if ! is_canonical "$basename"; then
        echo "Moving $basename to Old_Attempts/"
        mv "$f" "$OLD_DIR/"
    fi
done

echo "Kernel migration complete."
