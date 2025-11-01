#!/usr/bin/bash
#
# Compile backward pass CUDA kernels to PTX
#

set -e

PTX_DIR="knowledge3d/cranium/ptx"
ARCH="sm_75"  # RTX 3060

echo "==================================================================="
echo "Compiling CNN Backward Pass Kernels"
echo "==================================================================="

KERNELS=(
    "maxpool_2x2_backward"
    "batchnorm_backward"
    "conv2d_3x3_backward"
    "classification_loss"
    "sgd_optimizer"
)

for kernel in "${KERNELS[@]}"; do
    echo ""
    echo "Compiling $kernel.cu..."
    nvcc --ptx \
        -arch=$ARCH \
        -O3 \
        --use_fast_math \
        -I. \
        "$PTX_DIR/$kernel.cu" \
        -o "$PTX_DIR/$kernel.ptx"

    if [ $? -eq 0 ]; then
        echo "✓ $kernel.ptx created ($(stat -c%s "$PTX_DIR/$kernel.ptx") bytes)"
    else
        echo "✗ Failed to compile $kernel.cu"
        exit 1
    fi
done

echo ""
echo "==================================================================="
echo "All kernels compiled successfully!"
echo "==================================================================="
echo ""
ls -lh "$PTX_DIR"/*.ptx | grep -E "backward|loss|sgd"
