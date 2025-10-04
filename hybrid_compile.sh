#!/bin/bash
# Hybrid compilation: Use Docker to pre-compile, then run in Debian conda
set -e

echo "🔧 K3D Hybrid Compilation Strategy"
echo "=================================="
echo ""
echo "Step 1: Build Docker compilation environment..."

docker build -f Dockerfile.compile -t k3d-compile . 2>&1 | tail -20

echo ""
echo "Step 2: Pre-compile CuPy kernels in Docker (GCC 11, compatible)..."

docker run --gpus all \
    -v $(pwd):/workspace \
    k3d-compile

echo ""
echo "Step 3: Verify compiled kernels..."

if [ -d ".cupy_cache" ]; then
    cubin_count=$(find .cupy_cache -name "*.cubin" | wc -l)
    ptx_count=$(find .cupy_cache -name "*.ptx" | wc -l)
    cache_size=$(du -sh .cupy_cache | cut -f1)

    echo "✅ Pre-compilation complete!"
    echo "   - CUBIN files: $cubin_count"
    echo "   - PTX files: $ptx_count"
    echo "   - Cache size: $cache_size"
    echo ""
    echo "Step 4: Testing in Debian 13 conda environment..."

    # Activate conda and test with pre-compiled cache
    source /home/daniel/miniforge/bin/activate k3d-cranium
    export PYTHONPATH=.
    export CUPY_CACHE_DIR=$(pwd)/.cupy_cache

    echo "   Setting CUPY_CACHE_DIR=$CUPY_CACHE_DIR"
    echo ""
    echo "   Running navigation test..."

    python test_navigate.py

    echo ""
    echo "✅ Hybrid compilation SUCCESS!"
    echo "   Navigation works with pre-compiled Docker kernels in Debian 13!"
else
    echo "❌ Cache directory not found. Pre-compilation failed."
    exit 1
fi
