#!/usr/bin/env bash
#
# Run K3D live server in Docker with GPU support
#
# This bypasses the GCC 15 + NVRTC incompatibility on Debian 13
# by running in Ubuntu 22.04 + GCC 11 environment.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build runtime image if it doesn't exist
if ! docker images | grep -q k3d-runtime; then
    echo "Building k3d-runtime Docker image..."
    docker build -f Dockerfile.runtime -t k3d-runtime .
fi

# Run live server with GPU support
echo "Starting K3D live server in Docker..."
echo "Server will be available at ws://127.0.0.1:8765"
echo ""

# Set local directory
K3D_LOCAL_DIR="${K3D_LOCAL_DIR:-/K3D/Knowledge3D.local}"
K3D_HOUSE_ID="${K3D_HOUSE_ID:-default}"

echo "Using local directory: $K3D_LOCAL_DIR"
echo "Using House ID: $K3D_HOUSE_ID"
echo ""

docker run --rm -d \
    --gpus all \
    -v "$SCRIPT_DIR:/workspace" \
    -v "$K3D_LOCAL_DIR:/local" \
    -e K3D_LOCAL_DIR=/local \
    -e K3D_HOUSE_ID="$K3D_HOUSE_ID" \
    -e CUPY_CACHE_DIR=/local/cache/.cupy_cache \
    -p 8765:8765 \
    --name k3d-live-server \
    k3d-runtime

# Usage notes:
# - Press Ctrl+C to stop the server
# - Test with: python test_navigate.py (from host)
# - View logs with: docker logs -f k3d-live-server
