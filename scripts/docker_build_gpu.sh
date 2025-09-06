#!/usr/bin/env bash
set -euo pipefail
IMG_NAME=${IMG_NAME:-k3d-gpu:latest}
docker build -f docker/Dockerfile.k3d-gpu -t "$IMG_NAME" .
echo "Built $IMG_NAME"

