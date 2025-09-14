FROM debian:13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Optionally add CUDA toolkit in derived images if needed
# RUN apt-get update && apt-get install -y cuda-toolkit-12-1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY knowledge3d ./knowledge3d
COPY k3dgen ./k3dgen
COPY viewer ./viewer

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install . && \
    python3 -m pip install websockets pygltflib numpy

EXPOSE 8765-8800

ENTRYPOINT ["python3", "-m", "knowledge3d.bridge.live_server", "--auto-port"]

