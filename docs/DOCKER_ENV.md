# Docker Environment Setup for Knowledge3D

**Date**: 2025-10-06
**Purpose**: Document the complete Docker/GPU environment for unified FSM development

## Environment Overview

Knowledge3D development uses a **hybrid approach**:
- **Host GPU**: Direct NVIDIA driver access (RTX 3060, CUDA 12.4)
- **Conda Environment**: `/home/daniel/miniforge` (Python 3.12.11)
- **Docker Services**: Supporting infrastructure (databases, UIs, etc.)

## GPU Environment

### Hardware
- **GPU**: NVIDIA GeForce RTX 3060
- **VRAM**: 12288 MiB
- **Driver**: 550.163.01
- **CUDA**: 12.4.131
- **Compute Capability**: 8.6 (sm_86)

### CUDA Libraries
```bash
# NVRTC (Runtime Compilation)
/usr/lib/x86_64-linux-gnu/libnvrtc.so.12.4.127

# PTX Assembler
/usr/bin/ptxas  # For kernel verification
```

## Python Environment (Conda)

### Primary Environment
```bash
# Location
/home/daniel/miniforge

# Python Version
Python 3.12.11 | packaged by conda-forge

# Activate (inside tmux)
tmux new -As k3d
conda activate k3dml  # Or base for GPU work
```

### GPU-Critical Packages
```bash
# CuPy (CUDA 12.x) - MANDATORY for PTX kernels
pip install cupy-cuda12x==13.6.0

# NumPy (compatible version)
numpy==2.3.3

# PyTorch (optional, not used in PTX pipeline)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Installation Script
```bash
#!/bin/bash
# Setup GPU environment for K3D

# Install CuPy for CUDA 12.x
pip install cupy-cuda12x

# Install testing dependencies
pip install pytest

# Install development tools
pip install pygltflib pillow av soundfile

# Verify GPU access
python3 -c "
import cupy as cp
print(f'CuPy: {cp.__version__}')
print(f'CUDA: {cp.cuda.runtime.getDeviceCount()} device(s)')
print(f'Device 0: {cp.cuda.Device(0).compute_capability}')
"
```

## Docker Services

### Running Containers
```bash
$ docker ps
CONTAINER ID   IMAGE                                 PORTS
f67bb2bed330   ghcr.io/browserless/chromium:latest   0.0.0.0:3100->3000/tcp
03b21b52f743   qdrant/qdrant:latest                  0.0.0.0:6333-6334->6333-6334/tcp
e3c2b7677acb   docker.n8n.io/n8nio/n8n:latest        0.0.0.0:5678->5678/tcp
a9ebb5c14ff2   ghcr.io/open-webui/open-webui:cuda    0.0.0.0:3000->8080/tcp
2ef9bb5f92aa   yanwk/comfyui-boot:cu126-slim         0.0.0.0:8188->8188/tcp
c411bd12e4d6   containrrr/watchtower                 8080/tcp
```

### K3D GPU Container (Optional)

**Dockerfile**: `docker/Dockerfile.k3d-gpu`
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PATH=/opt/conda/bin:$PATH

# Install micromamba
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Install GPU FAISS, cuML, PyTorch
RUN micromamba install -y -c conda-forge -c pytorch -c nvidia \
    python=3.10 faiss-gpu=1.7.2 cudatoolkit=11.8 \
    && micromamba clean -a -y

# For CUDA 12.4 (recommended)
RUN pip install cupy-cuda12x

CMD ["bash"]
```

**Build & Run**:
```bash
# Build GPU container
docker build -f docker/Dockerfile.k3d-gpu -t k3d-gpu:latest .

# Run with GPU access
docker run --gpus all -it \
    -v /K3D/Knowledge3D:/workspace \
    k3d-gpu:latest
```

### Docker Compose (Future)
```yaml
# docker-compose.gpu.yml
version: '3.8'
services:
  k3d-cranium:
    build:
      context: .
      dockerfile: docker/Dockerfile.k3d-gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - K3D_PTX_STRICT=1
      - K3D_FORCE_PTX_FUSE=1
    volumes:
      - ./:/workspace
    ports:
      - "8765-8800:8765-8800"
```

## Tmux Workflow

### Session Management
```bash
# Create/attach to K3D session
tmux new -As k3d

# Split for multi-terminal work
Ctrl+B %  # Vertical split
Ctrl+B "  # Horizontal split

# Navigate panes
Ctrl+B arrow-keys
```

### Development Layout
```
┌─────────────────────┬─────────────────────┐
│ Editor/Tests        │ PTX Compilation     │
│ pytest tests/ -v    │ ptxas --gpu-name... │
│                     │                     │
├─────────────────────┼─────────────────────┤
│ Python REPL         │ Git/Commit          │
│ ipython             │ git status          │
└─────────────────────┴─────────────────────┘
```

## Environment Variables

### K3D Configuration
```bash
# In ~/.bashrc or tmux session
export K3D_PTX_STRICT=1          # No CPU fallbacks
export K3D_FORCE_PTX_FUSE=1      # Always use PTX fusion
export K3D_DISABLE_TEXT_MODALITY=0
export K3D_FUSE_DIMS="512:256:128:128"  # text:image:audio:video

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export CUPY_CACHE_DIR=/tmp/cupy_cache
```

### Testing Configuration
```bash
# Pytest with GPU
export PYTEST_TIMEOUT=60
export PYTHONPATH=/K3D/Knowledge3D

# Run GPU tests
pytest tests/test_unified_fsm.py tests/test_warp_modality_fuse.py -v
```

## Verification Checklist

### GPU Access
```bash
# 1. NVIDIA driver
nvidia-smi

# 2. CUDA runtime
nvcc --version

# 3. CuPy
python3 -c "import cupy as cp; print(cp.cuda.is_available())"

# 4. PTX compilation
ptxas --version
```

### PTX Kernel Testing
```bash
# Compile kernel
ptxas --gpu-name sm_86 knowledge3d/cranium/ptx/fused_head_fsm_full.ptx -o /tmp/test.cubin

# Run tests
pytest tests/test_unified_fsm.py -v

# Expected output:
# tests/test_unified_fsm.py::test_fsm_kernels_load PASSED
# tests/test_unified_fsm.py::test_unified_attention_kernel PASSED
# tests/test_unified_fsm.py::test_rpn_dispatch_kernel PASSED
```

### Performance Verification
```bash
# Run with profiling
python3 -c "
from knowledge3d.cranium.unified_fsm import UnifiedFSMContext
import numpy as np
import time

fsm = UnifiedFSMContext()
buf = fsm.create_unified_buffer(n_nodes=100)
query = np.random.randn(512).astype(np.float32)

start = time.perf_counter()
scores = fsm.launch_unified_attention(buf, query)
elapsed = (time.perf_counter() - start) * 1000

print(f'Attention: {elapsed:.3f}ms for 100 nodes')
# Expected: <1ms
"
```

## Troubleshooting

### CuPy Import Error
```bash
# Error: libnvrtc.so.11.2 not found
# Solution: Install correct CuPy version
pip uninstall cupy-cuda11x
pip install cupy-cuda12x  # Match CUDA 12.4
```

### PTX Compilation Errors
```bash
# Error: Invalid PTX syntax
# Solution: Use ptxas to get detailed errors
ptxas --gpu-name sm_86 file.ptx -o /tmp/test.cubin

# Common fixes:
# - Special registers: mov to intermediate first
# - Address types: use mul.wide for u64
# - Predicates: @!p bra SKIP; instead of @p { }
```

### Test Timeouts
```bash
# Error: Test hangs in FSM dispatch
# Solution: FSM loop bug - check terminal condition
# Workaround: Test individual kernels first
pytest tests/test_unified_fsm.py::test_unified_attention_kernel -v
```

## Migration to Full Docker

### Future: Pure Docker Workflow
```bash
# Build GPU-enabled container
docker build -f Dockerfile.gpu -t k3d:gpu .

# Run all tests in container
docker run --gpus all -v $(pwd):/workspace k3d:gpu \
    pytest tests/ -v

# Interactive development
docker run --gpus all -it -v $(pwd):/workspace k3d:gpu bash
```

### Benefits
- ✅ Reproducible environment
- ✅ Isolated dependencies
- ✅ Easy CI/CD integration
- ✅ Multi-GPU support

## Summary

**Current Setup** (Hybrid):
- Host: NVIDIA RTX 3060 + CUDA 12.4
- Conda: Python 3.12.11 + CuPy 13.6.0
- Docker: Supporting services only

**Recommended Setup** (Production):
- Full GPU Docker container
- Conda env for local dev
- Tmux for session persistence

**Key Requirements**:
- CUDA 12.x or 11.x
- CuPy matching CUDA version
- PTX assembler (ptxas)
- 12GB+ VRAM for full pipeline

---

**Setup Time**: ~10 minutes
**First Test**: `pytest tests/test_unified_fsm.py::test_fsm_kernels_load -v`
**Full Pipeline**: `pytest tests/ -v` (requires GPU)
