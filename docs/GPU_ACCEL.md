GPU Acceleration (FAISS + RAPIDS UMAP)

Overview
- k-NN: FAISS GPU (fallback to CPU)
- Dimensionality Reduction: RAPIDS cuML UMAP (fallback to umap-learn or PCA)
- Control via env: `K3D_ACCEL=auto|gpu|cpu`

Host prerequisites
- NVIDIA driver with CUDA 11.8+ runtime support (driver >= 520 is fine; 550 shows CUDA 12.4 OK)
- Docker with `--gpus all` working (`nvidia-smi` inside containers)

Option A — RAPIDS Docker (recommended)
1) Pull a RAPIDS image (CUDA 11.8 example):
   `docker pull rapidsai/rapidsai-core:24.06-cuda11.8-runtime-ubuntu22.04-py3.10`
2) Run a container with the repo mounted:
   `docker run --rm --gpus all -it -v "$PWD":"/workspace" -w /workspace rapidsai/rapidsai-core:24.06-cuda11.8-runtime-ubuntu22.04-py3.10 bash`
3) Inside the container, install this repo and FAISS GPU via conda:
   `conda install -y -c pytorch -c nvidia faiss-gpu=1.7.2 cudatoolkit=11.8`
   `pip install -U pip wheel setuptools && pip install -e . pandas numpy scikit-learn pygltflib umap-learn sentence-transformers`
4) Generate a GPU-accelerated GLB:
   `export K3D_ACCEL=gpu`
   `python -m k3dgen data/ai_compendium_80k.txt --gltf ../Knowledge3D.local/datasets/ai_compendium.80k.umap.gpu.glb --k 10 --reducer umap --emb-precision f16`

Option B — Conda (recommended on bare metal)
1) GPU setup (RAPIDS + FAISS GPU):
   - `conda env create -f envs/k3d-rapids.yml && conda activate k3d-rapids`
   - `export K3D_ACCEL=gpu K3D_FAISS_DEVICE=gpu`
2) CPU-only setup:
   - `conda env create -f envs/k3d-testing.yml && conda activate k3d-testing`
   - `export K3D_ACCEL=cpu`

Notes
- On Debian-like hosts, heavy training is guarded and requires Conda or Docker unless `K3D_ALLOW_NATIVE=1` is set.

Notes
- For larger-than-100k datasets, FAISS IndexFlatL2 works but IVF indexes may be faster. We will add IVF support on request.
- `--emb-precision f16` halves embedding buffer size; most viewers decode to float32 on read.

## Debian 13 Quickstart (Container‑first)

Debian 13 (“trixie”) is supported via containers to ensure consistent CUDA stacks.

1) Ensure NVIDIA Container Toolkit is installed and `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` works.
2) From the repo root, launch RAPIDS:
```
docker run --rm --gpus all -it \
  -v "$PWD":"/workspace" -w /workspace \
  rapidsai/rapidsai-core:24.06-cuda11.8-runtime-ubuntu22.04-py3.10 bash
```
3) Inside the container:
```
conda install -y -c pytorch -c nvidia faiss-gpu=1.7.2 cudatoolkit=11.8
pip install -U pip wheel setuptools
pip install -e . pandas numpy scikit-learn pygltflib umap-learn sentence-transformers
export K3D_ACCEL=gpu K3D_FAISS_DEVICE=gpu
python -m k3dgen examples/sample_vectors.csv --gltf ../Knowledge3D.local/datasets/sample.umap.gpu.glb --k 5 --reducer umap --emb-precision f16
```
4) Outside the container, run the viewer locally (Node 16+) and point it at the generated GLB.
Strict GPU mode (no CPU fallback)
- The project is configured to require GPU for heavy steps:
  - UMAP reduction must use RAPIDS cuML on GPU; CPU fallbacks are removed.
  - FAISS k‑NN must use GPU; CPU/sklearn fallbacks are removed.
- If you need CPU support later, uncomment the noted sections in `knowledge3d/accel.py` where the CPU fallbacks were removed.
- Non‑UMAP PCA remains CPU-based as it is lightweight and used for far‑LOD positioning.
- The generator respects `K3D_STRICT_GPU=1` to prevent fallback to CPU at the CLI layer as well.
