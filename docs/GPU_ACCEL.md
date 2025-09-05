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

Option B — Host venv with FAISS CPU only
- Create venv: `python -m venv ~/k3d-venv && . ~/k3d-venv/bin/activate`
- Install: `pip install -e . pandas numpy scikit-learn pygltflib umap-learn sentence-transformers faiss-cpu`
- Run with `K3D_ACCEL=cpu` for deterministic CPU builds.

Notes
- For larger-than-100k datasets, FAISS IndexFlatL2 works but IVF indexes may be faster. We will add IVF support on request.
- `--emb-precision f16` halves embedding buffer size; most viewers decode to float32 on read.

