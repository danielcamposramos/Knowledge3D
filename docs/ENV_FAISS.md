FAISS‑GPU Dedicated Environment
===============================

Goal
----
Avoid package conflicts between RAPIDS/PyTorch/CUDA stacks by isolating FAISS‑GPU in its own conda env and selectively using it for tools that require FAISS neighbors.

Bootstrap
---------

1) Create the env and install FAISS‑GPU (tries conda‑forge first, then pip cu12; installs CUDA 12.4 libs if needed):

```
scripts/bootstrap_faiss_env.sh
```

2) Validate:

```
export PATH="$HOME/miniconda3/bin:$PATH"
conda run -n k3dfaiss python -c "import faiss;print('faiss',getattr(faiss,'__version__','?'));print('gpus',getattr(faiss,'get_num_gpus',lambda:-1)())"
```

Usage
-----

- Use this env only for FAISS‑dependent steps, via the existing runner:

```
K3D_CONDA_ENV=k3dfaiss scripts/k3d_env.sh run python -c "import faiss;print(faiss.get_num_gpus())"
```

- Standard GPU work (UMAP via RAPIDS, Transformers, etc.) remains in `k3dml`.
- If you prefer to run a specific tool entirely in the FAISS env (e.g., ingestion that requires FAISS), set `K3D_CONDA_ENV=k3dfaiss` for that command only.

Notes
-----

- libcublasLt symbol mismatches: The script installs CUDA 12.4 libraries from NVIDIA if FAISS fails to detect GPUs. This typically resolves the `cublasLtGetEnvironmentMode` symbol errors.
- Strict GPU policy: Our KNN fallback remains GPU‑only (cuML NearestNeighbors) if FAISS still cannot attach to GPUs.

