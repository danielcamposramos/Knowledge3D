# Containerized Execution Policy (Debian)

Always run K3D tooling inside a managed environment. Do not use the system Python directly on Debian.

Approved environments
- Conda (recommended): Python 3.10 with ML deps
- venv (fallback): Python 3.10+ virtualenv
- Docker (optional): for fully pinned images

Quick start (Conda)
- Create env once:
  conda create -y -n k3dml python=3.10
  conda run -n k3dml python -m pip install --upgrade pip
  conda run -n k3dml python -m pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  conda run -n k3dml python -m pip install open_clip_torch pillow av soundfile \
    laion_clap umap-learn scikit-learn numpy pandas pygltflib

- Run any K3D command:
  conda run -n k3dml env PYTHONPATH=. python -m knowledge3d.tools.ingest_coco --help

Fallback (venv)
- python3 -m venv .venv_k3dml
- . .venv_k3dml/bin/activate
- python -m pip install --upgrade pip
- python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
- python -m pip install open_clip_torch pillow av soundfile laion_clap \
  umap-learn scikit-learn numpy pandas pygltflib

Rationale
- Debian’s system Python and packages can block ML wheels (e.g., PyTorch, open_clip, av, laion_clap) on newer versions. Using conda/venv ensures compatible wheels and repeatable runs.

Notes
- Set PYTHONPATH=. when running from the repo root so local modules resolve.
- For large downloads/ingests, prefer storing raw media under /home/daniel/K3D_llama_cpp/datasets and copying curated subsets into ../Knowledge3D.local/datasets for builds.

GPU setup (NVIDIA)
- Create GPU env (CUDA 12.1+):
  scripts/k3d_env.sh bootstrap-gpu
- Run commands inside the env:
  scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_video --help
- Validate GPU availability:
  nvidia-smi
  scripts/k3d_env.sh run python -c "import torch; print(torch.cuda.is_available())"

Environment selection and pitfalls
- Select the conda env explicitly to avoid surprises from shell auto‑activation:
  - `export K3D_CONDA_ENV=k3dml` (default GPU env)
  - Optional RAPIDS: `export K3D_CONDA_ENV=k3d-rapids`
- Always use the wrapper to run Python: `scripts/k3d_env.sh run python -m ...`
  - Do not nest `bash -lc` inside `conda run` — it may drop to system Python 2.7 and cause syntax errors.
  - The wrapper sets `PYTHONPATH=.` so local modules resolve from repo root.
