# Containerized Execution Policy (Debian)

Always run K3D tooling inside a managed environment. Do not use the system Python directly on Debian.

Approved environments
- Conda (recommended): Python 3.10 with ML deps
- venv (fallback): Python 3.10+ virtualenv
- Docker (optional): for fully pinned images

Quick start (Conda)
- Create env once:
  conda create -y -n k3dml python=3.10
- Attach/refresh a tmux session (keeps kernels alive):
  tmux new -As k3d
- Activate the env inside tmux and seed packages:
  conda activate k3dml
  python -m pip install --upgrade pip
  python -m pip install     torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  python -m pip install open_clip_torch pillow av soundfile     laion_clap umap-learn scikit-learn numpy pandas pygltflib

- Run any K3D command (env stays active in tmux):
  env PYTHONPATH=. python -m knowledge3d.tools.ingest_coco --help

Fallback (venv)
- python3 -m venv .venv_k3dml
- . .venv_k3dml/bin/activate
- python -m pip install --upgrade pip
- python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
- python -m pip install open_clip_torch pillow av soundfile laion_clap   umap-learn scikit-learn numpy pandas pygltflib

Rationale
- Debian’s system Python and packages can block ML wheels (e.g., PyTorch, open_clip, av, laion_clap) on newer versions. Using conda/venv ensures compatible wheels and repeatable runs.

Notes
- Set PYTHONPATH=. when running from the repo root so local modules resolve.
- For large downloads/ingests, prefer storing raw media under /home/daniel/K3D_llama_cpp/datasets and copying curated subsets into ../Knowledge3D.local/datasets for builds.

GPU setup (NVIDIA)
- **For PTX/FSM development**: See [DOCKER_ENV.md](DOCKER_ENV.md) for complete CUDA 12.4 + CuPy setup
- Bootstrap CUDA-ready env once:
  scripts/k3d_env.sh bootstrap-gpu
- Inside tmux after `conda activate k3dml`, run GPU tooling:
  env PYTHONPATH=. python -m knowledge3d.tools.ingest_video --help
- Validate GPU availability (env active):
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available())"
- **PTX kernel development requires CuPy**:
  pip install cupy-cuda12x  # For CUDA 12.x
  # or
  pip install cupy-cuda11x  # For CUDA 11.x

Environment selection and pitfalls
- Work inside tmux so the activated env persists across long GPU jobs.
- Select the env explicitly after attaching: `conda activate k3dml`
  - Optional RAPIDS env: `conda activate k3d-rapids`
- If you need the helper script, `scripts/k3d_env.sh run ...` now shells into tmux and activates the env for you.
- Avoid mixing system Python with the project; all tooling assumes the conda env is active.

Live server ports and WebSockets
- Port 8787 is commonly used by ComfyUI; the live benchmark script now avoids this port by default.
  - Override ports via `K3D_LIVE_PORTS` (e.g., `K3D_LIVE_PORTS="8791 8793 8797"`).
- For stability, we pin `websockets==10.4` in all environments.
  - On Debian 13, newer `websockets` versions (e.g., 15.x) caused opening-handshake timeouts in our live server; pinning resolves this.
  - `knowledge3d.tools.register_galaxy` uses a longer `open_timeout=30` to tolerate slower startup.
  - Fast start mode: set `K3D_LIVE_FAST=1` (default) or pass `--fast-start` to delay heavy imports/model loads until after the WS server is listening. This improves readiness and avoids client handshake timeouts.
