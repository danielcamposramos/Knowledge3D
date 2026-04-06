# Containerized Execution Policy (Debian)

Always run K3D tooling inside a managed environment. Do not use the system Python directly on Debian.

Approved environments
- Conda (recommended): Python 3.10 with ML deps
- venv (fallback): Python 3.10+ virtualenv
- Docker (optional): for fully pinned images

Quick start (Conda)
- **GPU / PTX work:** `conda env create -f envs/k3d-cranium.yml`
- **GPU PTX test rig:** `conda env create -f envs/k3d-trm.yml`
- **CPU-only mock testing:** `conda env create -f envs/k3d-testing.yml`
- **RAPIDS pipeline:** `conda env create -f envs/k3d-rapids.yml`
- `k3d-testing` ships the CPU-only dependencies needed by the Step 13‑B harness (`pytest-benchmark`, `memory_profiler`, `matplotlib`, `psutil`). Use it to materialise reports without touching GPU-only stacks.
- Attach/refresh a tmux session (keeps kernels alive):
  tmux new -As k3d
- Activate the env inside tmux:
  conda activate k3d-cranium   # or k3d-trm / k3d-testing / k3d-rapids as needed
- Run PTX/CuPy tests via the SSD env:
  source /home/daniel/miniforge/etc/profile.d/conda.sh
  conda run -p /K3D/Knowledge3D.local/envs/k3d-trm env PYTHONPATH=$(pwd) pytest …
- All K3D conda envs now live on the SSD (`/K3D/Knowledge3D.local/envs`) for faster startup. `conda activate k3d-cranium` resolves there via `~/.condarc`.
- For legacy manual bootstraps (rare):
  conda create -y -n k3dml python=3.10
  conda activate k3dml
  python -m pip install --upgrade pip
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  python -m pip install open_clip_torch pillow av soundfile laion_clap umap-learn scikit-learn numpy pandas pygltflib

- Run any K3D command (env stays active in tmux):
  env PYTHONPATH=. python -m knowledge3d.tools.ingest_coco --help
- Canonical sovereign artifact rebuild:
  `env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/rebuild_sovereign_artifact.py --refresh-feed-source --refresh-build-feed --force-rebuild --verbose`
  See [MAINTENANCE.md](MAINTENANCE.md) for when rebuilds are mandatory and how retention works.

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
- Inside tmux after `conda activate k3d-cranium`, run GPU tooling:
  env PYTHONPATH=. python -m knowledge3d.tools.ingest_video --help
- On the Debian 14 workstation the KDE session runs on the iGPU; export `CUDA_VISIBLE_DEVICES=0`
  before launching tmux so the RTX 3070 is exposed inside the conda shell.
- Validate GPU availability (env active):
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available())"
- **PTX kernel development requires CuPy** (already bundled in `k3d-cranium`; run manually only if rebuilding):
  pip install cupy-cuda12x  # For CUDA 12.x
  # or
  pip install cupy-cuda11x  # For CUDA 11.x

Environment selection and pitfalls
- Work inside tmux so the activated env persists across long GPU jobs.
- Select the env explicitly after attaching: `conda activate k3d-cranium`
  - CPU mock/testing harness: `conda activate k3d-testing`
  - RAPIDS pipeline: `conda activate k3d-rapids`
- If you need the helper script, `scripts/k3d_env.sh run ...` now shells into tmux and activates the env for you (e.g., `scripts/k3d_env.sh run -e k3d-testing pytest tests/...`).
- Avoid mixing system Python with the project; all tooling assumes the conda env is active. Keep hot paths on `k3d-cranium`; reserve `k3d-testing` for CPU-bound pytest and benchmarking.

Live server ports and WebSockets
- Port 8787 is commonly used by ComfyUI; the live benchmark script now avoids this port by default.
  - Override ports via `K3D_LIVE_PORTS` (e.g., `K3D_LIVE_PORTS="8791 8793 8797"`).
- For stability, we pin `websockets==10.4` in all environments.
  - On Debian 13, newer `websockets` versions (e.g., 15.x) caused opening-handshake timeouts in our live server; pinning resolves this.
  - `knowledge3d.tools.register_galaxy` uses a longer `open_timeout=30` to tolerate slower startup.
  - Fast start mode: set `K3D_LIVE_FAST=1` (default) or pass `--fast-start` to delay heavy imports/model loads until after the WS server is listening. This improves readiness and avoids client handshake timeouts.
