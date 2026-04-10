## Containerized Execution Policy (Debian)

Always run K3D tooling inside a managed environment. Do not use the system Python directly on Debian.

## Approved environments

- Conda (recommended): Python 3.10 with ML deps
- venv (fallback): Python 3.10+ virtualenv
- Docker (optional): for fully pinned images

## Quick start (Conda)

- **GPU / PTX work:** `conda env create -f envs/k3d-cranium.yml`
- **GPU PTX test rig:** `conda env create -f envs/k3d-trm.yml`
- **CPU-only mock testing:** `conda env create -f envs/k3d-testing.yml`
- **RAPIDS pipeline:** `conda env create -f envs/k3d-rapids.yml`
- **ARC Python 3.11 orchestration lane:** `conda env create -f envs/trmc_core.yml`
- `k3d-testing` ships the CPU-only dependencies needed by the Step 13‑B harness
  (`pytest-benchmark`, `memory_profiler`, `matplotlib`, `psutil`). Use it to materialise
  reports without touching GPU-only stacks.
- All K3D conda envs live on the SSD (`/K3D/Knowledge3D.local/envs`) for faster startup.
  `conda activate k3d-cranium` resolves there via `~/.condarc`.

Attach/refresh a tmux session (keeps kernels alive):

```bash
tmux new -As k3d
```

Activate the env inside tmux:

```bash
conda activate k3d-cranium   # or k3d-trm / k3d-testing / k3d-rapids as needed
```

Run PTX/CuPy tests via the SSD env:

```bash
source /home/daniel/miniforge/etc/profile.d/conda.sh
conda run -p /K3D/Knowledge3D.local/envs/k3d-trm env PYTHONPATH=$(pwd) pytest …
```

For legacy manual bootstraps (rare):

```bash
conda create -y -n k3dml python=3.10
conda activate k3dml
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install open_clip_torch pillow av soundfile laion_clap umap-learn scikit-learn numpy pandas pygltflib
```

Run any K3D command (env stays active in tmux):

```bash
env PYTHONPATH=. python -m knowledge3d.tools.ingest_coco --help
```

Canonical sovereign artifact rebuild:

```bash
env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/rebuild_sovereign_artifact.py \
  --refresh-feed-source --refresh-build-feed --force-rebuild --verbose
```

See [MAINTENANCE.md](MAINTENANCE.md) for when rebuilds are mandatory and how retention works.

## Fallback (venv)

```bash
python3 -m venv .venv_k3dml
. .venv_k3dml/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install open_clip_torch pillow av soundfile laion_clap umap-learn scikit-learn numpy pandas pygltflib
```

## Rationale

Debian's system Python and packages can block ML wheels (e.g., PyTorch, open_clip, av, laion_clap)
on newer versions. Using conda/venv ensures compatible wheels and repeatable runs.

## Notes

- Set `PYTHONPATH=.` when running from the repo root so local modules resolve.
- For large downloads/ingests, prefer storing raw media under `/K3D/K3D_llama_cpp/datasets/`
  (canonical local dataset root). Curated subsets may be symlinked into
  `../Knowledge3D.local/datasets/` for builds.

## GPU setup (NVIDIA)

- **For PTX/FSM development**: See [DOCKER_ENV.md](DOCKER_ENV.md) for complete CUDA 12.4 + CuPy setup.
- On the Debian 14 workstation the KDE session runs on the iGPU; export `CUDA_VISIBLE_DEVICES=0`
  before launching tmux so the RTX 3070 is exposed inside the conda shell.
- **PTX kernel development requires CuPy** (already bundled in `k3d-cranium`; run manually
  only if rebuilding): `pip install cupy-cuda12x`

Bootstrap CUDA-ready env once:

```bash
scripts/k3d_env.sh bootstrap-gpu
```

Validate GPU availability (env active):

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Environment selection and pitfalls

- Work inside tmux so the activated env persists across long GPU jobs.
- Select the env explicitly after attaching: `conda activate k3d-cranium`
  - CPU mock/testing harness: `conda activate k3d-testing`
  - RAPIDS pipeline: `conda activate k3d-rapids`
  - ARC orchestration / SDK-side probing: `conda activate trmc_core`
- `bash scripts/k3d_env.sh run -e <env> ...` resolves named envs under
  `/K3D/Knowledge3D.local/envs` and activates them automatically.
- Avoid mixing system Python with the project; all tooling assumes the conda env is active.
  Keep hot paths on `k3d-cranium`; reserve `k3d-testing` for CPU-bound pytest and benchmarking.
- For ARC work, keep the sovereign GPU path on `k3d-cranium`. Use `trmc_core` only for
  Python 3.11 orchestration (SDK-side adapters). If the official ARC runtime is unavailable,
  code must report that truthfully — never silently fall back to a non-sovereign path.

## Secrets and API keys

**Never store secrets inside the repository working tree.** The repo root is under
`/mnt/arquivos/.../Knowledge3D/` — any file written there can end up in git.

**Canonical secrets path:** `/K3D/Knowledge3D.local/secrets/`

This directory lives on the local SSD alongside `envs/` and `datasets/`, physically outside
the git repository. It is never committed.

**File naming convention:** `{service}_api_key.txt` — lowercase, underscores, suffix `_api_key.txt`.

Examples:

- `/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt`
- `/K3D/Knowledge3D.local/secrets/openai_api_key.txt`
- `/K3D/Knowledge3D.local/secrets/anthropic_api_key.txt`

**Runtime resolution pattern** (two-tier fallback used throughout the codebase):

```python
import os
from pathlib import Path

def _resolve_api_key(service: str, env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if key:
        return key
    path = Path(f"/K3D/Knowledge3D.local/secrets/{service}_api_key.txt")
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""
```

Env var takes precedence (useful for CI/tmux sessions); file read is the local default.

**To add a new secret** — write the key value (one line, no trailing whitespace):

```bash
echo "your_key_here" > /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt
chmod 600 /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt
```

**Confirm a key is present without printing it:**

```bash
test -s /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt && echo OK || echo MISSING
```

## Live server ports and WebSockets

- Port 8787 is commonly used by ComfyUI; the live benchmark script avoids it by default.
  Override via `K3D_LIVE_PORTS` (e.g., `K3D_LIVE_PORTS="8791 8793 8797"`).
- Pin `websockets==10.4` in all environments. Newer versions (e.g., 15.x) caused
  opening-handshake timeouts on Debian 13; pinning resolves this.
- `knowledge3d.tools.register_galaxy` uses `open_timeout=30` to tolerate slower startup.
- Fast start mode: set `K3D_LIVE_FAST=1` (default) or pass `--fast-start` to delay heavy
  imports until after the WS server is listening.
