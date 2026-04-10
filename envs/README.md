# Environment Usage Notes

Conda location (local):
- `/home/daniel/miniforge/bin/conda`
- `/home/daniel/miniforge/condabin/conda`

Usage examples:
- One-shot command:
  ```
  /home/daniel/miniforge/bin/conda run -n k3d-cranium python -c "import pyphen; print(pyphen.__version__)"
  ```
- Activate for a shell (without editing system PATH permanently):
  ```
  export PATH="/home/daniel/miniforge/bin:/home/daniel/miniforge/condabin:$PATH"
  conda activate k3d-cranium
  ```

Envs provided:
- `k3d-cranium.yml`
- `k3d-rapids.yml`
- `k3d-testing.yml`
- `k3d-trm.yml`
- `trmc_core.yml`

Preferred launcher:
- `bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q tests/test_arc_r0_surface.py`
- `bash scripts/k3d_env.sh run -e trmc_core python benchmarks/arc3_sdk_agent.py --game-id ls20`

Notes:
- All named envs resolve first against `/K3D/Knowledge3D.local/envs/<name>` so the SSD-backed prefixes are used automatically.
- `trmc_core` is the Python 3.11 ARC orchestration lane. Keep sovereign GPU/PTX execution in SSD-managed envs only; never run these paths on system Python.
