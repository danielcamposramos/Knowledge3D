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
