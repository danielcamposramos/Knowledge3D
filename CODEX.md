# Codex Tasks

This file lists tasks for Codex agents working on this repository. They mirror the phases described in [docs/ROADMAP.md](docs/ROADMAP.md).

## Immediate tasks

1. **Repo housekeeping**
   - Ensure `LICENSE`, `NOTICE`, and `LICENSE-CC-BY-4.0.txt` are present.
   - Update `README.md` with a minimal badge table.
2. **Skeleton layout**
   - Create `/docs` for the white paper (`k3d_whitepaper_v1.md`).
   - Create `/spec` with `k3d_schema.json` and a stub `glTF_K3D_extension.md`.
   - Create `/examples` containing `sample_vectors.csv` and `demo.k3d`.
3. **Hello-World pipeline**
   - Implement a tiny CLI `k3dgen` in Python.
   - Read `sample_vectors.csv`, run 3D PCA, and output a glTF scene plus `.k3d` sidecar.
4. **Lightweight browser viewer**
   - Under `/viewer`, scaffold a Vite + Three.js demo.
   - Load the glTF, read `.k3d` metadata, color nodes, and show a hover tooltip.
5. **Project management setup**
   - Create a GitHub Projects board with columns Backlog / In-progress / Done.
   - File issues tracking phases 0–3.
6. **Continuous Integration**
   - Add `.github/workflows/ci.yml` running black, flake8, and jest.
   - Deploy `/viewer` build to GitHub Pages.
7. **CONTRIBUTING guide**
   - Draft `CONTRIBUTING.md` covering coding style, branch naming, DCO sign-off, and clip limits.
8. **Update roadmap**
   - Keep `docs/ROADMAP.md` current with milestones through 2026.

Agents should reference this file to coordinate efforts and keep contributions aligned with the overall roadmap.

