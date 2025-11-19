# Contributing to Knowledge3D

Thank you for helping build Knowledge3D. This project is a sovereign, GPU‑native research system where AI and humans co‑develop a shared spatial operating system. Contributions are welcome, but they must respect the project’s architecture, philosophy, and resource constraints.

Before proposing changes, please read:

- `CLAUDE.md` — primary onboarding guide for AI + human collaborators
- `CLAUDE_LOCAL.md` — local environment layout and constraints
- `K3D_Technical_White_Paper.md` — architecture and design overview
- `docs/ROADMAP.md` — current phase plan and priorities
- `docs/HOUSE_GALAXY_TABLET.md` — memory architecture and tablet contract
- `docs/ENV_POLICY.md` — environment and GPU usage policy

All contributions should align with the active roadmap phase and uphold the House/Galaxy/Cranium memory model.

## How We Work

- **Sovereign GPU stack**: Hot paths run on GPU via pre‑compiled PTX kernels loaded with `ctypes` and `libcuda.so`. No CuPy, PyTorch, or other GPU frameworks at runtime in the sovereign path.
- **Memory architecture first**: Treat the Galaxy (RAM) and House (GLB on disk) as the primary weight store. Features that touch memory must honor the Memory Tablet workflow.
- **Procedural‑first**: Prefer storing how‑to‑reconstruct (procedures, glyphs, generators) instead of raw pixels or large binary blobs. Large assets live in `Knowledge3D.local/` with regeneration recipes in `Large_Assets_Kitchen/`.
- **Docs and tests**: Each change that affects behavior should update the relevant spec in `docs/` and, when applicable, add or adjust tests under `tests/` or `knowledge3d/**/tests/`.
- **Budget reality**: This is a self‑funded favela lab. Keep dependencies minimal, avoid heavy new tooling, and favor efficient solutions.

## Types of Contributions

- **Bug fixes** — Fix crashes, incorrect behavior, or inconsistencies between code and specs.
- **Documentation** — Improve clarity, fix inaccuracies, or add missing explanations in `docs/` and README‑level guides.
- **Performance improvements** — Optimize kernels, bridges, or memory flows without breaking sovereignty guarantees.
- **New capabilities** — Add carefully scoped features aligned with `docs/ROADMAP.md` and existing kernel map, preferably by extending existing `.cu`/PTX modules.
- **Testing and validation** — Strengthen unit and integration tests, or add benchmark coverage for critical kernels.

If you are unsure whether an idea fits the roadmap, please open an issue first (see Issue Templates) and link to the relevant spec sections.

## Development Environment

Follow `docs/ENV_POLICY.md` for the canonical environments. In short:

- Use the `k3d-cranium` conda environment for daily development:
  - `conda env create -f envs/k3d-cranium.yml` (once)
  - `conda activate k3d-cranium`
- Set `PYTHONPATH=.` and usually:
  - `export K3D_PTX_STRICT=1`
  - `export K3D_FORCE_PTX_FUSE=1`
- For data‑prep and analytics, use the RAPIDS environment:
  - `conda env create -f envs/k3d-rapids.yml`

When running GPU jobs, follow the pattern in `docs/ENV_POLICY.md` (e.g., `tmux`, `CUDA_VISIBLE_DEVICES=0`, stable CUDA context).

## Coding Guidelines

- **Respect sovereignty**:
  - No new runtime dependencies that bypass the sovereign PTX stack for core logic.
  - Keep orchestration in Python; keep math and reasoning in CUDA `.cu` and PTX.
  - Extend `.cu` sources in `knowledge3d/cranium/kernels/` and rebuild PTX with `nvcc` rather than editing `.ptx` directly.
- **Reuse existing kernels**:
  - Consult `SOVEREIGN_SWARM_BRIEFING.md` and `docs` for the kernel reuse map before introducing new kernels.
  - Prefer augmenting existing bridges in `knowledge3d/cranium/bridges/` and loaders in `knowledge3d/cranium/sovereign/`.
- **Style and structure**:
  - Follow the existing structure and naming conventions in the area you’re editing.
  - Avoid one‑letter variable names outside of short loops; keep functions small and focused.
  - Do not add large formatting or style tools unless they are already part of the workflow.

## Tests and Validation

Before opening a pull request:

- Run Python tests (where applicable):
  - `pytest -q`
- For viewer‑related changes, follow the testing instructions in `docs/` (e.g., Vite dev server, Jest tests).
- For kernel changes:
  - Rebuild PTX with `nvcc` using the commands in `docs/` or the relevant script.
  - Run or extend the tests under `knowledge3d/cranium/tests/` and any phase‑specific test scripts in `scripts/` or `TEMP/`.

If tests are flaky or environment‑sensitive, note this clearly in your pull request.

### About CI and PTX‑backed Tests

Most of the critical test suite exercises sovereign PTX kernels and requires a real NVIDIA GPU with `libcuda.so` available. GitHub‑hosted runners do not provide this environment, so CI‑based test runs are disabled by design and will not pass there. Contributors are expected to run relevant tests locally in a properly configured `k3d-cranium` environment and report results in their pull requests.

## Opening Issues

When filing an issue, please:

- Use the appropriate issue template (bug report or feature request).
- For bugs:
  - Include steps to reproduce.
  - Provide environment details (OS, GPU, CUDA version, Python version, active conda env).
  - Attach relevant logs or tracebacks (redacting sensitive information).
- For feature requests:
  - Reference the relevant sections of `docs/ROADMAP.md` and other specs.
  - Explain how the proposal fits the House/Galaxy/Tablet architecture and GPU sovereignty rules.

Issues that lack sufficient context may be closed if they cannot be reproduced or aligned with the roadmap.

## Opening Pull Requests

When submitting a PR:

- Keep the scope focused; smaller, incremental changes are easier to review.
- Describe:
  - What you changed and why.
  - Which parts of the architecture it touches (Cranium, Galaxy, House, Tablet, viewer, tools, etc.).
  - Any new or changed environment requirements.
- Indicate:
  - Which tests you ran and their results.
  - Any follow‑up work that should be tracked in issues or `TEMP/` notes.

If your change affects the memory model, PTX kernels, or Tablet UX, clearly describe the impact on:

- Galaxy (RAM) semantics and performance
- House (GLB) representations and consolidation
- Tablet interactions and constraints

## Code of Conduct

By participating in this project, you agree to abide by the Code of Conduct (`CODE_OF_CONDUCT.md`). Be respectful, collaborative, and mindful of the constraints and realities under which this project operates.
