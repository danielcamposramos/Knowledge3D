# Knowledge3D Roadmap

This document outlines the planned phases for getting the Knowledge3D repository into a community‑ready state. Each phase introduces new deliverables and sets expectations for future contributions.

| Phase | Deliverable | Purpose | Owner/Tooling Hints |
|------|-------------|---------|--------------------|
| **0. Repo housekeeping (1 day)** | `LICENSE` (Apache‑2.0), `LICENSE-CC-BY-4.0.txt`, `NOTICE`, minimal `README.md` badge table | Legal clarity & polished first impression | Manual commit |
| **1. Skeleton layout (1 day)** | `/docs` → white‑paper (`k3d_whitepaper_v1.md`), `/spec` → `k3d_schema.json`, `glTF_K3D_extension.md` stub, `/examples` → `sample_vectors.csv`, `demo.k3d` | Gives contributors clear folders |  |
| **2. "Hello‑World" pipeline (3‑5 days)** | Python CLI `k3dgen` that reads `sample_vectors.csv`, runs PCA → 3D and writes minimal glTF with `.k3d` metadata sidecar | Proves the spec is implementable | `pygltflib`, `scikit-learn` |
| **3. Lightweight browser viewer (5‑7 days)** | `/viewer` with Vite + Three.js demo that loads glTF, reads `.k3d` JSON, colors nodes and shows basic hover tooltip | Visual "wow" for README GIF | Codex to stub JS |
| **4. Milestone issues & project board (0.5 day)** | GitHub Projects board with columns (Backlog / In‑progress / Done) and issues for phases 0‑3 | Open‑source workflow clarity | GitHub UI |
| **5. Continuous integration (CI) (0.5 day)** | `.github/workflows/ci.yml` running black + flake8, jest for viewer, and building/uploading demo to GitHub Pages | Prevents broken demo | `actions/setup-python`, `actions/upload-pages-artifact` |
| **6. Draft CONTRIBUTING.md (1 day)** | Coding style, branch naming, DCO sign-off, clip limit reminders | Smooth external PRs | Manual draft |
| **7. Outreach placeholder (ongoing)** | `docs/ROADMAP.md` with 2025‑Q3 "Spec v0.1", Q4 "LOD & AR extension", 2026 "Khronos proposal" | Sets vision for newcomers |  |

## Status

Phases 0 and 1 are complete. With this update the repository enters **Phase 2**
and includes the first working prototype of `k3dgen`. The repository also
aggregates AI-generated documents from ChatGPT, Gemini, Claude, Manus AI, and
Perplexity AI under `Diverse_AI_Reports.md` to reference the many formats that
will be joined going forward.

**Why this order?**

- Phases 0–1 make the repo public-ready before code.
- Phase 2 delivers a tangible artifact proving the concept (<100 LOC).
- Phase 3 provides a browser demo for the README GIF.
- Phases 4–6 turn the repo from a personal playground into a community project.
- Phase 7 seeds long-term ambition without blocking immediate progress.

**Tactical hints**

- Keep `k3dgen` tiny: `pandas` → `scikit-learn.PCA(3)` → `pygltflib` mesh of instanced spheres. Embed per‑point color in extras.
- `.k3d` sidecar: simple JSON `{ id, vector, metadata }[]` to be read by the viewer—no glTF patching yet.
- Viewer: start with point sprites and `OrbitControls`; add ray‑caster hover later.
- GIF for README: record the viewer rotating the demo cloud—30 fps, 6 s loop.
- CI: any commit touching `/viewer` should auto‑deploy to `gh-pages`, so folks can click‑try without cloning.

