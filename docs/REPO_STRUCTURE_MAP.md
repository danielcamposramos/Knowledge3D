# Knowledge3D Repository Structure Map (2025‑10‑06)

This map reflects the post-cleanup repository, where all heavy assets now live in `Knowledge3D.local/`. Pair this with `Large_Assets_Kitchen/README.md` for reproduction recipes and the manifests in `Old_Attempts/Legacy_Fancy_RAG/` for historical references.

## Top-Level Layout

| Path | Purpose | Notes |
| --- | --- | --- |
| `.github/` | CI workflows and templates | GPU-first CI settings |
| `.vscode/` | Editor recommendations | Optional |
| `Large_Assets_Kitchen/` | Recipes for rebuilding large artifacts in `.local` | Text only |
| `Old_Attempts/` | Legacy fancy-RAG manifests | No binaries stored here |
| `codeopt/` | Dual-code CLI runtime | Active |
| `data/` | Lightweight seed datasets, ontologies, templates | Heavy exports removed |
| `docs/` | Vision, roadmap, specs | Authoritative references |
| `ext/` | External datasets/tools (vendored) | Treat as read-only |
| `k3dgen/` | Generator CLI modules | Active |
| `knowledge3d/` | Core Cognitive OS runtime (PTX-first) | Active |
| `scripts/` | Shell helpers for pipelines | Active |
| `spec/` | Formal specifications | Active |
| `tests/` | Pytest suite | Active |
| `viewer/` | Viewer sources (TS + Three.js) | `viewer/public/` now empty stub |
| `build/`, `logs/` | Placeholder directories (empty, `.gitkeep`) | Runtime outputs redirected to `.local` |

## Legacy Artifacts

All fancy-RAG assets (examples, viewer exports, historical reports, binaries) were moved out of Git. Consult:
- `Old_Attempts/Legacy_Fancy_RAG/manifests/*.txt` for inventories
- `Large_Assets_Kitchen/README.md` for regeneration guidance
- Runtime location: `../Knowledge3D.local/old_attempts/legacy_fancy_rag/`

## Runtime Workspace

Set the following when running generators or viewers:
```
export K3D_LOCAL_DIR="$(pwd)/../Knowledge3D.local"
export K3D_HOUSE_ID=default
```
Keep GLBs, logs, and dataset dumps inside `Knowledge3D.local/` to avoid polluting Git.

## Viewer & Tablet Notes

- `viewer/public/.gitkeep` maintains the folder for Vite builds; copy generated assets into `Knowledge3D.local/viewer_public/` instead.
- `Large_Assets_Kitchen` describes how to rebuild galaxy/house GLBs via `npm run build` and `knowledge3d.tools` exporters.

## Data Seeds

The remaining files inside `data/` are small seeds (intent templates, ontologies, sample GLBs). When promoting new datasets, store the heavy outputs in `.local` and document the process in `Large_Assets_Kitchen`.

## Testing & Tooling

- `tests/` runs with `pytest -q` and assumes runtime assets are available through `.local`.
- `scripts/` expect you to activate the `k3dml` conda environment before execution.

Keep this map up to date whenever repository layout changes.
