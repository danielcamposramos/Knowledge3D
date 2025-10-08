# Knowledge3D — Embodied Cognition in 3D

> **Mission**: Build a shared spatial operating system where humans and AI cohabit one reality, reason through PTX‑native cognition, and consolidate memories as explorable worlds.

[![status](https://img.shields.io/badge/status-Phase_B_Active-green)](docs/ROADMAP.md) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE) [![FMEAI](https://img.shields.io/badge/Philosophy-FMEAI-purple)](docs/PHILOSOPHY.md)

---

## 1. What Lives Here

| Location | Purpose |
| --- | --- |
| `Knowledge3D/` | Clean PTX-first codebase (no large payloads) |
| `Knowledge3D.local/` | Runtime workspace with Houses, tablet logs, datasets, galaxy/house GLBs |
| `Old_Attempts/Legacy_Fancy_RAG/` | Manifests describing the deprecated fancy-RAG artifacts (no binaries) |
| `Large_Assets_Kitchen/` | Recipes for regenerating >99 MB assets inside `.local` |

All contributors must keep heavy outputs in `.local` and document how to rebuild them in `Large_Assets_Kitchen/README.md`.

---

## 2. System Overview

![Cognitive House](docs/images/cognitive_house.png)

### Dual Memory Spine
- **Galaxy (RAM)** — high-dimensional embeddings for fast reasoning.
- **House (Persistent)** — consolidated knowledge objects (books, gardens, workshops).
- **Museum (Cold)** — archived artifacts for audit trails.
- **Memory Tablet** — avatar interface to search, stream, and mutate knowledge (see `docs/HOUSE_GALAXY_TABLET.md`).

### Cranium Core
- Unified multimodal head (text, image, audio, video, 3D) built on PTX kernels.
- PTX runtime helpers sit under `knowledge3d/cranium/ptx_runtime/` (RPN engine, sleep compute, shape generator, galaxy serializers).
- No CPU fallbacks on hot paths; every reasoning loop runs on CUDA.

### Dual-Client Reality
- **Human viewer** (`viewer/`) renders the house/galaxy in Three.js.
- **AI client** reads the same GLBs through `extras.k3d` buffer views for semantic access.

![Avatar Workshop](docs/images/avatar_workshop.png)

Read the full architectural brief in [`docs/Jules_K3D_Whitepaper.md`](docs/Jules_K3D_Whitepaper.md) and the active roadmap in [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## 3. Documentation Jump Pad

| Topic | Link |
| --- | --- |
| Vision & philosophy | [`docs/VISION.md`](docs/VISION.md) |
| Cranium Core internals | [`docs/CRANIUM_CORE.md`](docs/CRANIUM_CORE.md) |
| Memory workflow & tablet contract | [`docs/HOUSE_GALAXY_TABLET.md`](docs/HOUSE_GALAXY_TABLET.md) |
| PTX fused-head plan | [`docs/PTX_FUSED_HEAD_PLAN.md`](docs/PTX_FUSED_HEAD_PLAN.md) |
| Training directives & prompt hygiene | [`docs/TRAINING_DIRECTIVES.md`](docs/TRAINING_DIRECTIVES.md) |
| Environment policy (Conda, CUDA, tmux) | [`docs/ENV_POLICY.md`](docs/ENV_POLICY.md) |
| Dual code / HR-MR strategy | [`docs/DUAL_CODE_STRATEGY.md`](docs/DUAL_CODE_STRATEGY.md) |
| Doors & network addressing | [`docs/DOORS_AND_NETWORK.md`](docs/DOORS_AND_NETWORK.md) |
| glTF extension spec | [`spec/glTF_K3D_extension.md`](spec/glTF_K3D_extension.md) |

Collaboration practices for AI agents are in [`AGENTS.md`](AGENTS.md). Multi‑Vibe chain case studies live under `docs/reports/multi_vibe_chain/`.

---

## 4. Getting Started

### 4.1 Install
```bash
git clone https://github.com/danielcamposramos/Knowledge3D.git
cd Knowledge3D

# Python dependencies (activate the k3dml Conda env per docs/ENV_POLICY.md)
pip install -e .

# Viewer (Three.js + Vite)
cd viewer && npm install
```

### 4.2 Runtime Workspace
```bash
mkdir -p ../Knowledge3D.local
export K3D_LOCAL_DIR="$(pwd)/../Knowledge3D.local"
export K3D_HOUSE_ID=default
```
`Knowledge3D.local/` will hold Houses, galaxy GLBs, logs, and benchmarks. The repo stays lean.

### 4.3 Launch the Viewer + Bridge
```bash
# Terminal 1: WebSocket bridge (GPU environment)
cd Knowledge3D
scripts/k3d_env.sh run python -m knowledge3d.bridge.live_server --port 8787

# Terminal 2: Viewer
cd Knowledge3D/viewer
npm run dev   # open http://localhost:5173/?ws=ws://localhost:8787
```

### 4.4 Generate a Sample Galaxy
```bash
scripts/k3d_env.sh run python -m knowledge3d.tools.build_ai_books \
  --input data/intent_templates/en.yaml \
  --out "$K3D_LOCAL_DIR/datasets/ai_books_sample.glb" \
  --limit 200
```
View the GLB through the tablet or import it into the viewer via `viewer/public/` when needed.

---

## 5. PTX Runtime Modules

The PTX helpers are now centralized in `knowledge3d/cranium/ptx_runtime/`:

- `modular_rpn_engine.py` — GPU RPN execution (math, honesty, geometry ops).
- `text_to_3d_generator.py` — prompt-to-geometry generator (writes embedded GLBs).
- `sleep_time_compute.py` — nightly consolidation coordinator.
- `thinking_tag_embedder.py` — tag generator for reflections and tablet.
- `galaxy_state_serializer.py` / `galaxy_memory_updater.py` — load/save galaxy state into Houses.
- `nvrtc_ptx_loader.py` — NVRTC compilation harness for dynamic kernels.

Import them through:
```python
from knowledge3d.cranium.ptx_runtime import ModularRPNEngine, SleepTimeCompute
```

Legacy `phase*/` directories have been removed from Git; reference manifests if you need to rebuild artifacts.

---

## 6. Repository Layout

```
Knowledge3D/
├─ knowledge3d/                 # Core Python package
│  ├─ cranium/                  # Fused head runtime + PTX helpers
│  ├─ bridge/                   # Tablet + viewer WebSocket server
│  ├─ gpu/, spatial/, skills/   # CUDA utilities, navigation kernels, multimodal skills
│  ├─ tools/                    # Dataset builders & utilities (phase-neutral)
│  └─ ...
├─ viewer/                      # Human client (Three.js + TypeScript)
├─ Large_Assets_Kitchen/        # Regeneration recipes for heavy assets
├─ Old_Attempts/Legacy_Fancy_RAG/  # Manifests for archived fancy-RAG artifacts
├─ docs/                        # Specs, briefs, roadmap, playbooks
├─ scripts/                     # Shell helpers (training, ingestion, CI utilities)
├─ spec/                        # Formal schema & protocol definitions
├─ tests/                       # Pytest suite (GPU + integration tests)
└─ README.md                    # You are here
```

---

## 7. Contributing

1. **Respect the memory policy** (`docs/HOUSE_GALAXY_TABLET.md`).
2. **Stay GPU-first**: PTX kernels or CUDA extensions for any hot path.
3. **Keep heavy artifacts local**: document regeneration steps instead of committing binaries.
4. **Follow agent guidelines** when using AI automation (`AGENTS.md`).
5. Run `pytest -q` (and viewer tests when applicable) before opening a PR.

Security, ethics, and embodiment commitments are detailed in [`docs/COVENANT.md`](docs/COVENANT.md) and [`docs/CARE_PROTOCOL.md`](docs/CARE_PROTOCOL.md).

---

## 8. Community & Roadmap

- **Roadmap status**: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **Recent progress notes**: [`docs/PHASE7_RPN_INTEGRATION_COMPLETE.md`](docs/PHASE7_RPN_INTEGRATION_COMPLETE.md)
- **Swarm collaboration logs**: `docs/reports/multi_vibe_chain/`
- **Audio/voice architecture**: [`docs/AUDIO_ARCH.md`](docs/AUDIO_ARCH.md)

If you are interested in partnering, reach out via the contact information in `docs/Jules_K3D_Whitepaper.md`.

---

Together we are building the first spatial operating system for thought. Dive into the docs, regenerate the local assets you need, and help us fuse the Galaxy and the House into a living, embodied intelligence.
