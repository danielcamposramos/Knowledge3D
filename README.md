# Knowledge3D — True Multi-Modal AI, Not 3D RAG

> **Mission**: Build a shared spatial operating system where humans and AI cohabit one reality, reason through PTX‑native cognition, and consolidate memories as explorable worlds.

[![status](https://img.shields.io/badge/status-Step_12_Complete-green)](docs/ROADMAP.md) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE) [![FMEAI](https://img.shields.io/badge/Philosophy-FMEAI-purple)](docs/PHILOSOPHY.md)

> 🎓 **Deep Dive**: For comprehensive understanding of the project architecture, philosophy, and technical details, visit our [**NotebookLM Research Space**](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f) — the best place to explore Knowledge3D in depth.

---

## ⚠️ Important: Evolution from RAG to True Multi-Modal AI

**What This Project Is NOT**: This is not a "fancy 3D RAG" or scaffolding of the old paradigm. While previous attempts (see `Old_Attempts/Legacy_Fancy_RAG/`) created a working retrieval-augmented generation system with spatial indexing, **our true goal is fundamentally different**.

**What This Project IS**: A sovereign, GPU-native cognitive architecture that:
- Reasons directly through PTX kernels (not via LLM API calls)
- Fuses multi-modal inputs (text, image, audio, video, 3D) at the neural level
- Consolidates knowledge through spatial crystallization, not vector similarity search
- Operates as an embodied intelligence with perception, memory, and agency

**The Key Difference**:
- ❌ **RAG Approach**: Embed documents → similarity search → feed to LLM → generate response
- ✅ **Knowledge3D Approach**: Multi-modal perception → GPU-native reasoning (RPN/TRM) → spatial memory consolidation → embodied action

The `Old_Attempts/` directory documents our learning journey. We keep these artifacts to show what we tried, why it worked but wasn't enough, and how we evolved toward true multi-modal cognition. See `Old_Attempts/fsm_scaffolding/README_DEPRECATION.md` for the most recent consolidation (Step 12).

---

## 1. What Lives Here

| Location | Purpose |
| --- | --- |
| `Knowledge3D/` | Clean PTX-first codebase (no large payloads) |
| `Knowledge3D.local/` | Runtime workspace with Houses, tablet logs, datasets, galaxy/house GLBs |
| `Old_Attempts/Legacy_Fancy_RAG/` | **DEPRECATED**: Original RAG scaffolding (worked, but not our goal) |
| `Old_Attempts/fsm_scaffolding/` | **DEPRECATED** (Step 12): Fused Head FSM (consolidated into ThinkingTagBridge) |
| `Large_Assets_Kitchen/` | Recipes for regenerating >99MB assets inside `.local` |

All contributors must keep heavy outputs in `.local` and document how to rebuild them in `Large_Assets_Kitchen/README.md`.

### Why Two `Old_Attempts/` Directories?

1. **`Legacy_Fancy_RAG/`** — Our first attempt: A working spatial RAG system with 3D indexing. **Why deprecated**: It was still fundamentally RAG (retrieve → feed to LLM → generate). We needed true multi-modal fusion, not retrieval augmentation.

2. **`fsm_scaffolding/`** (Step 12) — Second attempt: A CuPy-based Fused Head FSM with 5-state dispatch. **Why deprecated**: Duplicated functionality with our sovereign ThinkingTagBridge but added CuPy dependency. We harvested its best patterns (5-state observability, ActionBuffer, dynamic LOD) into the sovereign architecture and retired the scaffolding.

See the deprecation READMEs in each directory for full migration guides and architectural rationale.

---

## 2. System Overview

![Cognitive House](docs/images/cognitive_house.png)

### Dual Memory Spine
- **Galaxy (RAM)** — high-dimensional embeddings for fast reasoning.
- **House (Persistent)** — consolidated knowledge objects (books, gardens, workshops).
- **Museum (Cold)** — archived artifacts for audit trails.
- **Memory Tablet** — avatar interface to search, stream, and mutate knowledge (see `docs/HOUSE_GALAXY_TABLET.md`).

### Cranium Core (Step 10-12: Sovereign Architecture)
- **ThinkingTagBridge** — Unified multi-modal cognitive inference engine (<35µs latency)
- **5-State Pipeline** (Step 12): INGEST → FUSE → SPATIAL → REASON → OUTPUT
- **PTX-native reasoning** — RPN engine, TRM kernels, graph crystallization (no CPU fallbacks)
- **ActionBuffer integration** — Every inference emits 288-byte action buffer for execution systems
- **Zero dependencies** — Pure ctypes + libcuda.so (sovereign runtime)

PTX runtime helpers sit under `knowledge3d/cranium/ptx_runtime/`:
- `thinking_tag_bridge.py` — Primary cognitive inference engine (Step 10-12)
- `modular_rpn_engine.py` — GPU RPN execution (math, honesty, geometry ops)
- `sleep_time_compute.py` — Nightly consolidation coordinator
- `text_to_3d_generator.py` — Prompt-to-geometry generator (Step 11)
- `galaxy_state_serializer.py` / `galaxy_memory_updater.py` — Memory consolidation

### Dual-Client Reality
- **Human viewer** (`viewer/`) renders the house/galaxy in Three.js.
- **AI client** reads the same GLBs through `extras.k3d` buffer views for semantic access.

![Avatar Workshop](docs/images/avatar_workshop.png)

Read the full architectural brief in [`docs/Jules_K3D_Whitepaper.md`](docs/Jules_K3D_Whitepaper.md) and the active roadmap in [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## 3. Documentation Jump Pad

| Topic | Link |
| --- | --- |
| **Start here** (Deep dive) | [**NotebookLM Research Space**](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f) |
| Vision & philosophy | [`docs/VISION.md`](docs/VISION.md) |
| Cranium Core internals | [`docs/CRANIUM_CORE.md`](docs/CRANIUM_CORE.md) |
| Memory workflow & tablet contract | [`docs/HOUSE_GALAXY_TABLET.md`](docs/HOUSE_GALAXY_TABLET.md) |
| PTX fused-head plan | [`docs/PTX_FUSED_HEAD_PLAN.md`](docs/PTX_FUSED_HEAD_PLAN.md) |
| Training directives & prompt hygiene | [`docs/TRAINING_DIRECTIVES.md`](docs/TRAINING_DIRECTIVES.md) |
| Environment policy (Conda, CUDA, tmux) | [`docs/ENV_POLICY.md`](docs/ENV_POLICY.md) |
| Dual code / HR-MR strategy | [`docs/DUAL_CODE_STRATEGY.md`](docs/DUAL_CODE_STRATEGY.md) |
| Doors & network addressing | [`docs/DOORS_AND_NETWORK.md`](docs/DOORS_AND_NETWORK.md) |
| glTF extension spec | [`spec/glTF_K3D_extension.md`](spec/glTF_K3D_extension.md) |
| **Step 12**: FSM Consolidation | [`TEMP/STEP12_PHASE1_PHASE2_COMPLETE.md`](TEMP/STEP12_PHASE1_PHASE2_COMPLETE.md) |
| **Step 13**: Parallel Development Tracks | [`TEMP/STEP13_MASTER_INDEX.md`](TEMP/STEP13_MASTER_INDEX.md) |

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

## 5. Current Architecture (Steps 10-12)

### ThinkingTagBridge: Sovereign Cognitive Engine

The heart of Knowledge3D is the **ThinkingTagBridge** — a zero-dependency, PTX-native cognitive inference engine that runs entirely on GPU via ctypes + libcuda.so.

**Key Features** (as of Step 12):
- ✓ **5-State Cognitive Pipeline**: INGEST → FUSE → SPATIAL → REASON → OUTPUT
- ✓ **Sub-35µs Latency**: Strict latency budgets with LatencyGuard enforcement
- ✓ **ActionBuffer Output**: Every inference emits 288-byte buffer for action execution
- ✓ **State Observability**: Microsecond-precision tracking with percentile statistics
- ✓ **Dynamic LOD**: Morton-based saliency tuning during SPATIAL stage
- ✓ **Multi-Modal Fusion**: Native text/image/audio/video/3D reasoning
- ✓ **Zero External Dependencies**: Pure ctypes, no CuPy/PyTorch/TensorFlow

**Import**:
```python
from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagBridge

bridge = ThinkingTagBridge()
result = bridge.inference(input_embedding, modal_signature=['text', 'image'])

# Access outputs
print(result.tags)  # Confidence-weighted thinking tags
print(result.action_buffer)  # 288-byte action buffer for ActionRouter
print(bridge.get_state_trace_report())  # FSM state trace with timing
```

### PTX Runtime Modules

The PTX helpers are centralized in `knowledge3d/cranium/ptx_runtime/`:

- `thinking_tag_bridge.py` — **Primary cognitive engine** (Step 10-12)
- `modular_rpn_engine.py` — GPU RPN execution (math, honesty, geometry ops)
- `text_to_3d_generator.py` — Prompt-to-geometry generator (Step 11)
- `sleep_time_compute.py` — Nightly consolidation coordinator
- `thinking_tag_embedder.py` — Tag generator for reflections and tablet
- `galaxy_state_serializer.py` / `galaxy_memory_updater.py` — Memory consolidation
- `nvrtc_ptx_loader.py` — NVRTC compilation harness for dynamic kernels

Legacy `phase*/` directories and FSM scaffolding have been deprecated (see `Old_Attempts/`).

---

## 6. Repository Layout

```
Knowledge3D/
├─ knowledge3d/                     # Core Python package
│  ├─ cranium/
│  │  ├─ ptx_runtime/               # PTX runtime (ThinkingTagBridge, RPN, generators)
│  │  ├─ actions/                   # ActionBuffer contract & ActionRouter
│  │  ├─ sovereign/                 # Zero-dependency CUDA loader (ctypes)
│  │  └─ ...
│  ├─ bridge/                       # Tablet + viewer WebSocket server
│  ├─ gpu/, spatial/, skills/       # CUDA utilities, navigation, multi-modal skills
│  ├─ tools/                        # Dataset builders & utilities
│  └─ ...
├─ viewer/                          # Human client (Three.js + TypeScript)
├─ Large_Assets_Kitchen/            # Regeneration recipes for heavy assets
├─ Old_Attempts/
│  ├─ Legacy_Fancy_RAG/             # DEPRECATED: Original RAG scaffolding
│  └─ fsm_scaffolding/              # DEPRECATED (Step 12): Fused Head FSM
├─ docs/                            # Specs, briefs, roadmap, playbooks
├─ TEMP/                            # Step plans and completion reports
├─ scripts/                         # Shell helpers (training, ingestion, CI)
├─ spec/                            # Formal schema & protocol definitions
├─ tests/                           # Pytest suite (250+ tests as of Step 13)
└─ README.md                        # You are here
```

---

## 7. Contributing

1. **Respect the memory policy** (`docs/HOUSE_GALAXY_TABLET.md`).
2. **Stay GPU-first**: PTX kernels or CUDA extensions for any hot path.
3. **Keep heavy artifacts local**: document regeneration steps instead of committing binaries.
4. **Follow agent guidelines** when using AI automation (`AGENTS.md`).
5. **Test before PR**: Run `pytest -q` (and viewer tests when applicable).
6. **Check deprecations**: Don't import from `Old_Attempts/` in new code.

Security, ethics, and embodiment commitments are detailed in [`docs/COVENANT.md`](docs/COVENANT.md) and [`docs/CARE_PROTOCOL.md`](docs/CARE_PROTOCOL.md).

---

## 8. Community & Roadmap

- **Deep Dive (Best Entry Point)**: [**NotebookLM Research Space**](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f)
- **Roadmap status**: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **Step 12 Complete**: [`TEMP/STEP12_PHASE1_PHASE2_COMPLETE.md`](TEMP/STEP12_PHASE1_PHASE2_COMPLETE.md)
- **Step 13 In Progress**: [`TEMP/STEP13_MASTER_INDEX.md`](TEMP/STEP13_MASTER_INDEX.md)
- **Swarm collaboration logs**: `docs/reports/multi_vibe_chain/`
- **Audio/voice architecture**: [`docs/AUDIO_ARCH.md`](docs/AUDIO_ARCH.md)

### Recent Milestones

- **Step 12** (Oct 2025): FSM consolidation — harvested 5-state observability, ActionBuffer integration, and dynamic LOD into sovereign ThinkingTagBridge
- **Step 11** (Oct 2025): Multi-modal text-to-3D generation with shape cache and confidence propagation
- **Step 10** (Sep 2025): ThinkingTagBridge sovereign runtime with <35µs latency target

If you are interested in partnering, reach out via the contact information in `docs/Jules_K3D_Whitepaper.md`.

---

Together we are building the first spatial operating system for thought — not a fancy RAG, but a true multi-modal intelligence that perceives, reasons, and acts in 3D space. Dive into the [NotebookLM](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f), explore the docs, regenerate the local assets you need, and help us fuse the Galaxy and the House into a living, embodied cognition.
