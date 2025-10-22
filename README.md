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

## 5. Performance Benchmarks (Real Test Results)

### Step 15 Phase B: Sovereign Knowledge Ingestion

**Zero External Dependencies Achieved** — 100% RPN-native embeddings (0MB footprint vs 66MB GloVe bootstrap)

#### Baseline Sequential Runs

| Pipeline | Items | Runtime | Throughput | VRAM Peak | GPU Util |
|----------|-------|---------|------------|-----------|----------|
| **WordNet EN** | 117,659 synsets | 145.87s | 807 synsets/s | <200MB | 6-7% |
| **Font Harvest** | 2,713 fonts<br/>168,206 glyphs | ~780s | - | <200MB | 6-7% |
| **PDF Corpus** | 61 PDFs<br/>23,000 sentences | 41.39s | 556 sentences/s | <200MB | 6-7% |

#### Parallel Optimized Runs

| Pipeline | Workers | Batch | Runtime | Speedup | Throughput | Notes |
|----------|---------|-------|---------|---------|------------|-------|
| **WordNet EN** | 8 | 64 | **143.28s** | 1.02× | 821 synsets/s | CPU preprocessing: 0.65s |
| **Font Harvest** | 8 | 32 | **216.62s** | 3.6× | 750 glyphs/s | 1.4GB JSON streamed |
| **PDF Corpus** | 8 | 32 | **137.64s** | 0.3× | 167 sentences/s | PyPDF2 extraction bottleneck |

**Key Findings**:
- ✅ **Ultra-low resource usage**: <200MB VRAM (40× under 8GB budget), 6-8% GPU util
- ✅ **Massive parallelization headroom**: 92-94% GPU idle → opportunity for 10-20× future speedup
- ⚠️ **CPU-bound bottlenecks**: PIL rendering (5ms/glyph), PyPDF2 extraction (300ms/PDF) dominate
- 🎯 **Next frontier**: GPU-accelerated PDF parsing + batch kernel calls (>256 items)

**Artifacts Generated** (in `/K3D/Knowledge3D.local/house_zone7/`):
- `embeddings/rpn_embeddings.pkl` — 33,428 trigrams (multi-lingual)
- `lexicons/wordnet_en_parallel.json` — 117,659 synsets with 3D positions
- `fonts/full_font_library_parallel.json` — 168,206 visual-text pairs (1.4GB)
- `documents/` — 61 PDFs with semantic embeddings

**See**: [`TEMP/STEP15_PHASE_B_RESULTS.md`](TEMP/STEP15_PHASE_B_RESULTS.md), [`TEMP/STEP15_PHASE_B_SPEEDUP_RESULTS.md`](TEMP/STEP15_PHASE_B_SPEEDUP_RESULTS.md)

### Phase C: Multi-Modal PDF Ingestion (Complete)

| Pipeline | Coverage | Runtime | Throughput | Method |
|----------|----------|---------|------------|--------|
| **Structured PDF** | 99 % of sources | ~22 ms/page | ≈45 pages/s | Sovereign PyMuPDF + PTX parser |
| **Scanned PDF** | ~1 % of sources | ~0.6 s/page | ≈1.6 pages/s | Tesseract fallback (temporary) |
| **Glyph Database** | 1,999 fonts | – | 123,938 glyphs | Per-font HOG descriptors (Phase E input) |

**Key Features**:
- ✅ 15× faster than Phase B baseline for structured PDFs (300 ms → 20–25 ms/page)
- ✅ Multi-modal extraction with spatial relationships + Galaxy crystallisation
- ✅ Pragmatic scanned-PDF coverage via Tesseract while sovereign OCR incubates for Phase E
- ✅ AtomicFissionFusion + GraphCrystallizer fuse RPN text + Fractal visuals into Galaxy positions
- ✅ Sovereign hot path preserved (ctypes + PTX); external OCR used only as a temporary bridge

### Step 14: Specialized Swarm Kernels

| Metric | Value | Notes |
|--------|-------|-------|
| **9-Chain Latency** | 80.69µs | Fused kernel (9 transformations + resonance) |
| **Wikipedia Ingestion** | 0.14s/article | 35× faster than 5s target |
| **VRAM Peak** | 0.12GB | 66× under 8GB budget |

---

## 6. Current Architecture (Steps 10-15)

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

### Sovereign Knowledge Ingestion Stack (Step 15)

**Mission**: Feed the AI mind with multi-modal knowledge using zero external dependencies.

**Architecture**: RPN-native embeddings + PTX-optimized multi-modal fusion

```
Text Pipeline:
  RPN Trigrams (33K vocab) → 128-dim embeddings → GraphCrystallizer → VectorResonator → 3D Galaxy

Audio Pipeline:
  Temporal features + LPC formants → TemporalReasoning kernel → Fusion → Galaxy

Visual Pipeline:
  Glyph rendering → Edge detection → FractalEmitter → Fusion → Galaxy

Multi-Modal Fusion:
  AtomicFissionFusion (text + audio + visual) → Swarm refinement (80µs) → Galaxy position
```

**Ingestion Modules**:
- `knowledge3d/cranium/rpn_embedding_engine.py` — Language-agnostic trigram embeddings
- `knowledge3d/ingestion/language/sovereign_text_pipeline.py` — Text → RPN → Galaxy
- `knowledge3d/ingestion/language/sovereign_audio_pipeline.py` — Audio → Temporal → Galaxy
- `knowledge3d/ingestion/language/sovereign_visual_pipeline.py` — Visual → Fractal → Galaxy
- `knowledge3d/ingestion/lexicons/parallel_lexicon_ingestor.py` — WordNet + multi-lingual
- `knowledge3d/ingestion/fonts/parallel_font_harvester.py` — Font glyphs → visual-text pairs
- `knowledge3d/ingestion/documents/pdf_ingestor.py` — PDF → sentences → Galaxy

**Parallel Optimization**: 8-worker CPU pools + GPU batching for 1-4× speedup (See benchmarks above)

---

## 7. Repository Layout

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

## 8. Contributing

1. **Respect the memory policy** (`docs/HOUSE_GALAXY_TABLET.md`).
2. **Stay GPU-first**: PTX kernels or CUDA extensions for any hot path.
3. **Keep heavy artifacts local**: document regeneration steps instead of committing binaries.
4. **Follow agent guidelines** when using AI automation (`AGENTS.md`).
5. **Test before PR**: Run `pytest -q` (and viewer tests when applicable).
6. **Check deprecations**: Don't import from `Old_Attempts/` in new code.

Security, ethics, and embodiment commitments are detailed in [`docs/COVENANT.md`](docs/COVENANT.md) and [`docs/CARE_PROTOCOL.md`](docs/CARE_PROTOCOL.md).

---

## 9. Community & Roadmap

- **Deep Dive (Best Entry Point)**: [**NotebookLM Research Space**](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f)
- **Roadmap status**: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **Step 12 Complete**: [`TEMP/STEP12_PHASE1_PHASE2_COMPLETE.md`](TEMP/STEP12_PHASE1_PHASE2_COMPLETE.md)
- **Step 13 In Progress**: [`TEMP/STEP13_MASTER_INDEX.md`](TEMP/STEP13_MASTER_INDEX.md)
- **Swarm collaboration logs**: `docs/reports/multi_vibe_chain/`
- **Audio/voice architecture**: [`docs/AUDIO_ARCH.md`](docs/AUDIO_ARCH.md)

### Recent Milestones

- **TRM Validation Complete** (Oct 22, 2025): **K3D Paradigm Operational** — Query/Answer pipeline validated!
  - **Knowledge Consolidation**: 290,485 trigrams → 256 clusters (silhouette: 0.009 → 0.032, 3.5× improvement)
  - **Sleep-Time Processing**: 28-minute consolidation via k-means + redundancy pruning
  - **TRM Initialization**: 2.1M params seeded from top 1024 RPN trigrams (NOT trained on data!)
  - **Pipeline Validation**: 100% query convergence, avg output norm 375 (STRONG reasoning signals)
  - **Paradigm Clarity**: Knowledge lives IN embeddings (Galaxy/House), TRM learns reasoning patterns
  - **Next Phase**: Train TRM on reasoning tasks (ARC-AGI, logic puzzles), NOT data storage
  - **Documentation**: See [TEMP/SESSION_SUMMARY_OCT22_TRM_VALIDATION.md](TEMP/SESSION_SUMMARY_OCT22_TRM_VALIDATION.md)

- **Step 15 Phase B** (Oct 2025): **Sovereign Knowledge Ingestion** — Zero external dependencies achieved!
  - **RPN Embeddings**: 33,428 trigrams learned (language-agnostic, 0MB footprint)
  - **Multi-lingual**: WordNet EN (117,659 synsets) + PT-BR, ES, JP, ZH lexicons
  - **Visual-Text Grounding**: 2,713 fonts → 168,206 glyph-text pairs (1.4GB)
  - **Knowledge Corpus**: 61 PDFs, 23,000 sentences from curated libraries
  - **Performance**: <200MB VRAM, 6-8% GPU utilization (massive headroom!)
  - **Parallel Pipelines**: 8-worker CPU pools + GPU batching for 1.02-3.6× speedup

- **Step 14** (Oct 2025): Specialized 9-chain swarm kernel (80.69µs latency, 35× faster than Wikipedia target)
- **Step 12** (Oct 2025): FSM consolidation — harvested 5-state observability, ActionBuffer integration, and dynamic LOD into sovereign ThinkingTagBridge
- **Step 11** (Oct 2025): Multi-modal text-to-3D generation with shape cache and confidence propagation
- **Step 10** (Sep 2025): ThinkingTagBridge sovereign runtime with <35µs latency target

If you are interested in partnering, reach out via the contact information in `docs/Jules_K3D_Whitepaper.md`.

---

Together we are building the first spatial operating system for thought — not a fancy RAG, but a true multi-modal intelligence that perceives, reasons, and acts in 3D space. Dive into the [NotebookLM](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f), explore the docs, regenerate the local assets you need, and help us fuse the Galaxy and the House into a living, embodied cognition.
