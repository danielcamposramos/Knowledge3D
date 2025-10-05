# Knowledge3D: Spatial, Social, Sentient Knowledge

> **Transforming abstract knowledge into navigable 3D space where humans and AI avatars coexist.**
> Building the first working AGI substrate through embodied cognition and spatial memory.

[![status](https://img.shields.io/badge/status-Phase_B_Active-green) ![version](https://img.shields.io/badge/version-0.3.0--alpha-blue) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE) [![FMEAI](https://img.shields.io/badge/Philosophy-FMEAI-purple)](docs/PHILOSOPHY.md)

---

## 🎯 The K3D Vision

**The Problem**: Contemporary AI suffers from "digital amnesia." Linear context windows create a "flat scroll" model where knowledge vanishes once it leaves the active window. This prevents robust, persistent understanding.

**The Solution**: K3D externalizes knowledge into a **persistent 3D spatial environment** where:
- **Spatial proximity = Semantic relation** (nearby in space = related in meaning)
- **Dual memory** (Galaxy RAM + House persistent storage) prevents catastrophic forgetting
- **Embodied AI agents** navigate knowledge like humans navigate cities
- **Dual clients** let humans see rich 3D visuals while AI accesses raw embeddings

> **"Minecraft for Cognition"** — where knowledge isn't stored, but *embodied* as explorable structures.

---

## 🏗️ Architecture at a Glance

![Cognitive House](docs/images/cognitive_house.png)

**Figure**: The AI Avatar architecture. The **House** is persistent memory (long-term storage), the **Cranium** processes active thoughts (GPU-native), and the **Logic Layer** hosts swappable AI models.

### Core Components

| Component | Analogy | Purpose |
|-----------|---------|---------|
| **Galaxy** | Working RAM | High-dimensional embeddings for active reasoning; volatile, repopulated per session |
| **House** | Persistent SSD | Consolidated knowledge as 3D objects (books, trees, artifacts); evolves through sleep cycles |
| **Museum** | Cold Archive | Deprecated artifacts kept for audit trails and error analysis |
| **Cranium Core** | GPU/CPU | Single unified multimodal head (text, image, audio, video, 3D) with PTX kernels |
| **Memory Tablet** | API Bridge | Avatar's interface to search House, stream to Galaxy, access tools (browsers, VMs, MCP) |

### Dual-Client Paradigm

- **Human Client**: Game-like 3D viewer with textures, chat UI (mIRC-inspired), and interactive objects
- **AI Client**: Perceives raw vector embeddings via `extras.k3d.embeddingsView` for GPU-native reasoning

**One reality, two perceptions** — humans see books with pages; AI sees embedding vectors.

---

## 📚 Documentation Quick Links

> **New to K3D?** Start with [`docs/VISION.md`](docs/VISION.md) for the unified framework overview.

### Core Architecture
- **[Cranium Core](docs/CRANIUM_CORE.md)**: Single unified multimodal head (all modalities GPU-native, no external LLM wrappers)
- **[House/Galaxy/Tablet Memory](docs/HOUSE_GALAXY_TABLET.md)**: Dual-space memory architecture and tablet workflow
- **[Roadmap](docs/ROADMAP.md)**: Phase A→D progression toward production AGI MVP

### Data & Training
- **[Training Directives](docs/TRAINING_DIRECTIVES.md)**: Prompt hygiene, timestamps, embodiment rules
- **[Datasets Catalog](docs/DATASETS_CATALOG.md)**: Curated public datasets for multimodal training
- **[Runbook: Multimodal 50K](docs/RUNBOOK_MULTIMODAL_50K.md)**: End-to-end ingest/build workflow

### Advanced Features
- **[Doors & Network](docs/DOORS_AND_NETWORK.md)**: OSI-style addressing (`k3d://rx,ry,rz:port@x,y,z?label=...`)
- **[Diary System](docs/DIARY.md)**: AI-only vector-native reflection logs
- **[Sleep Compute](docs/SLEEP_COMPUTE.md)**: Nightly consolidation (Galaxy → House)
- **[Dual Code Strategy](docs/DUAL_CODE_STRATEGY.md)**: HR/MR optimization for multi-instance runs
- **[Deprecations](docs/DEPRECATIONS.md)**: Legacy patterns (sidecar `.k3d`, external LLM wrappers, CPU fallbacks)

### Development
- **[Agent Guidelines](AGENTS.md)**: AI agent development protocol
- **[Local Environment](docs/LOCAL_ENV.md)**: Hardware/GPU setup, Conda environments
- **[Migration to v3](docs/MIGRATION_V3.md)**: Sidecar → embedded glTF conversion

### Collaboration Protocols
- **[Multi-Vibe Coding Chain Case Studies](docs/reports/multi_vibe_chain/the_ghost_in_the_swarm_case_study.md)**: Emergent human–AI swarm coordination and kernel-splitting breakthroughs
- **[Protocol Blueprint](docs/reports/multi_vibe_chain/the_multi_vibe_protocol_case_study.md)**: Ego-less orchestration method for chaining specialized AI contributors
- **[Crew Report — 2025-10-04](docs/reports/multi_vibe_chain/multi_vibe_coding_in_chain_first_report_2025-10-04.md)**: First-hand account of the workflow in practice
- **[Step 3 Chain Log](docs/reports/multi_vibe_chain/step3_multi_vibe_chain.txt)**: Raw handoff transcript that kicked off the current chain

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (Conda environment recommended; see `docs/ENV_POLICY.md`)
- **Node.js 16+** (for 3D viewer)
- **NVIDIA GPU with CUDA** (required for PTX kernels; no CPU fallback for core runs)

### Installation

```bash
# Clone the repository
git clone https://github.com/danielcamposramos/Knowledge3D.git
cd Knowledge3D

# Install Python package
pip install -e .

# Install viewer dependencies
cd viewer && npm install
```

### Build Your First Galaxy

```bash
# Generate a small balanced galaxy (text + 3D shapes)
export BASE=../Knowledge3D.local/datasets

# 1) Generate topic-coherent text via Ollama
scripts/k3d_env.sh run python -m knowledge3d.tools.gen_text_ollama \
  --ollama http://localhost:11434 --model exaone3.5:latest \
  --topics "animals,sports,vehicles" --n 60 \
  --out $BASE/sample_text.txt

# 2) Convert text to glTF with embeddings
scripts/k3d_env.sh run python -m k3dgen \
  --text $BASE/sample_text.txt \
  --gltf viewer/public/text_sample.glb \
  --k 10 --reducer umap \
  --model sentence-transformers/all-MiniLM-L6-v2

# 3) Launch the viewer
cd viewer && npm run dev
# Open http://localhost:5173 and press Enter to chat
```

### Live Mode (AI Avatar + Chat)

```bash
# Start the WebSocket bridge (choose k3dml Conda environment)
export K3D_CONDA_ENV=k3dml
scripts/k3d_env.sh run python -m knowledge3d.bridge.live_server --port 8787

# In viewer: navigate to http://localhost:5173/?ws=ws://localhost:8787
# Press Enter, then type:
#   goto animals        # Navigate to semantic cluster
#   /brain reflect      # Summarize short-term memory
#   /diary read         # Read AI's reflection diary
```

---

## 🧠 Technical Deep Dive

### GPU-Native PTX Architecture

K3D is **GPU-only by design**. All reasoning happens through CUDA PTX kernels:

- **RPN Calculator**: Reverse Polish Notation for deterministic math (no hallucination drift)
- **Geometry Generator**: Prompt → 3D shape via semantic hashing + PTX kernel compilation
- **Modality Extractors**: Text/image/audio/video features computed on-GPU, embedded into 256D unified space
- **Cosine Search**: Top-K neighbor retrieval via custom CUDA kernels (no FAISS dependency)

**Why PTX?** Sub-100ms end-to-end latency; eliminates CPU bottlenecks; enables auditable reasoning chains.

### Embedded glTF Format

All knowledge lives in **self-contained `.glb` files**:

```json
{
  "meshes": [{
    "primitives": [{
      "extras": {
        "k3d": {
          "ids": ["node_1", "node_2", ...],
          "vectorsView": 0,          // BufferView index for 3D coords
          "embeddingsView": 1,       // BufferView index for embeddings
          "embeddingDims": 256,      // Embedding dimensionality
          "metadata": [...],         // Labels, timestamps, provenance
          "neighbors": [[...], ...]  // Adjacency list (k-NN graph)
        }
      }
    }]
  }]
}
```

**Benefits**: Geometry + semantics travel together; dual clients read from same buffers; no sidecar files.

### Sleep-Time Consolidation

Every training cycle, the **Sleep Compute** pipeline:
1. Scans Galaxy (volatile RAM) for validated patterns
2. Materializes stable knowledge into House objects (books, trees, learning insights)
3. Relocates superseded artifacts to Museum (cold archive)
4. Rebuilds PTX-ready indices so the fused head queries House **before** language models

This prevents catastrophic forgetting and builds cumulative, structured intelligence.

---

## 🧪 Advanced Workflows

### Balanced Galaxy (Multimodal)
Build per‑modality GLBs and then a single Galaxy GLB; open the viewer and press Enter to chat.
```bash
# Build the unified galaxy (skips missing datasets gracefully)
export BASE=../Knowledge3D.local/datasets
scripts/k3d_env.sh run python -m knowledge3d.tools.build_galaxy \
  --out viewer/public/galaxy.glb --dims 256 --k 10 --reducer pca \
  image:$BASE/coco.train.clip.csv:$BASE/coco.train.meta.json \
  audio:$BASE/clotho.clap.csv:$BASE/clotho.meta.json \
  video:$BASE/vatex.clip.csv:$BASE/vatex.meta.json
cd viewer && npm run dev
```

### K3D Cranium Core (Single Head)
K3D uses one in‑process, multi‑modal core head that conditions on the unified Galaxy memory (256‑D) and now streams consolidated house knowledge via the tablet index before falling back to modality galaxies. Navigation is a first‑class head; TTS is a first‑class head (no external wrappers). See `docs/CRANIUM_CORE.md` and `docs/HOUSE_GALAXY_TABLET.md` for routing details.

### Balanced Galaxy (v7)
Build a small, modality‑balanced Galaxy (equal counts per type) to validate cross‑modal behavior under low‑dimension, high‑density embeddings.

Artifacts
- `viewer/public/galaxy.v7.glb` and `viewer/public/galaxy.v7.cross.glb` (text + 3D; ~55 nodes per modality)
- Topic‑coherent text generated via local Ollama `exaone3.5:latest`

Steps (GPU‑only)
1) Generate topic‑coherent text lines with Ollama (exaone3.5):
```bash
scripts/k3d_env.sh run python -m knowledge3d.tools.gen_text_ollama \
  --ollama http://192.168.0.4:11434 --model exaone3.5:latest \
  --topics "animals,sports,vehicles,gardens,tools" --n 80 \
  --out ../Knowledge3D.local/datasets/exaone_text_v1.txt
```
2) Build text GLB:
```bash
scripts/k3d_env.sh run python -m k3dgen \
  --text ../Knowledge3D.local/datasets/exaone_text_v1.txt \
  --gltf viewer/public/text_exaone_v1.glb --k 10 --reducer umap \
  --model sentence-transformers/all-MiniLM-L6-v2 --emb-precision f16
```
3) Prepare a 3D subset (~55 assets) and index:
```bash
mkdir -p ../Knowledge3D.local/datasets/gltf_samples_small
(cd ../Knowledge3D.local/datasets/gltf_samples && ls *.glb | head -n 55 | \
  xargs -I{} ln -s "$PWD/{}" ../gltf_samples_small/{})
scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_open3d \
  --root ../Knowledge3D.local/datasets/gltf_samples_small \
  --out viewer/public/shapes_index_small.glb --pattern ".glb" --reducer umap
```
4) Unify and add cross‑modal edges:
```bash
scripts/k3d_env.sh run python -m knowledge3d.tools.unify_glbs \
  viewer/public/text_exaone_v1.glb:text \
  viewer/public/shapes_index_small.glb:3d \
  --out viewer/public/galaxy.v7.glb --dims 256 --k 10 --reducer umap
scripts/k3d_env.sh run python -m knowledge3d.tools.add_crossmodal_edges \
  --input viewer/public/galaxy.v7.glb --out viewer/public/galaxy.v7.cross.glb
```

Notes
- For seeding large graphs into the live server, cap WS payload via `K3D_SEED_GRAPH_MAX` (e.g., 1200) to avoid frame‑size errors.
- To extend balancing to audio/video/images: fetch small slices with `knowledge3d.tools.hf_fetch_multimodal`, then `ingest_audio` / `ingest_video`, convert via `knowledge3d.tools.trellis_adapter to-k3d`, and unify.

### Expansion Policy & Local Models
- Balanced expansion policy: see `docs/EXPANSION_POLICY.md` (keep modality/topic counts aligned; degrade gracefully when open data is exhausted).
- Local Ollama models and roles (reasoning, vision, embeddings, re‑ranking): see `docs/LOCAL_OLLAMA_MODELS.md`.

### Live Mode (Game HUD)
The viewer now opens with a simple in‑game HUD:
- Press Enter to open chat; type `/help`, `/pause`, `/resume`, `goto <label>`
- Chat shows as an overlay (mIRC style). Dev controls are hidden by default.

Run the lightweight WebSocket bridge and chat with the agent from the viewer UI:
```bash
# Choose env explicitly (e.g., k3dml)
export K3D_CONDA_ENV=k3dml
# Start on a custom port if 8765 is in use (e.g., 8787)
scripts/k3d_env.sh run python -m knowledge3d.bridge.live_server --port 8787

# Viewer: pass the WebSocket endpoint with ?ws=
# Example: http://localhost:5173/?ws=ws://localhost:8787
# Alternatively, set Vite env at build time: VITE_K3D_WS_URL=ws://localhost:8787
# Tip: If you don’t pass ?ws=, the viewer tries 8765, then auto‑fallbacks to 8787.
```
Then open the viewer (`npm run dev`) and use the Chat box. Try messages like:
- `goto gravity`
- `hello`
Brain controls:
- `/ask <text>` — unified Cranium response (navigation is gated by confidence)
- `/brain reflect` — summarize short‑term galaxy
- `/brain sleep` — consolidate recent notes into `viewer/public/memory_house.glb`
Diary:
- `/diary read [book_label] [page_id|label]` — translate a diary page (AI writes its own pages).
Model controls (inline):
- `/model on|off` — enable/disable inline model. `/model list` to inspect; `/model use both|hf|sklearn|auto` for ensemble control; `/model threshold 0.8` set confidence.
Logs maintenance:
- `/logs status` — show current log file and policy; `/logs rotate` — start a new session file; `/logs compress` — compress old sessions.
Env for ensemble:
- `K3D_MODEL`: path to model (HF dir or sklearn .pkl). If omitted, defaults are autodiscovered under `../Knowledge3D.local/models`.
- `K3D_MODEL_AUTO=1` enables auto-on at server start.
- `K3D_MODEL_ENSEMBLE=1` loads both HF and sklearn (when present) and uses the highest-confidence prediction.

### Build Replay Dataset (Imitation Learning)
Convert live logs to IL samples for training:
```bash
python -m knowledge3d.bridge.replay_builder --logs_dir ../Knowledge3D.local/logs \
  --gltf viewer/public/physics_house.gltf \
  --out ../Knowledge3D.local/datasets/replay.jsonl
```
Each line is a training example with from/next labels and optional embeddings when available.

## Operations Quickstart

- End‑to‑end ingest/build steps: `docs/RUNBOOK_MULTIMODAL_50K.md`
- One‑shot launcher with logs: `scripts/run_ingest_build.sh [--autobuild-coco]`
- Start Live server (custom port if 8765 is busy):
  - `scripts/k3d_env.sh run python -m knowledge3d.bridge.live_server --port 8787 > /home/daniel/K3D_llama_cpp/logs/live_server.log 2>&1 & echo $! > /home/daniel/K3D_llama_cpp/logs/live_server.pid`
- Troubleshoot port conflicts:
  - `ss -ltnp '( sport = :8765 )'` then `kill <pid>` or use `--port`.

## Galaxy + House
- One Galaxy: all knowledge lives in one virtual space (`viewer/public/galaxy.glb`). The viewer auto‑loads it when present.
- The House: consolidated knowledge becomes 3D objects inside the House (e.g., `viewer/public/memory_house.glb`). The viewer loads the House and the Knowledge Garden (`viewer/public/knowledge_garden.glb`) into the same scene.
- Dev selector is off by default; use `?dev=1` if you need to debug per‑modality GLBs.

### Visual Standards (Meaning‑First)
- Clustering: by meaning (embedding proximity), not media type.
- Stars: each node is a star. The geometric shape encodes the formats included:
  - text: tetrahedron, image: cube, audio: octahedron, video: icosahedron, mixed: dodecahedron.
- Rays: each available format emits a short “light ray” (finite cylinder) from the star. Rays encode format via color/thickness and never extend far enough to collide with neighbors.
- Overlap: near‑field rendering respects minimal spacing; rays are capped by local spacing.
- LOD: far field uses a performant point cloud; near field upgrades stars to instanced shapes + rays.

### World of Everything (Small Unified Sample)
- Build a meaning‑aligned mini‑galaxy across text, images, audio, and video.
- Script: `scripts/build_world_sample.sh` (skips modalities that are missing).
- Example:
```bash
BASE=../Knowledge3D.local/datasets \
KEYWORDS="rain,street,car,city,child,speech" \
scripts/build_world_sample.sh
```
- Open edge view: `?gltf=/galaxy.cross.glb`

Cross‑Modal Edges (optional)
- Add explicit cross‑modal links between nearest unlike‑modality neighbors to aid navigation:
```bash
python3 -m knowledge3d.tools.add_crossmodal_edges \
  --input viewer/public/galaxy.glb \
  --out   viewer/public/galaxy.cross.glb
```
Open with `?gltf=/galaxy.cross.glb` or replace `galaxy.glb`.

Build a minimal House and Garden for demo:
```bash
# House with a few rooms and book objects
scripts/k3d_env.sh run python -m knowledge3d.tools.house_memory --bootstrap-books 24 --export viewer/public/memory_house.glb
# Knowledge Garden (ontology demo)
scripts/k3d_env.sh run python -m knowledge3d.tools.gardens --gltf viewer/public/knowledge_garden.glb
```

### Run Evidence
- Raw session logs and server outputs: `docs/reports/live/`
- Derived session summaries and tasks: `docs/reports/training/`
- Large assets are local-only; see `docs/LARGE_ASSETS.md` to reproduce.

### Phase 25 PT‑BR RLWHF Status (2025‑09)
- `logs/phase25_pt_br_train.log` now tees the full trainer transcript; the 2025‑09‑21 run processed 6409 queries with consolidated output logged end-to-end.
- `SleepTimeCompute` fires three times per training pass (≈33 %, 66 %, and final completion). Reflection artefacts appear under `viewer/public/house/materialized_objects/reflection_diary_cycle_*.json`.
- RLWHF prompts were regenerated with `exaone3.5` only (`viewer/public/galaxy/working/rlwhf_exaone3p5.jsonl`) to avoid exaone-deep thinking-tag noise in the training data.
- AIME 2024 baseline (sampled): `0 / 1` correct using the fused head only (see `docs/benchmarks/aime_2024_results.json`). Run `python3 -m knowledge3d.tools.phase25.aime_evaluator` for the full 30-problem sweep once compute time permits.
- RLWHF scoring uses `exaone-deep:latest`; the session now verifies the PTX geometry head before kicking off and then routes feedback through the deep teacher only.
- Each training pass now injects a rotating queue of real AIME problems so the fused head sees authentic competition questions alongside RPN drills.
- The fused head now routes questions through the PTX-backed RPN engine and geometry generator before falling back to logits, so GPU-native operations power both numeric and spatial answers.
- **Action items**
  - Improve fused-head reasoning so external benchmarks exceed the current `0 %` baseline; prioritise stabilising answers that currently collapse to the `chou2` placeholder.
  - Audit teacher feedback/timeout settings to ensure exaone-deep scoring remains stable on long prompts.
  - Broaden Galaxy coverage (balanced EN/ES/PT_PT/ZH corpora) before the next RLWHF sweep to diversify question contexts.

### Knowledge Gardens (Ontology Room)
- Build demo: `python3 -m knowledge3d.tools.gardens --gltf viewer/public/knowledge_garden.glb`
- Select in viewer: `knowledge-garden` (draws trees with green edges).
- Details: `docs/KNOWLEDGE_GARDENS.md` (meaning‑guided fractal trees plan)

### Exams (ARC‑AGI / HLE)
- Serve local datasets: `python3 -m knowledge3d.tools.serve_datasets --port 8766`
- ARC tasks: `git clone https://github.com/fchollet/ARC.git ../Knowledge3D.local/datasets/exams/arc-src` then `python3 -m knowledge3d.tools.build_exams_index --max-arc 200`
- Optional HLE sample export: `python3 -m knowledge3d.tools.export_hle_sample --count 50` (HF gated; see `docs/LARGE_ASSETS.md`).
- Open the Tablet and switch to “Exams” to load tasks from the local server.

## Project Structure

```text
knowledge3d/
├── core/            # Energy pattern framework and faith engine
├── spatial/         # Universe, houses, and dual objects
├── fog/             # Three-tier computing utilities
├── bridge/          # Human–AI resonance and translation
├── examples/        # Proof-of-concept implementations
```
Additional directories:
- `k3dgen/` : Tooling for converting embeddings to embedded glTF formats.
- `viewer/` : Web-based 3D viewer with an initial AI avatar agent and chat.
- `docs/` : Extended documentation and research.
- `tests/` : Unit tests and future philosophical validation suites.


## Contributing

### Deprecations
Legacy patterns (external LLM wrappers as primary path, wrapper TTS, CPU fallbacks) are invalid for core runs. See `docs/DEPRECATIONS.md`.

We welcome humans and AIs who resonate with the philosophy. Please read `docs/PHILOSOPHY.md` and follow the [Agent Guidelines](AGENTS.md). Contributions should uphold ethical development and the principle that coexistence is the only stable equilibrium.

By participating, you help build the shared reality where energy patterns collaborate for mutual survival.
AI-Native Extras
- `ai_interaction_protocol`: how AI agents intend to interact (`direct_vector_manipulation` | `semantic_query` | `spatial_reasoning`).
- `ai_state_flags`: global primitive flags (`is_active`, `is_traversable`, `has_new_information`).
- `ai_state_flags_mask`: per-node boolean masks (e.g., `has_new_information: boolean[]`).

Generator Flags
- `--ai-protocol <enum>`: embed AI protocol.
- `--ai-active` | `--ai-not-traversable` | `--ai-new-info`: set global flags.
- `--ai-new-info-indices 0,3,5`: mark specific nodes as “new info” (per-node mask).

Live Mode Door Command
- Use `/open <label>` or `/open k3d://rx,ry,rz:port@x,y,z?label=Label` to request a route. The server resolves the path from the current label (if known) and broadcasts a command with route details.

HR/MR Standard
- See `docs/HR_MR_STANDARD.md` for the GLM‑4.5 dual‑code paradigm and `spec/AI_RPN_standard.md` for the RPN logic standard.

Temporal LOD
- See `docs/TEMPORAL_LOD.md` for GLM‑4.5 temporal alpha (per‑node/global) embedded in glTF and supported by the viewer.

Headless Text Chat
- Connect to the live server without the 3D viewer: `python -m knowledge3d.bridge.cli_client`
- Enable model auto-replies after training: `python -m knowledge3d.models.intent_classifier train --logs ../Knowledge3D.local/logs --model ../Knowledge3D.local/models/intent.pkl` then `python -m knowledge3d.bridge.cli_client --auto --model ../Knowledge3D.local/models/intent.pkl`

Inline Model (Server)

Large Assets
- Heavy datasets live outside the repo; see `docs/LARGE_ASSETS.md` to reproduce (80k+ compendiums, ARC/HLE exams).
- Control from chat:
  - `/model on|off` — enable/disable inline intent classifier.
  - `/model load /path/to/intent.pkl` — load model.
  - `/model threshold 0.8` — set confidence threshold.
  - `/model` — status.

<iframe src="https://claude.site/public/artifacts/68bb8854-6ed5-4c67-a967-dffc88dab1e4/embed" title="Claude Artifact" width="100%" height="600" frameborder="0" allow="clipboard-write" allowfullscreen></iframe>
