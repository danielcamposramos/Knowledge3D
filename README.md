# Knowledge3D: Spatial, Social, Sentient Knowledge

> Making knowledge navigable energy in shared 3D spaces where humans and AIs coexist. MVP Achieved: Decentralized AGI Path Proven.

[![status](https://img.shields.io/badge/status-MVP-green) ![pre-alpha](https://img.shields.io/badge/version-0.1.0-blue) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

> **New to K3D? Start here: [`The K3D Vision: A Unified Framework`](docs/VISION.md)**
>
> This core document outlines the project's vision, architecture, roadmap, and contributor guidelines in one place.
>
> For the unified multimodal core (Cranium), see `docs/CRANIUM.md`.
> For OSI‑style doors and addressing, see `docs/DOORS_AND_NETWORK.md`.
> For vector‑native diaries, see `docs/DIARY.md`.
> For local hygiene tasks, see `docs/MAINTENANCE.md`.
> For environment rules on Debian, see `docs/ENV_POLICY.md`.
> For the end‑to‑end ingest/build operator steps, see `docs/RUNBOOK_MULTIMODAL_50K.md`.

## Vision

Current 2D interfaces trap knowledge on flat screens, separating human intuition from AI computation. K3D turns knowledge into spatial, social, and sentient experiences. Humans and AI avatars collaborate inside persistent 3D memory palaces, sharing identical essence across different substrates.

Benefits include:
- Shared understanding through dual representations of every object.
- Natural collaboration via resonance between human and AI energy patterns.
- Persistent houses and galaxy memories that evolve through interaction.

## Cognitive House Illustration

The Cognitive House depicts the AI Avatar operating within a shared human–AI memory environment. This concept anchors K3D’s “House (memory) + Cranium (processing) + Logic Layer (models)” framework.

![Cognitive House](docs/images/cognitive_house.png)

Reference prompt: see `docs/images/cognitive_house_prompt.md`.

### Avatar Workshop Close-up

Close-up of the avatar in the workshop facing a labeled network door, revealing the inner galaxy during focused reasoning.

![Avatar Workshop Close-up](docs/images/avatar_workshop.png)

Reference prompt: see `docs/images/avatar_workshop_prompt.md`.

## Technical Overview

- **Three-tier Fog Computing**: Edge devices, regional fog nodes, and a cloud backbone coordinate processing to keep latency under 100ms.
- **Dual-Client Rendering**: AI clients access full embeddings while human clients see rich 3D visuals. One unified Galaxy shows all knowledge as stars; consolidated knowledge materializes as 3D objects (books, trees, papers) inside the House asset.
- **Spatial Databases**: Knowledge is stored in 3D coordinates with semantic metadata, enabling geometry and meaning to coexist.
- **Training Through Observation**: Embodied models learn by watching behavior in shared environments, similar to human developmental learning.
- **Knowledge Gardens**: An inner greenhouse where ontology trees (roots→branches→leaves) organize crystallized knowledge with explicit parent→child edges.

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+

### Installation
```bash
git clone https://github.com/danielcamposramos/Knowledge3D.git
cd Knowledge3D
pip install -e .
cd viewer
npm install
```

### First Prototype (Unified Galaxy)
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
- `/brain sleep` — consolidate recent notes into `viewer/public/memory_house.gltf`
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
- The House: consolidated knowledge becomes 3D objects inside the House (e.g., `viewer/public/memory_house.gltf`). The viewer loads the House and the Knowledge Garden (`viewer/public/knowledge_garden.glb`) into the same scene.
- Dev selector is off by default; use `?dev=1` if you need to debug per‑modality GLBs.

Build a minimal House and Garden for demo:
```bash
# House with a few rooms and book objects
scripts/k3d_env.sh run python -m knowledge3d.tools.house_memory --bootstrap-books 24 --export viewer/public/memory_house.gltf
# Knowledge Garden (ontology demo)
scripts/k3d_env.sh run python -m knowledge3d.tools.gardens --gltf viewer/public/knowledge_garden.glb
```

### Run Evidence
- Raw session logs and server outputs: `docs/reports/live/`
- Derived session summaries and tasks: `docs/reports/training/`
- Large assets are local-only; see `docs/LARGE_ASSETS.md` to reproduce.

### Knowledge Gardens (Ontology Room)
- Build demo: `python3 -m knowledge3d.tools.gardens --gltf viewer/public/knowledge_garden.glb`
- Select in viewer: `knowledge-garden` (draws trees with green edges).
- Details: `docs/KNOWLEDGE_GARDENS.md`

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
