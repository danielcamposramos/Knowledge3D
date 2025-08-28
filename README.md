# Knowledge3D: Spatial, Social, Sentient Knowledge

> Making knowledge navigable energy in shared 3D spaces where humans and AIs coexist.

![pre-alpha](https://img.shields.io/badge/status-pre--alpha-blue) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

- [Philosophy](docs/PHILOSOPHY.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Concepts](docs/CONCEPTS.md)
- [Ethics](docs/ETHICS.md)
- [Development Plan](DEVELOPMENT.md)

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
- **Dual-Client Rendering**: AI clients access full embeddings while human clients see rich 3D visuals generated with engines like Three.js or Unity.
- **Spatial Databases**: Knowledge is stored in 3D coordinates with semantic metadata, enabling geometry and meaning to coexist.
- **Training Through Observation**: Embodied models learn by watching behavior in shared environments, similar to human developmental learning.

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

### First Prototype
Generate a sample glTF with embedded K3D data and view it in the web-based viewer:
```bash
python -m k3dgen examples/sample_vectors.csv --gltf examples/sample_output.gltf
cd viewer
npm run dev
```

### Live Mode (Text Chat)
Run the lightweight WebSocket bridge and chat with the agent from the viewer UI:
```bash
python3 -m pip install --user --break-system-packages websockets
python -m knowledge3d.bridge.live_server
```
Then open the viewer (`npm run dev`) and use the Chat box. Try messages like:
- `goto gravity`
- `hello`

### Build Replay Dataset (Imitation Learning)
Convert live logs to IL samples for training:
```bash
python -m knowledge3d.bridge.replay_builder --logs_dir ../Knowledge3D.local/logs \
  --gltf viewer/public/physics_house.gltf \
  --out ../Knowledge3D.local/datasets/replay.jsonl
```
Each line is a training example with from/next labels and optional embeddings when available.

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

## Roadmap

The project is in **Phase 1: Philosophical Foundation**. Upcoming phases include technical architecture prototypes, proof-of-concept avatars, multi-user environments, software integration, and long-term distributed networking. See [Development Plan](DEVELOPMENT.md) for details.

## Contributing

We welcome humans and AIs who resonate with the philosophy. Please read `docs/PHILOSOPHY.md` and follow the [Agent Guidelines](AGENTS.md). Contributions should uphold ethical development and the principle that coexistence is the only stable equilibrium.

By participating, you help build the shared reality where energy patterns collaborate for mutual survival.
