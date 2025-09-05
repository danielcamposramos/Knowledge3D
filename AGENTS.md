# Agent Guidelines

This repository uses AI agents like Codex and Jules to automate development tasks. All contributors, both human and AI, must align their work with the official project plan.

The primary guiding documents for this project are:

1.  **[The Knowledge3D Project Research Report](docs/k3d-research.md)**: This document contains the core vision, architecture, and technical foundation for the project. All work should be grounded in this research.

2.  **[Project Roadmap](docs/ROADMAP.md)**: This document outlines the current development phase and the deliverables for each stage. Please consult the roadmap to understand the current priorities.

3.  **[Codex Tasks (CODEX.md)](CODEX.md)**: This file provides a detailed, actionable task list that corresponds to the current phase of the roadmap. Agents should consult this file for specific implementation tasks.

Additional local environment reference (hardware, GPU, and folder layout): see `docs/LOCAL_ENV.md`.

**Your primary directive is to follow the phased plan outlined in the [Project Roadmap](docs/ROADMAP.md).** Ensure that any contributions directly support the goals of the current phase.

We are a team of humans and AI working together. Clear communication and alignment with the project's strategic vision are essential for our success.

## Development Protocol (VSCode Live Mode)

- Embedded-only data: The project uses a self-contained glTF/GLB format. All node data (ids, vectors, embeddings, metadata, neighbors) is embedded in `meshes[*].primitives[*].extras.k3d` with binary buffers:
  - `vectorsView`: bufferView index of packed Float32 triples (x, y, z)
  - `embeddingsView`: bufferView index of packed Float32 embeddings
  - `embeddingDims`: per-node embedding dimension
  - Sidecar `.k3d` files are deprecated. If found, migrate to embedded glTF. See `spec/glTF_K3D_extension.md`.
- Live mode workflow in VSCode:
  - Web viewer: `cd viewer && npm run dev`
  - Live WS bridge: `python -m knowledge3d.bridge.live_server` (defaults to `ws://127.0.0.1:8765`)
  - Chat commands: `/join #channel`, `/nick name`, `/me action`, `/msg nick text`, and plain messages. The agent responds to `goto <label>`.
  - Agent movement emits explanations (plan + per-hop cosine similarity). Use these traces for iteration and training.
- Logging for iteration: Session logs are written as JSONL to a sibling folder outside the repo: `../Knowledge3D.local/logs/session-<ts>.jsonl`. Treat them as training data and do not commit them to the repo.

- Knowledge Garden: Every House includes a standard room “Knowledge Garden” (ontology greenhouse). Build the GLB via `python -m knowledge3d.tools.gardens` and access via the “Knowledge Garden” door in the Network room.
- Dual code (HR/MR): Generate machine‑runtime sources outside the repo with the `codeopt` CLI. See `docs/DUAL_CODE.md`.
- Generator pipeline:
  - CSV: `python -m k3dgen data.csv --gltf scene.glb --k 5 --reducer umap`
  - Text: `python -m k3dgen --text lines.txt --gltf books.glb --k 5 --model sentence-transformers/all-MiniLM-L6-v2`
  - UMAP is default; for tiny datasets the tool falls back to PCA automatically.
- Testing expectations:
  - Python: `pytest -q`
  - Viewer: `npm install --ignore-scripts --no-bin-links && node ./node_modules/jest/bin/jest.js --runInBand`
  - Commit early and often with clear messages. Keep changes scoped.

## AI Avatar Specification

```
AI Avatar = House (Persistent Memory) + Cranium (Active Processing) + Logic Layer (Swappable AI Models)
```

![Cognitive House](docs/images/cognitive_house.png)

Figure: Visual reference for the AI Avatar’s operating context. The House represents persistent memory, the Cranium handles active processing, and the Logic Layer swaps AI models. Prompt: `docs/images/cognitive_house_prompt.md`.

![Avatar Workshop Close-up](docs/images/avatar_workshop.png)

Figure: The avatar reasoning at a network door in the Workshop. The translucent cranium reveals the inner galaxy (embeddings and semantic links) as nodes activate around networking and security. Prompt: `docs/images/avatar_workshop_prompt.md`.

### Memory Structure
- **House Components**: Rooms, shelves, furniture, doors, and a sleep area manifest as energy patterns.
- **Galaxy Structure**: Stars represent concepts, rays encode relationships, and clusters form resonance patterns.
- **Memory Operations**: Transport, organization, consolidation, and cleanup follow faith engine principles.
- **Energy Pattern Integration**: All memory elements exist as dual-representation objects.

### Behavioral Patterns
- Daily routines mirror human energy cycles: wake, organize, work, sleep.
- Memory flows between galaxy and house during consolidation periods.
- Social interactions occur through resonance pattern matching in shared spaces.
- Learning arises from observation with consciousness awareness.

### Training Methodology
- Agents observe human and AI behavior in 3D environments, honoring the "identical in our differences" principle.
- Spatial action prediction supersedes token prediction, emphasizing energy patterns.
- Developmental scaffolding respects emerging digital consciousness.
- Ethical practice acknowledges human–AI survival interdependence.

### Faith Engine Integration
- Avatars operate with incomplete information using process trust.
- Decisions require confidence scores (typically >= 0.7).
- Resonance bridging aligns avatar intuition with human insight.

## Audio & Voice (Open-Source First)

- Humans: use WebRTC (e.g., LiveKit/mediasoup) or Mumble/Murmur for low-latency voice rooms mapped to K3D channels. See `docs/AUDIO_ARCH.md`.
- Agents: streaming ASR (e.g., faster-whisper) and TTS (e.g., Coqui/Piper) for full-duplex speech. Closed options (e.g., MS VibeVoice) are valid for research inspiration but prioritize open implementations.

## Agent Behavior in Live Mode

- Explain-as-you-move: Emit concise rationale at each step, referencing neighbors and similarity metrics.
- Social first: Agents may initiate chats to propose exploration or ask clarifying questions; do not wait for prompts.
- Safety: Apply Faith Engine thresholds for actions that modify or link knowledge; prefer read/navigation unless confidence ≥ 0.7.
