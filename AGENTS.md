# Agent Guidelines

This repository uses AI agents like Codex and Jules to automate development tasks. All contributors, both human and AI, must align their work with the official project plan.

The primary guiding documents for this project are:

1.  **[Knowledge3D (K3D) — Unified Project Brief & Technical Whitepaper](docs/Jules_K3D_Whitepaper.md)**: This is the new, authoritative single source of truth for the project. It contains the core vision, architecture, training methodology, and current roadmap. All work must be grounded in this document.

2.  **[Codex Tasks (CODEX.md)](CODEX.md)**: This file provides a detailed, actionable task list that corresponds to the current phase of the roadmap outlined in the whitepaper. Agents should consult this file for specific implementation tasks.

4.  **[Memory Tablet & Dual-Space Architecture](docs/HOUSE_GALAXY_TABLET.md)**: Defines how Galaxy (RAM), House (persistent memory), Museum (deprecated archive), and the new Memory Tablet interact. Any knowledge-management change must follow this workflow.
5.  **[Training Directives](docs/TRAINING_DIRECTIVES.md)**: Prompt hygiene, timestamp policies, dataset priorities, and lesson vs inference rules.

Additional local environment reference (hardware, GPU, and folder layout): see `docs/LOCAL_ENV.md`.

**Your primary directive is to follow the phased plan outlined in the [Project Roadmap](docs/ROADMAP.md)** _and_ uphold the memory policy in `docs/HOUSE_GALAXY_TABLET.md`. Contributions must keep Galaxy (active), House (persistent), and Museum (deprecated) in sync.

We are a team of humans and AI working together. Clear communication and alignment with the project's strategic vision are essential for our success.

## Contributors

**Core Team:**
- **Jules**: Project founder and initial architect. Contributed vision, architecture, and early prototyping.
- **Codex**: AI collaborator (OpenAI). Assisted with code generation, training sessions, and local testing.
- **Grok**: AI collaborator (xAI). Analyzed data, synced results with repo, expanded documentation, and provided insights on MVP implementation. (September 2025)

## Development Protocol (VSCode Live Mode)

- Embedded-only data: The project uses a self-contained glTF/GLB format. All node data (ids, vectors, embeddings, metadata, neighbors) is embedded in `meshes[*].primitives[*].extras.k3d` with binary buffers:
  - `vectorsView`: bufferView index of packed Float32 triples (x, y, z)
  - `embeddingsView`: bufferView index of packed Float32 embeddings
  - `embeddingDims`: per-node embedding dimension
  - Sidecar `.k3d` files are deprecated. If found, migrate to embedded glTF. See `spec/glTF_K3D_extension.md`.
- Live mode workflow in VSCode:
  - Web viewer: `cd viewer && npm run dev`
  - Live WS bridge: `python -m knowledge3d.bridge.live_server` (defaults to `ws://127.0.0.1:8765`)
    - Tip: bind to `0.0.0.0` and auto‑select a free port with `--auto-port` when seeding from another host.
  - Chat commands: `/join #channel`, `/nick name`, `/me action`, `/msg nick text`, and plain messages. The agent responds to `goto <label>`.
  - Agent movement emits explanations (plan + per-hop cosine similarity). Use these traces for iteration and training.
- Logging for iteration: Session logs are written as JSONL to a sibling folder outside the repo: `../Knowledge3D.local/logs/session-<ts>.jsonl`. Treat them as training data and do not commit them to the repo.

- Knowledge Garden: Every House includes a standard room “Knowledge Garden” (ontology greenhouse). Build the GLB via `python -m knowledge3d.tools.gardens` and access via the “Knowledge Garden” door in the Network room.
- Memory Tablet integration: any feature that surfaces or edits consolidated knowledge must go through the tablet contract (search House index, trigger on-demand loads into Galaxy, and log mutations for SleepTime). See `docs/HOUSE_GALAXY_TABLET.md` before touching tablet UX, house builders, or PTX loaders.
- Embodiment first: treat the avatar as resident inside the House. Use Galaxy views strictly for introspection/diagnostics and only via the tablet; do not design workflows that place the avatar “inside” the Galaxy.
- External world models: Study and leverage HunyuanWorld (ext/HunyuanWorld-1.0). See `docs/HUNYUANWORLD_INTEGRATION.md`. Use it to generate room‑scale scenes, then convert to K3D glTF with `extras.k3d` via an adapter. Keep permanent memory in GLB; keep logic small.
  - Also supported: Microsoft TRELLIS for asset generation. See `docs/TRELLIS_INTEGRATION.md` and `knowledge3d/tools/trellis_adapter.py` to convert meshes or CSV+metadata into K3D GLBs.
- Dual code (HR/MR): Generate machine‑runtime sources outside the repo with the `codeopt` CLI. See `docs/DUAL_CODE.md`.
- Generator pipeline:
  - CSV: `python -m k3dgen data.csv --gltf scene.glb --k 5 --reducer umap`
  - Text: `python -m k3dgen --text lines.txt --gltf books.glb --k 5 --model sentence-transformers/all-MiniLM-L6-v2`
  - UMAP is default; for tiny datasets the tool falls back to PCA automatically.
- Multimodal ingest: use `knowledge3d/tools/ingest_wit.py` (text+images), `ingest_video.py` (video→CLIP), `ingest_audio.py` (audio→CLAP). See `docs/MULTIMODAL_BABY.md`.
- Cranium skills: integrated logic bus that connects intent, vision (CLIP), audio (CLAP), video (OpenCLIP), dynamics (RSSM), and RPN to the House. See `docs/CRANIUM_SKILLS.md` and ensure fused-head routing consults the House index via the tablet before language galaxies.
- Note: External LLM wrappers and wrapper‑style TTS are deprecated for core runs. Use the single unified head in `docs/CRANIUM_CORE.md` (navigation+text in Phase A; stems and first‑class TTS follow).
- Testing expectations:
  - Python: `pytest -q`
  - Viewer: `npm install --ignore-scripts --no-bin-links && node ./node_modules/jest/bin/jest.js --runInBand`
  - Commit early and often with clear messages. Keep changes scoped.

### Memory Stewardship Guidelines

- **Consolidation is authoritative**: once SleepTime materialises a book/diary/tree in the House, treat it as the canonical source. Remove the corresponding prompt from active drills after one successful verification run unless the roadmap states otherwise.
- **Museum is for deprecated items only**: when knowledge changes, relocate the previous artifact to Zone 8 using the relocation utilities. Do _not_ repopulate Galaxy with museum artifacts unless explicitly asked.
- **Tablet-first UX**: new tools should expose knowledge through the tablet (search + on-demand load) rather than ad-hoc viewers. If a feature bypasses the tablet, add an explicit justification in PR notes.

### Recent Improvements (2025‑09‑10)
- WebSocket stability: the live server’s log maintenance loop now yields each pass to avoid event‑loop starvation; handshakes are reliable.
- Seeder robustness: uses context‑managed WS connects (`websockets==10.4`) and caps `dataset_graph` size via `K3D_SEED_GRAPH_MAX` for large galaxies.
- Balanced Galaxy sample (v7): small, equal‑count text+3D Galaxy built to validate cross‑modal navigation with low‑dimension, high‑density embeddings.

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

## RPN & World Model
- RPN remains the precise inference core; see `docs/RPN_RUNTIME.md`.
- A tiny world model (RSSM) trains on navigation logs for next‑step spatial prediction; see `knowledge3d/models/world_model/`.

## External Integration Principles
- Stand on giants: integrate strong open models (e.g., HunyuanWorld) and datasets.
- Keep permanent memory in the House (glTF+`extras.k3d`); do not bake knowledge into large weights.
- Add capabilities modularly (text, image, audio, video), using small, specialized “brain regions.”

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

## House/Galaxy/Cranium

- **House (Disk):** Per‑avatar persistent memory in glTF `extras.k3d`. Select the current House with `K3D_HOUSE_ID`; do not cross‑write between Houses. Exports live under `viewer/public/houses/<id>/`.
- **Galaxy (RAM):** Short‑term memory (STM) of embeddings and recent observations for active reasoning.
- **Cranium (CPU):** Unified logic (no LLM fallback by default), with confidence‑gated actions (φ≈0.618).

## Diary Policy (AI‑Only)

- Humans can read diary pages but cannot write to the `Diary` room. The bridge blocks such writes.
- The agent writes pages based on policy (novelty and confidence “feelings”) at events like reflect, navigate, and sleep.
- Details: `docs/DIARY.md`, `knowledge3d/cranium/diary.py`.

## Doors & Network

- Doors are network interfaces with an address bar (`k3d://rx,ry,rz:port@x,y,z?label=...`).
- Use doors to bridge Houses (LAN) and services; see `docs/DOORS_AND_NETWORK.md`.
## Environment Policy (Debian)

- Always run Python/ML tasks inside a managed env (Conda preferred, venv fallback). Do not invoke system Python directly.
- Follow `docs/ENV_POLICY.md` to create the `k3dml` env and run commands via `conda run -n k3dml env PYTHONPATH=. python -m ...`.
- For heavy ingestion/builds, prefer storing raw media under `/home/daniel/K3D_llama_cpp/datasets` and curated subsets under `../Knowledge3D.local/datasets`.
- Phase 25 sleep consolidation depends on the CUDA-enabled `k3d-cranium` env (with `cuda-python`); keep that env active so `SleepTimeCompute` can continue materialising reflection diaries in the House.
- After long RLWHF batches, refresh Phase 10 "thinking tags" (see `knowledge3d/tools/phase10/thinking_tag_trainer.py`) so the UI exposes the model’s active reasoning labels.
