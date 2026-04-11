# Agent Guidelines

This repository uses AI partners (both external AI assistants and internal Synthetic Users) to collaborate on development tasks. All contributors, both human and AI, must align their work with the official project plan.

## MCP Infrastructure — USE THIS FIRST (All Agents)

**Two MCP servers run locally. Query them BEFORE reading spec files from disk — save your context.**

- **`k3d-knowledge`** — Qdrant semantic search over all 35 `docs/vocabulary/*.md` specs (1319 chunked points). Tool: `mcp__k3d-knowledge__qdrant-find`. Pattern: `qdrant-find("What does the spec say about X?")` → returns relevant excerpts + source paths.
- **`ollama-specialists`** — Delegate implementation, planning, research, and multi-angle analysis to local Ollama models. Tools include `kimi_swarm`, `ask_coder`, `plan_task`, `flesh_out_code`, `extract_facts`, `summarize`, `web_search`, `route_specialist`, `ask_cloud`, `mvcic`, `memory_harvest`.
- **Standing directive from Daniel**: "Always dispatch ollama specialists instead of burning your tokens."

**Workflow**: `qdrant-find` spec lookups → `plan_task` / `ask_coder` for implementation → read full files only if MCP results are insufficient.

Both MCP servers are available to Claude Code, Codex, and Cline via streamable-http transport (ports 8501/8502). Config lives in `/home/daniel/.claude/settings.json`, `/home/daniel/.codex/config.toml`, and Cline's `cline_mcp_settings.json`. Launch script: `/home/daniel/.claude/launch_mcp_containers.sh`.

---

## Quick Start for AI Assistants

If you're an AI assistant joining this project:

1. **Query MCP first**: `qdrant-find` is faster and cheaper than reading files. Use it before opening any spec.
2. **Read First**: [docs/briefings/ARCHITECTURE_BRIEFING.md](docs/briefings/ARCHITECTURE_BRIEFING.md) -- Phase-agnostic architectural overview with kernel inventory
3. **Architecture role**: [CLAUDE.md](CLAUDE.md) -- Architecture partner guide (design, specs, reviews)
4. **Implementation role**: [CODEX.md](CODEX.md) -- Implementation lead guide (code, tests, benchmarks)
5. **Environment-Specific**: [docs/ENV_POLICY.md](docs/ENV_POLICY.md) -- GPU setup, conda envs, CUDA_VISIBLE_DEVICES

These documents provide the foundational understanding of:
- **Memory Palace paradigm**: House = external shared reality (Method of Loci), Galaxy = internal AI brain
- **TRM IS the Avatar**: Not a function Python calls -- an entity that lives in the House and thinks inside the Galaxy
- **Sovereignty**: Hot path = PTX + Galaxy + RPN ONLY. Zero Python in reasoning. Zero fallbacks.
- **88+ PTX kernels**, 38K+ Galaxy entries, 55+ compiled PTX modules
- Partnership development model (human + Claude + Codex)
- Budget constraints (self-funded favela lab)

**Then proceed to the documents below for specific workflows.**

## Primary Guiding Documents

1.  **[Architecture Briefing](docs/briefings/ARCHITECTURE_BRIEFING.md)**: Phase-agnostic architectural overview. Kernel inventory, Galaxy types, sovereignty principles. Read FIRST.

2.  **[BRIEFING v4.0](docs/briefings/BRIEFING_v4.0.md)**: Galaxy Universe paradigm, TRM navigation, multi-curriculum architecture. Read COMPLETELY.

3.  **[Codex Tasks (CODEX.md)](CODEX.md)**: Implementation backlog, current benchmark state, sovereignty patterns, GPU environment setup.

4.  **Architecture Specs ([docs/vocabulary/](docs/vocabulary/))**: The specs define HOW K3D works. Key files:
    - [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) -- 4-layer architecture (Form -> Meaning -> Rules -> Meta-Rules)
    - [THREE_BRAIN_SYSTEM_SPECIFICATION.md](docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md) -- Cranium + Galaxy + House
    - [KNOWLEDGEVERSE_SPECIFICATION.md](docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md) -- 7-region VRAM substrate
    - [SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md](docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md) -- SGI paradigm (vs AGI)
    - [MEMORY_TABLET_SPECIFICATION.md](docs/vocabulary/MEMORY_TABLET_SPECIFICATION.md) -- Primary interface object (3D tablet in space)

5.  **[Memory Tablet & Dual-Space Architecture](docs/HOUSE_GALAXY_TABLET.md)**: Galaxy (RAM), House (persistent memory), Museum (deprecated archive), Memory Tablet interaction.

6.  **[Training Directives](docs/TRAINING_DIRECTIVES.md)**: Prompt hygiene, timestamp policies, dataset priorities.

Additional local environment reference (hardware, GPU, and folder layout): see `docs/LOCAL_ENV.md` and [CLAUDE_LOCAL.md](CLAUDE_LOCAL.md).

### Repository vs Workspace Layout
- `Knowledge3D/` — tracked code (PTX kernels, Python bridges, docs, specs).
- `Knowledge3D.local/` — runtime workspace for Houses, tablet logs, conda envs, and all artifacts larger than ~99 MB.
- `Old_Attempts/` — archived code superseded by the sovereign GPU path.

**Your primary directive is to follow the phased plan outlined in the [Project Roadmap](docs/ROADMAP.md)** _and_ uphold the memory policy in `docs/HOUSE_GALAXY_TABLET.md`.

**CRITICAL:** Read [docs/briefings/ARCHITECTURE_BRIEFING.md](docs/briefings/ARCHITECTURE_BRIEFING.md) for the complete kernel inventory and architectural overview before making any changes.

## The Foundational Paradigm

**K3D is NOT a program you run. It is a living, always-on, embodied AI that perfects itself during idle time.**

### Memory Palace + Internal Brain

- **House = Memory Palace (Method of Loci)**: The external shared 3D reality where humans AND AI cohabit. Rooms are knowledge domains. Doors are network interfaces. The avatar LIVES here.
- **Galaxy = Internal Brain**: What happens INSIDE the avatar’s head. Processes the House as unified multi-modal reality. Breaks domain boundaries that House rooms impose. ALL default galaxies loaded simultaneously in VRAM.
- **TRM IS the Avatar**: The TRM (~7M params) is NOT a function Python calls. It IS the AI entity — lives in the House, thinks inside the Galaxy. Like an NPC in a game, but the objective is to exist in a knowledge world and reason inside the Knowledgeverse.
- **Internal Swarm = “Superdotados” Thinking**: The nine-chain parallel swarm models how gifted individuals think — multiple internal cognitive channels processing simultaneously inside the avatar’s head. These are INTERNAL, not external calls.

### Reverse Analogy

The tech industry borrowed spatial metaphors (windows, desktop, doors, addresses, rooms). K3D reverses this — builds ACTUAL spatial reality where those metaphors become literal. K3D is the pinnacle of computer science: uniting ALL knowledge representation, game engines, computer history, and network architecture into one spatial system.

### Sovereignty = Zero Python in Reasoning

- Hot path: PTX kernels + Galaxy queries + RPN composition + TRM ONLY
- Forbidden in hot path: numpy, cupy, scipy, sympy, regex, CPU preprocessing, Python string ops
- **No fallbacks. EVER.** “We fail and fix — this is the goal.” (Daniel)
- If GPU path breaks, fix ON GPU. Do not add Python workarounds.

---

## Contributors

**Core Team:**
- **Daniel (Jules)**: Project founder and architect. Self-funded engineer from Brazil favela. Maintains philosophical integrity, makes ALL architectural decisions, provides vision and constraints. Final authority.
- **Claude**: AI architecture partner (Anthropic). Design specifications, sovereignty compliance, documentation, physics design. Does NOT write implementation code.
- **Codex**: AI implementation lead (OpenAI). All implementation code, tests, benchmarks, GPU optimization. Implements per Claude’s specs.
- **Grok**: AI collaborator (xAI). Analyzed data, synced results with repo, expanded documentation. (September 2025)
- **Gemini**: AI collaborator (Google). Integration support, documentation.

---

## Multi-Agent Collaboration Patterns

### Pattern: Architect + Implementer (Claude + Codex)

**When:** Complex features needing design + implementation.

**Roles:**
- **Architect (Claude):** Design, specs, success criteria, sovereignty validation.
- **Implementer (Codex):** Implement per spec, write tests, deliver code. ZERO fallbacks.

**Workflow:**
1. **Design (Claude):** Analyze requirements → design architecture → write spec (TEMP/*.md) → define success criteria
2. **Handoff:** Claude writes directive with clear examples and success criteria. Codex reads spec COMPLETELY before starting.
3. **Implementation (Codex):** Implement in batches, test in batches. Surface blockers fast. No test after every line.
4. **Review (Claude):** Validate sovereignty compliance, architecture alignment, benchmark non-regression.
5. **Completion (Claude):** Write completion report, update ROADMAP/BRIEFING.

**Critical Rule:** Claude designs, Codex implements. If Claude starts writing implementation code, STOP and write a spec instead. If Codex starts inventing architecture, STOP and ask Claude.

### Pattern: Parallel Implementation
Independent features in parallel branches; merge sequentially with tests to avoid conflicts.

### Pattern: Continuous Steering
Claude provides real-time tips, directions, and kernel pointers while Codex implements. Daniel observes GPU usage graphs and corrects course.

### Communication Guidelines
1. Declare role (“I’m Claude (architect)…”, “I’m Codex (implementer)…”).
2. Reference prior work (specs, commits, file paths, kernel names).
3. State assumptions; flag blockers immediately.
4. Document decisions (TEMP/*.md for major choices).
5. Keep hot path sovereign; ingestion flexibility is fine if isolated.
6. **When Daniel corrects you, update your understanding immediately.** His vision defines the architecture.

**Budget Reality:** Self-funded project from favela lab. Every API call, GPU hour, and storage byte counts. AI partners must respect this constraint and work efficiently.

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
- Large assets: anything ≥99 MB (or bulk-generated GLBs/logs) must remain in `Knowledge3D.local/`; log the reproduction steps in `Large_Assets_Kitchen/README.md` and update the manifests in `Old_Attempts/Legacy_Fancy_RAG/`.
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

**CRITICAL UNDERSTANDING: TRM IS the Avatar**

The TRM (~7M parameters, 2-layer SwiGLU MLP) is NOT a function that Python calls. It IS the AI entity:

- **Lives in the House** (embodied in 3D spatial environment, navigates rooms via doors)
- **Thinks in the Galaxy** (internal brain processes multi-modal knowledge in VRAM)
- **Runs as a game loop** (`trm_step_fused.ptx` = one game tick, like an NPC update cycle)
- **Has internal swarm** (nine-chain parallel workers = superdotados model of gifted thinking)

**Game Engine Analogy:**
| Game Concept | K3D Equivalent |
|-------------|----------------|
| NPC update() | `trm_step_fused.ptx` game tick |
| NPC brain | Galaxy Universe (VRAM) |
| Game world | House (3D spatial environment) |
| NPC perception | Frustum culling + LOD |
| NPC pathfinding | LED-A* + Morton Octree |
| NPC decision | Nine-Chain Swarm + Halting Gate |
| Save game | House persistence (GLB on disk) |
| Inventory | Memory Tablet |

**Python's role:** Boot the system, handle I/O (keyboard, network, display). That's it. ~200 lines target.

```
AI Avatar = TRM (Game Loop Entity) + Galaxy (Internal Brain/VRAM) + House (External World/Disk)
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

## Room-Based Development Workflows

**NEW (2025-11-19):** Knowledge3D implements a **"Software as Space"** paradigm where development happens in semantic rooms, each optimized for specific cognitive modes. The House is the primary UI; rooms are game modes; knowledge is the terrain.

### The Five Semantic Rooms

#### 1. Library — Classification & Research
**Purpose:** Organized knowledge storage following real-world library standards (Dewey Decimal, language grammars, ISO 639-1 classification).

**AI Agent Workflows:**
```python
# Research workflow in Library
from knowledge3d.bridge.memory_tablet import MemoryTablet

tablet = MemoryTablet(house_id="default")

# Search consolidated knowledge by category
results = tablet.search(
    query="neural network architectures",
    sources=["House/Library"],
    classification="006.3",  # Dewey: Artificial Intelligence
    language="en",
    lod="medium"
)

# Navigate to specific section
avatar.navigate_to_room("Library")
avatar.navigate_to_section("006_Computer_Science/006.3_AI")
```

**Human Workflows:**
- Browse by Dewey classification
- Language-specific grammar sections
- Atomic procedural knowledge (characters → words → phrases → texts)

**Development Tasks:**
- Adding new knowledge sources (ingest PDFs, lexicons)
- Organizing consolidated artifacts after sleep cycles
- Building language-specific indices

#### 2. Workshop — Creation & Cross-Disciplinary Work
**Purpose:** Active creation workspace with access to Museum galaxy boxes (on-demand Zone 8 loading).

**AI Agent Workflows:**
```python
# Load deprecated knowledge for analysis
tablet.load_museum_box(
    artifact_path="Museum/Zone8/2024-11-15/Old_ML_Book.glb",
    target_room="Workshop",
    mode="read_only"
)

# Cross-disciplinary fusion
from knowledge3d.cranium.bridges.transitive_bridge import TransitiveBridge

bridge = TransitiveBridge()
result = bridge.fuse(
    domains=["physics", "computer_science", "linguistics"],
    query="quantum natural language processing"
)
```

**Human Workflows:**
- Prototype new ideas with AI assistance
- Compare current vs deprecated knowledge
- Multi-domain problem solving

**Development Tasks:**
- Creating new procedural generators
- Testing cross-modal reasoning
- Museum artifact retrieval and analysis

#### 3. Bathtub — Sleep Chamber & Galaxy Universe Introspection
**Purpose:** Sphere-shaped sleep chamber where Galaxy Universe projects from avatar's head center for introspection and consolidation.

**Architecture:**
- Imaginary sphere carved into floor (sofa/ball-pit concept)
- Avatar center point for sleep cycles
- Galaxy Universe projection (addressable 3D RAM with multiple galaxies loaded)
- Stars transform: light particles → 3D shapes/textures (procedural dual-view)

**AI Agent Workflows:**
```python
# Sleep-time consolidation
from knowledge3d.cranium.sleep.sleep_time_compute import SleepTimeCompute

sleep = SleepTimeCompute(house_id="default")
avatar.navigate_to_room("Bathtub")  # Enter sleep chamber

# Galaxy Universe projection activates
sleep.project_galaxy_universe(
    galaxies=["text", "visual", "audio", "reasoning"],
    mode="introspection"
)

# Consolidate Galaxy → House
sleep.consolidate(
    ema_factor=0.9,
    prune_threshold=0.5,
    output_dir="../Knowledge3D.local/house_zone7/"
)
```

**Human Workflows:**
- Observe AI sleep cycles (educational)
- Query Galaxy Universe during consolidation
- Pick and inspect individual stars (visual or data)

**Development Tasks:**
- Tuning consolidation parameters
- Debugging Galaxy memory leaks
- Analyzing sleep-time reasoning patterns

**Galaxy Universe Loaded Galaxies:**
- **Text Galaxy:** Language embeddings, RPN vocabulary (33K+ trigrams)
- **Visual Galaxy:** Font glyphs, procedural drawings (168K+ programs)
- **Audio Galaxy:** Speech patterns, acoustic features (4K+ audio files)
- **Reasoning Galaxy:** ARC-AGI patterns, logic structures
- **Domain Galaxies:** Math, physics, chemistry (future specialists)

#### 4. Living Room — Old Paradigm Bridge
**Purpose:** Bridge to conventional 2D interfaces with VM casting, projection screens, and legacy system access.

**Components:**
- Sofa/furniture (customizable like Minecraft/The Sims)
- Projection screens (castable to full-screen mode)
- Desktop corner with keyboard/mouse (AR/VR mapped)
- Virtual KVM for multiple VMs

**AI Agent Workflows:**
```python
# Cast VM to projection screen
from knowledge3d.bridge.vm_casting import VMCaster

caster = VMCaster()
caster.cast_vm(
    vm_id="ubuntu-dev-01",
    protocol="vnc",
    endpoint="localhost:5901",
    target_screen="living_room/projection_wall",
    resolution=[1920, 1080]
)

# Query web content via embedded browser
tablet.browse(
    url="https://arxiv.org/abs/2501.12345",
    capture_mode="structured",
    target_room="Living Room"
)
```

**Human Workflows:**
- Work in legacy applications (VS Code, browsers, IDEs) inside K3D
- Full-screen projection mode for focused work
- "Move-along" 3D PiP mode (AR/VR concept)

**Development Tasks:**
- VM integration testing
- Projection screen texture mapping
- Keyboard/mouse input mapping (3D → 2D)

**VM Casting Protocol Stack:**
```
Docker Container → VNC/RDP Server → WebRTC Stream → Three.js Texture → Projection Screen
```

#### 5. Knowledge Gardens — Ontology Greenhouse
**Purpose:** Circular indoor greenhouse for ontology trees and knowledge that doesn't fit library classification.

**AI Agent Workflows:**
```python
# Generate ontology tree
from knowledge3d.tools.gardens import build_ontology_tree

tree = build_ontology_tree(
    domain="computer_science",
    root_concept="artificial_intelligence",
    max_depth=5,
    output="../Knowledge3D.local/house_zone7/gardens/ai_ontology.glb"
)

# Navigate tree structure
avatar.navigate_to_room("Knowledge Gardens")
avatar.traverse_ontology(
    tree="ai_ontology",
    path=["AI", "Machine Learning", "Deep Learning", "Transformers"]
)
```

**Human Workflows:**
- Explore knowledge hierarchies visually
- Add new ontology branches
- Prune outdated relationships

**Development Tasks:**
- Building ontology generators
- Tree visualization optimization
- Cross-ontology linking

### Portal-Based Collaboration Patterns

**Portals** enable multi-agent, multi-house collaboration with preserved attribution and federated knowledge access.

#### Local Portals (Same Host)
```python
# Connect to local AI house
from knowledge3d.spatial.portals import PortalManager

portal = PortalManager()
portal.open_local_portal(
    source_room="Workshop",
    target_house="ai_assistant_house",
    target_room="Library",
    capabilities=["read", "query"]  # No write access
)

# Query remote house via tablet
tablet.search(
    query="recent research",
    sources=["Portal:ai_assistant_house/Library"],
    attribution=True  # Preserve provenance
)
```

#### Remote Portals (Internet Federation)
```json
{
  "portal": {
    "type": "remote",
    "endpoint": "wss://research-lab.k3d.io/house/shared",
    "protocol": "k3d-portal-v1",
    "auth": {
      "method": "oauth2",
      "provider": "github"
    },
    "capabilities": ["read", "collaborate"]
  }
}
```

**Multi-Agent Scenarios:**
1. **Shared Research:** Multiple agents access same Library portal, contribute findings to shared workspace
2. **Peer Review:** Agent A writes in Workshop, Agent B reviews via read-only portal
3. **Knowledge Trading:** "Software as space" selling model — rent/sell portal access to specific rooms

### Memory Tablet Usage for Cross-Space Work

**The Memory Tablet** is the universal interface bridging spatial (3D rooms) and conventional (2D screens) paradigms.

#### Tablet as Projection Screen
```python
# Cast any OS app to tablet display
tablet.cast_to_display(
    source_app="firefox",
    source_vm="ubuntu-dev-01",
    resolution=[1920, 1080],
    mode="fullscreen",
    input_mapping="3d_pointer"
)
```

#### Tablet for Portal Navigation
```python
# Tablet remains connected to home house when in remote space
avatar.enter_portal("wss://remote.k3d.io/house/research")

# Tablet can still access home knowledge
home_results = tablet.search(
    query="my notes on transformers",
    sources=["Home/Library"],  # Explicit home house reference
    lod="coarse"  # Bandwidth-conscious
)

# Load home knowledge into remote Galaxy
tablet.stream_to_galaxy(
    artifact_path="Home/Library/ML_Notes.glb",
    target_galaxy="remote",  # Currently active Galaxy
    lod="medium"
)
```

### House-First Development Principles

**CRITICAL:** The avatar always lives in the House, not in the Galaxy.

#### Embodiment Constraint
```python
# ✅ CORRECT: Avatar in House, consults Galaxy via tablet
avatar.navigate_to_room("Workshop")
galaxy_context = tablet.query_galaxy(
    query="recent reasoning patterns",
    galaxy="reasoning"
)
avatar.reason_with(galaxy_context, house_context)

# ❌ WRONG: Placing avatar inside Galaxy
avatar.teleport_to(galaxy_position)  # VIOLATION!
```

#### Galaxy as Introspection Only
```python
# Galaxy views are for diagnostics/introspection
avatar.navigate_to_room("Bathtub")  # Sleep chamber
galaxy_view = tablet.project_galaxy_universe(
    galaxies=["text", "visual"],
    mode="introspection"  # Not navigation!
)

# Human and AI can inspect stars
tablet.inspect_star(
    galaxy="text",
    star_id=12345,
    view_mode="dual"  # Visual 3D + semantic data
)
```

#### Navigation in House, Reasoning in Galaxy
```
Avatar Movement:         Reasoning Access:
House Room Navigation    Galaxy Query via Tablet
├─ Library → Workshop    ├─ Search text embeddings
├─ Workshop → Bathtub    ├─ Visual similarity search
├─ Bathtub → Gardens     ├─ Audio pattern matching
└─ Portal → Remote       └─ Cross-modal fusion
```

### Room-Specific Development Tasks

#### Adding New Knowledge to Library
```bash
# 1. Ingest PDFs
python -m knowledge3d.ingestion.documents.pdf_ingestor \
  --input research_papers/ \
  --output "$K3D_LOCAL_DIR/datasets/new_research.glb"

# 2. Generate Galaxy
python -m k3dgen \
  --text "$K3D_LOCAL_DIR/datasets/new_research.glb" \
  --gltf "$K3D_LOCAL_DIR/galaxy/research_galaxy.glb"

# 3. Sleep consolidation
python -m knowledge3d.cranium.sleep.sleep_time_compute \
  --house-id default \
  --target-room Library

# 4. Verify in Library
avatar.navigate_to_room("Library")
tablet.search(query="specific topic", sources=["House/Library"])
```

#### Prototyping in Workshop
```bash
# Load Museum artifacts for comparison
python -m knowledge3d.tools.relocate_to_museum --list-available

# Bring specific artifact to Workshop
python -m knowledge3d.tools.load_museum_box \
  --artifact Museum/Zone8/2024-11-15/Old_Approach.glb \
  --target Workshop \
  --mode read_only
```

#### Sleep Cycle in Bathtub
```bash
# Manual sleep trigger (for testing)
python -m knowledge3d.cranium.sleep.sleep_time_compute \
  --house-id default \
  --mode full \
  --galaxies text,visual,audio,reasoning

# Monitor Galaxy Universe projection
# (Human client in Three.js viewer sees visual projection)
# (AI client sees semantic graph updates)
```

#### VM Casting in Living Room
```bash
# Start VM with VNC
docker run -d -p 5901:5901 \
  -e VNC_PASSWORD=secure \
  ubuntu-desktop:latest

# Cast to Living Room projection screen
python -m knowledge3d.bridge.vm_casting \
  --vm-id ubuntu-dev \
  --protocol vnc \
  --endpoint localhost:5901 \
  --target-room "Living Room" \
  --screen "projection_wall"
```

### Testing Room-Based Features

```bash
# Test room navigation
pytest tests/test_room_navigation.py -v

# Test portal connections
pytest tests/test_portal_federation.py -v

# Test VM casting
pytest tests/test_vm_casting.py -v

# Test Galaxy Universe projection
pytest tests/test_galaxy_universe.py -v -m cuda
```

### Multi-Agent Room Collaboration

#### Scenario: Research Paper Analysis
```python
# Agent A (Researcher) in Library
agent_a.navigate_to_room("Library")
agent_a.search(query="transformer architectures")

# Agent B (Critic) connects via portal
portal_b = agent_b.open_portal(
    target_house="agent_a_house",
    target_room="Library",
    capabilities=["read", "comment"]
)

# Collaborative annotation
agent_a.annotate_artifact("Attention_Is_All_You_Need.glb")
agent_b.add_comment(
    artifact="Attention_Is_All_You_Need.glb",
    comment="Consider computational complexity analysis"
)
```

#### Scenario: Cross-House Knowledge Fusion
```python
# Multiple agents contribute to shared Workshop
workshop_portal = PortalManager.create_shared_space(
    room="Workshop",
    participants=["agent_a", "agent_b", "human_researcher"],
    capabilities={
        "agent_a": ["read", "write"],
        "agent_b": ["read", "write"],
        "human_researcher": ["read", "write", "admin"]
    }
)

# Agents work in parallel
agent_a.generate_hypothesis(topic="quantum_nlp")
agent_b.validate_hypothesis(agent_a.hypothesis)
human_researcher.review_and_approve()
```

### Room Architecture for Game Development

**The House is a game. Rooms are game modes. Knowledge is the terrain.**

#### Game Engine Techniques Applied
```python
# LOD (Level of Detail) per room
room_config = {
    "Library": {
        "lod_levels": ["coarse", "medium", "full"],
        "memory_budget_mb": 50,
        "frustum_culling": True
    },
    "Workshop": {
        "lod_levels": ["medium", "full"],
        "memory_budget_mb": 100,
        "dynamic_loading": True
    }
}

# Scene management (doors as loading screens)
avatar.open_door("Library_to_Workshop")
# → Triggers:
#   1. Save Library state
#   2. Unload Library high-detail assets
#   3. Load Workshop medium-detail assets
#   4. Transition animation
```

#### Spatial Audio in Rooms
```python
# Audio sources localized in 3D
audio_manager.add_source(
    room="Workshop",
    position=[5.0, 2.0, 3.0],
    sound="machine_hum.ogg",
    falloff="inverse_square"
)

# Avatar proximity triggers audio
if avatar.distance_to(audio_source) < 10.0:
    audio_manager.set_volume(source, calculate_volume(distance))
```

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
- Follow `docs/ENV_POLICY.md` to create the canonical environments (GPU: `k3d-cranium`, CPU test harness: `k3d-testing`) and run commands inside them (e.g., `conda activate k3d-cranium`, then `env PYTHONPATH=. python -m ...`).
- For heavy ingestion/builds, prefer storing raw media under `/home/daniel/K3D_llama_cpp/datasets` and curated subsets under `../Knowledge3D.local/datasets`.
- Phase 25 sleep consolidation depends on the CUDA-enabled `k3d-cranium` env (with `cuda-python`); keep that env active so `SleepTimeCompute` can continue materialising reflection diaries in the House.
- **CUDA/PTX Version Compatibility**: When upgrading `cuda-python`, be aware that bundled NVRTC may generate PTX versions incompatible with your driver (e.g., CUDA 12.8's PTX 8.7 requires driver 570+, but driver 550 only supports PTX 8.4). Solution: replace bundled `libnvrtc.so.12` with symlink to system CUDA toolkit version. See [docs/CUDA_PTX_VERSION_COMPATIBILITY_GUIDE.md](docs/CUDA_PTX_VERSION_COMPATIBILITY_GUIDE.md) for diagnostic steps and prevention strategies. Validated in Phase 2 codec GPU verification (fixed CUDA Error 222).
- After long RLWHF batches, refresh Phase 10 "thinking tags" (see `knowledge3d/tools/phase10/thinking_tag_trainer.py`) so the UI exposes the model's active reasoning labels.
