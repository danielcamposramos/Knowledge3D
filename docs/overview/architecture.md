# Knowledge3D: architecture and alternatives

Part of [Knowledge3D](../../README.md), moved here whole from its front page.

## Architecture Overview

### Three-Brain System (Neuroscience-Inspired)

| Component | Biological Analogy | Role | Storage |
|-----------|-------------------|------|---------|
| **Cranium** | Prefrontal cortex | Reasoning via 46+ PTX kernels | GPU execution units |
| **Galaxy** | Hippocampus | Working memory during active reasoning | VRAM (ephemeral) |
| **House** | Neocortex | Permanent knowledge as 3D spatial objects | Disk (GLB assets) |

**The House** is a literal 3D virtual world — software as a space. A Library has bookshelves with books. A Garden has knowledge trees whose branches carry domain details. A Workshop has tools the AI uses. The avatar LIVES here.

**The Galaxy** is the AI's working memory — loaded from the House on demand. During reasoning, concepts organize via **"semantic gravity cohered by meaning"** (Christoph Dorn): a ternary force where meaning replaces mass. After reasoning, stars return to their House positions unchanged.

**The Cranium** executes reasoning via 46+ hand-written PTX kernels with zero external dependencies. The composed head pipeline: Morton Octree → LED-A* → Frustum Cull → Dynamic LOD → Nine-Chain Swarm → Halting Gate.

### Sovereignty: Zero External Dependencies in Hot Path

The reasoning path uses **only** PTX kernels + Galaxy queries + RPN composition. No numpy, scipy, torch, or any external framework. "We fail and fix — this is the goal." Python handles boot (~200 lines) and I/O. Everything else runs on GPU.

### Key Specifications

**Architecture & System:**
- [THREE_BRAIN_SYSTEM_SPECIFICATION.md](../vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md) — Cranium + Galaxy + House
- [KNOWLEDGEVERSE_SPECIFICATION.md](../vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md) — 7-region unified VRAM substrate
- [SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md](../vocabulary/SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md) — Houses, rooms, portals, Memory Tablet

**Knowledge Representation:**
- [MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md](../vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md) — Atomic knowledge unit + semantic gravity
- [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](../vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) — Same source for humans AND AI
- [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](../vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) — 4-layer architecture (Form → Meaning → Rules → Meta-Rules)

**Execution & Reasoning:**
- [SOVEREIGN_NSI_SPECIFICATION.md](../vocabulary/SOVEREIGN_NSI_SPECIFICATION.md) — PTX-only neurosymbolic integration
- [RPN_DOMAIN_OPCODE_REGISTRY.md](../vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) — Reverse Polish Notation opcode registry
- [HYPER_PARALLEL_PROCESSING.md](../vocabulary/HYPER_PARALLEL_PROCESSING.md) — Parallel cognitive channels + ternary logic

**Domain Galaxies:**
- [REALITY_ENABLER_SPECIFICATION.md](../vocabulary/REALITY_ENABLER_SPECIFICATION.md) — Procedural physics/chemistry/biology
- [PROCEDURAL_VISUAL_SPECIFICATION.md](../vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md) — Drawing Galaxy + VectorDotMap codec
- [UNIFIED_SIGNAL_SPECIFICATION.md](../vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md) — Audio, SDR, video as unified signal

**Full index**: [docs/vocabulary/README.md](../vocabulary/README.md)

## Alternatives Considered

### Why Not Traditional Knowledge Graphs (RDF/OWL)?
RDF/OWL already support explicit meaning, relationships and reasoning. K3D explores how reusable procedures can connect those descriptions to executable behavior. Both approaches need an execution environment. Proposed mappings must show what meaning survives exchange and where information is lost.

### Why Not LLM Embeddings?
Embeddings are opaque vectors — you can't inspect why two concepts are similar. K3D's meaning-centric stars carry explicit RPN programs, surface forms, and taxonomic references. The reasoning is auditable. Additionally, embeddings duplicate knowledge per model; stars store it once.

### Why Not Existing Game Engines (Unity/Unreal)?
Game engines provide mature rendering, spatial indexing and simulation. K3D builds on such techniques while investigating a knowledge model shared by human and AI clients. Whether that additional model helps a workflow must be demonstrated against existing approaches.

### Why Not Framework-Based AI (PyTorch/TensorFlow)?
K3D chooses direct GPU execution for control over its reasoning path. That choice has engineering and portability costs. Framework-based systems remain useful alternatives; comparisons need matched workloads and measured results, not an assumption that abstraction prevents inspection.

### Why Build the House as Literal 3D Space?
The IT industry borrowed spatial metaphors (windows, desktop, doors, folders, addresses) and kept them flat. K3D reverses this: spatial metaphors become actual 3D objects. A "door" is a real door you walk through. A "window" is a real window you look out of. This isn't metaphor — it's software as a space.
