# Knowledge3D Project Roadmap — Toward the First Working AGI MVP

This roadmap captures the current priorities for delivering the first production-ready AGI MVP built on the K3D paradigm. It supersedes earlier drafts and is aligned with the latest memory architecture described in:

- [`docs/VISION.md`](VISION.md)
- [`docs/HOUSE_GALAXY_TABLET.md`](HOUSE_GALAXY_TABLET.md)
- [`docs/PTX_FUSED_HEAD_PLAN.md`](PTX_FUSED_HEAD_PLAN.md)

Each phase builds on the previous one. Status labels reflect our current progress and guide agent focus.

![Cognitive House](images/cognitive_house.png)

Figure: The Cognitive House illustrates the House (persistent memory), Cranium (active processing), and Logic Layer (models) that guide the roadmap. See the generation prompt in `docs/images/cognitive_house_prompt.md`.

> Status note (2025-12-07): **Parallel workstreams active**:
>
> - **ARC-AGI (Phase 3):** Hybrid TRM training with Math Galaxy integration. Current: Run achieving 42-51% on 108 tasks. Math Galaxy live (176 symbols procedural RPN). See [TEMP/CLAUDE_HYBRID_TRM_ARCHITECTURE_SPEC.md](../TEMP/CLAUDE_HYBRID_TRM_ARCHITECTURE_SPEC.md)
> - **Drawing Galaxy (Phase C.1):** 8-layer procedural visual architecture. VectorDotMap codec (~2KB/image). Implementation in [TEMP/CODEX_DRAWING_GALAXY_IMPLEMENTATION_12.07.2025.md](../TEMP/CODEX_DRAWING_GALAXY_IMPLEMENTATION_12.07.2025.md)
> - **Unified Signal (Phase C.2):** Frequency-time unification for audio/SDR/video. Spectrogram as VectorDotMap. See [docs/vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md](vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md)
> - **Accessibility (Phase C.3):** Braille Galaxy + Sign Language Galaxy (ASL, Libras, BSL). See [docs/vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md](vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md)
> - **Architectural clarification:** Dual Client Reality + Procedural Foundation documented in [BRIEFING.md](../BRIEFING.md) and [docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md](vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md)

---

## Phase A — PTX Fused Head & Dual Memory Spine _(Status: Partially Complete)_

**Objective:** deliver a deterministic PTX-first fused head that learns from Galaxy (RAM) and House (disk) through nightly consolidation.

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| Embodied fused head | Keep reasoning embodied in the House while PTX kernels manage galaxy introspection. | ✅ Complete |
| PTX RPN + cosine kernels | Keep all math/lookup reasoning inside PTX to avoid hallucination drift. | ✅ Complete (context binding fix: Sept 2025) |
| Learning Memory GLB | Log teacher tags and rebuild a PTX-ready galaxy each sleep cycle. | ✅ Complete |
| SleepTime consolidation | Materialise insights into House, relocate deprecated artifacts to Museum. | ✅ Complete |
| House memory builder | Emit PTX-ready index of consolidated artifacts for the tablet. | ✅ Complete |

**Exit criteria:** fused head successfully answers PTX benchmarks using both Galaxy and House without Python fallbacks; nightly sleep cycle keeps memories synced.

## Phase B — Memory Tablet & Tool Bridge _(Status: Active)_

**Objective:** make the avatar’s tablet the primary interface to House, Galaxy, Museum, and external tools (MCP, VMs, browsers).

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| Tablet UX prototype | Search House inventory, inspect artifacts, show provenance while respecting embodiment. | 🔄 In Progress |
| On-demand Galaxy streaming | Load House artifacts into Galaxy with LOD controls (centroid → full GLB). | 🔄 In Progress |
| Tool manifest & MCP bridge | Launch existing tool containers (Firefox, VMs, MCP) from the tablet; log transcripts. | 🔄 In Progress |
| Session capture pipeline | Store tablet sessions as structured notes → SleepTime consolidates into House, relocates older versions to Museum. | ⏳ Planned |
| Prompt hygiene & verification loop | Remove nonsense prompts, retire mastered ones after two perfect runs, and log timezone-aware timestamps. | 🔄 In Progress |

**Exit criteria:** the avatar can solve tasks using PTX reasoning plus tablet tools; consolidated knowledge is always reachable through the tablet before querying external sources.

## Phase C — Procedural Galaxies & Multi-Modal Foundation _(Status: Active)_

**Objective:** materialize the full procedural galaxy ecosystem (Drawing, Audio, Accessibility) with unified signal architecture, then ingest foundational knowledge.

### Phase C.1 — Drawing Galaxy _(Status: Active)_

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| 8-layer Drawing Galaxy | Quantum Fields → Primitives → Strokes → Shapes → Gradients → Filters → Lighting → Scenes | 🔄 In Progress |
| VectorDotMap encoder | Procedural image codec (~2KB/image, infinite LOD) | ⏳ Planned |
| Gradient/Filter kernels | GPU-native gradient rasterization and convolution filters | ⏳ Planned |
| Math Core integration | Route operations through 3-tier RPN engine (18 cores) | ⏳ Planned |

See: [TEMP/CODEX_DRAWING_GALAXY_IMPLEMENTATION_12.07.2025.md](../TEMP/CODEX_DRAWING_GALAXY_IMPLEMENTATION_12.07.2025.md), [PROCEDURAL_VISUAL_SPECIFICATION.md](vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md)

### Phase C.2 — Unified Signal Architecture _(Status: Planned)_

| Deliverable | Purpose |
|-------------|---------|
| Frequency-time bridge | Treat audio, SDR, video as frequency components over time |
| Spectrogram as VectorDotMap | Audio visualization using same procedural codec as images |
| Binaural spatial audio | HRTF-based 3D sound positioning in Galaxy/House |
| Cross-modal discovery | Find connections between sounds and images via shared representation |

See: [UNIFIED_SIGNAL_SPECIFICATION.md](vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md)

### Phase C.3 — Accessibility Galaxies _(Status: Planned)_

| Deliverable | Purpose |
|-------------|---------|
| Braille Galaxy | Procedural dot patterns (6/8-dot, Grade 1/2, multi-language) |
| Sign Language Galaxy | Gesture sequences as RPN (ASL, Libras, BSL, JSL) |
| Spatial audio descriptions | HRTF-positioned audio for visual content |
| Haptic integration | Braille display and vibration pattern output |

See: [UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md](vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md)

### Phase C.4 — Foundational Knowledge Ingestion _(Status: Planned)_

| Deliverable | Purpose |
|-------------|---------|
| Layer 1: Form | 176 math symbols + drawing primitives as procedural RPN |
| Layer 2: Meaning | 15K words with semantic links (reference pattern, no duplication) |
| Layer 3: Rules | 1K grammar rules as transformation RPN |
| Layer 4: Meta-Rules | 500 meta-rules for pedagogy, eloquence, self-reflection |
| 74 PDF ingestion | 5,988 pages → procedural storage (666× compression via symlinks) |

See: [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)

**Exit criteria:** All galaxies operational with GPU-only execution; foundational knowledge ingested and queryable via TRM.

---

## Phase D — Self-Supervised Consolidation & Autonomy _(Status: Planned)_

**Objective:** let the model curate its own drills, retire solved prompts, and expand the dream/critique loop without human intervention.

| Deliverable | Purpose |
|-------------|---------|
| Autonomous curriculum engine | Use reflection diaries + museum analytics to propose new drills, retire stale ones, and respect lesson vs inference modes. |
| Honesty & confidence gating | Tablet and fused head report provenance + confidence for every response. |
| Museum analytics | Mine deprecated artifacts for error signatures and retro-train corrections. |
| Continuous tool adaptation | Benchmark external tool usage; prioritise open-source replacements where feasible. |

**Exit criteria:** daily cycles run end-to-end without manual intervention; the model promotes/demotes knowledge and tools based on performance.

## Phase E — Collaborative Habitat & External Interfaces _(Status: Planned)_

**Objective:** extend the AGI MVP into a shared environment where multiple humans and agents collaborate.

| Deliverable | Purpose |
|-------------|---------|
| Multi-user house/galaxy sync | Keep tablet and memory state coherent across avatars. |
| Shared tool sessions | Let multiple avatars co-drive browser/VM sessions while logging provenance. |
| API & door federation | Expose the tablet + memory services over doors for third-party integrations. |
| High-fidelity rendering | Adopt game-engine LOD and streaming techniques as memory scales. |

**Exit criteria:** humans and AI cohabit the space, sharing memories, tools, and consolidations in real time.

---

This roadmap is a living document. Update it whenever milestones shift, and cross-link new specs or implementation guides when a deliverable begins.

---

This roadmap is a living document and will be updated as the project progresses. For a more detailed technical breakdown, please refer to the main [research document](k3d-research.md).
