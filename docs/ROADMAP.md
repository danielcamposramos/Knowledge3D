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

## Current Active Phase: Phase E — ARC-AGI-3 Game-Loop + Python→PTX Sovereignty _(Status: Active, reconciled April 2026)_

> **Note (2026-03-27):** Phases A–D above reflect original roadmap structure. Current work is tracked via TEMP/ phase reports. The active workstream is Phase E below, which subsumes the sovereignty migration goals of the original Phase D.

> **Reconciliation note (2026-04-08):** local repo reality has moved beyond the original March-only Phase E table. The live track now has a landed April baseline for physics, reality textures, entities, proceduralizer/OCR ingest, zero-copy control-plane repair, and ARC R0 scaffolding. The sections below distinguish landed baseline work from still-planned hot-path migration items.

**Objective:** Integrate ARC-AGI-3 (interactive game-loop benchmark), eliminate remaining Python from the hot path, and proceduralize all grammar rules/meta-rules as sovereign Galaxy entries.

**Context:** ARC-AGI-3 shifted from static grid I/O to a real-time interactive game (agents submit actions, receive frames). This matches K3D's TRM game-loop architecture exactly. See: [TEMP/CLAUDE_PHASE_E_ARC_AGI_3_SOVEREIGNTY_SPEC_03.27.2026.md](../TEMP/CLAUDE_PHASE_E_ARC_AGI_3_SOVEREIGNTY_SPEC_03.27.2026.md)

### Phase E.0 — Reconciled April Baseline

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| Sovereign physics P0 surface | XPBD/body/material/runtime base on modular RPN path | ✅ Landed locally |
| Reality engine Step 2 textures | procedural texture opcodes + kernel surface | ✅ Landed locally |
| Reality engine Step 3 entities | MeaningCentricStar entity projection + hot-path struct | ✅ Landed locally |
| Proceduralizer + OCR retry ingest | ordered PDF ingest, cloud OCR retry, staged second pass | ✅ Live |
| Zero-copy import/control-plane repair | `knowledge3d.cranium.kernels` utility surface + honest tests | ✅ Landed locally |
| ARC-AGI-2 submission formatter/runner | R0 competition artifact surface | ✅ Landed locally |
| Paper-track scaffold | evidence bundle + manuscript scaffold | ✅ Landed locally |

**Baseline gate:** focused slices for sovereign physics, procedural texture, entity surface, proceduralizer/ingest, drawing/quarantine honesty, and zero-copy control plane are green.

### Phase E.1 — Grid Perception (GPU-Native)

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| ARC-AGI-3 thin Python client | HTTP I/O only (~50 lines), marshal frames to GPU, read actions from GPU | ⏳ Planned |
| `gre_flood_fill.cu` | Replace Python DFS connected components (knowledgeverse.py:92-128) | ⏳ Planned |
| GPU histogram reduction | Replace `_dominant_grid_color()` Python | ⏳ Planned |
| GPU symmetry detection | Replace `_grid_has_symmetry()` Python | ⏳ Planned |
| TRM always-on | Remove env-var gating (`K3D_TRM_SHADOW`, `K3D_TRM_NAVIGATE`) | ⏳ Planned |

### Phase E.2 — Action Selection (GPU-Native)

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| Replace `_select_composed_head_candidate()` | 1,157 lines Python → GPU scoring in composed head chain | ⏳ Planned |
| Wire all 15 GRE specialist kernels | All loaded kernels callable by swarm workers | ⏳ Planned |
| Action output protocol | TRM tick output → GameAction mapping | ⏳ Planned |

### Phase E.3 — Grammar Sovereignty

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| Grammar rules → VRAM hash table | 245+ ARC primitives as first-class Galaxy entries | ⏳ Planned |
| Bidirectional symlink index | GPU pointer graph for dual-way traversal | ⏳ Planned |
| Regex elimination | Text normalization as Grammar Galaxy rule application | ⏳ Planned |
| Meta-rules as Galaxy entries | Game strategies as procedural RPN programs | ⏳ Planned |

### Phase E.4 — Strategy + Consolidation

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| Sleeptime sovereignty | Remove numpy/fallbacks, implement Stage A/B consolidation | ⏳ Planned |
| Game trace consolidation | Successful ARC-AGI-3 action sequences → Grammar rules | ⏳ Planned |
| RPN composition enrichment | Replace text-append enrichment with RPN program composition | ⏳ Planned |

**Quality gates (per sub-phase):** ARC-AGI-2 10/10 regression pinned, Math 20/20 pinned, GPU utilization increase, knowledgeverse.py line count decrease, zero Python fallbacks in hot path.

**R0 competition order (current truth):**
1. ARC-AGI-2 submission infrastructure
2. paper-track evidence/manuscript assets
3. score improvement on ARC-AGI-2 / ARC-AGI-3 / Track C

**Exit criteria:** K3D plays ARC-AGI-3 games autonomously via TRM game loop. Grammar rules and meta-rules are sovereign Galaxy entries with bidirectional symlinks. `knowledgeverse.py` < 1,000 lines. GPU utilization > 30%.

**Key specs:**
- [ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md](vocabulary/ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md) — ternary-gated computation governance
- [PM_KR_X3D_ADAPTIVE_REASONING_COMPONENT.md](w3c/x3d/PM_KR_X3D_ADAPTIVE_REASONING_COMPONENT.md) — X3D budget component
- [HYPER_PARALLEL_PROCESSING.md](vocabulary/HYPER_PARALLEL_PROCESSING.md) — specialist swarm paradigm

**Benchmark state entering Phase E (D.3, 2026-03-26):**
- Combined: 857/5954 (14.39%) — warm 35%
- ARC: 2/42, Math: 1/500, GSM8K: 3/462, LHE: 1/35, MMLU: 850/4915
- Sleep-time: checkpoint saved, 18,189 specialist routes updated
- GPU util: 0.17% avg (target: >30%)

---

This roadmap is a living document. Update it whenever milestones shift, and cross-link new specs or implementation guides when a deliverable begins.
