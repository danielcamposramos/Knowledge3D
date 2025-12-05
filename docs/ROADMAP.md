# Knowledge3D Project Roadmap — Toward the First Working AGI MVP

This roadmap captures the current priorities for delivering the first production-ready AGI MVP built on the K3D paradigm. It supersedes earlier drafts and is aligned with the latest memory architecture described in:

- [`docs/VISION.md`](VISION.md)
- [`docs/HOUSE_GALAXY_TABLET.md`](HOUSE_GALAXY_TABLET.md)
- [`docs/PTX_FUSED_HEAD_PLAN.md`](PTX_FUSED_HEAD_PLAN.md)

Each phase builds on the previous one. Status labels reflect our current progress and guide agent focus.

![Cognitive House](images/cognitive_house.png)

Figure: The Cognitive House illustrates the House (persistent memory), Cranium (active processing), and Logic Layer (models) that guide the roadmap. See the generation prompt in `docs/images/cognitive_house_prompt.md`.

> Status note (2025-11-25): **Parallel workstreams active**:
> - **ARC-AGI (Phase 3):** Sovereign visual reasoning using Drawing + Grammar + Character Galaxy composition. Current: Training library growth (1662 rules, 1556 shapes). Next: Deduplication + quality filtering. Goal: 5-10% accuracy via semantic TRM routing. See [TEMP/CODEX_IMPLEMENT_QUALITY_SEMANTIC_CORRECT_11.25.2025.txt](../TEMP/CODEX_IMPLEMENT_QUALITY_SEMANTIC_CORRECT_11.25.2025.txt)
> - **Reality Galaxy (Phases 4-5):** Phase 5 capacity demonstration (CPU path) complete. Stress + scaling benchmarks executed (artifacts in `output/benchmarks/`), 26 GLBs exported (`output/gltf/`), analysis in [TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md](../TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md). Next: Phase 6 UI integration and GPU-kernel validation once cupy available.
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

## Phase C — Self-Supervised Consolidation & Autonomy _(Status: Planned)_

**Objective:** let the model curate its own drills, retire solved prompts, and expand the dream/critique loop without human intervention.

| Deliverable | Purpose |
|-------------|---------|
| Autonomous curriculum engine | Use reflection diaries + museum analytics to propose new drills, retire stale ones, and respect lesson vs inference modes. |
| Honesty & confidence gating | Tablet and fused head report provenance + confidence for every response. |
| Museum analytics | Mine deprecated artifacts for error signatures and retro-train corrections. |
| Continuous tool adaptation | Benchmark external tool usage; prioritise open-source replacements where feasible. |
| Time & math enrichment | Expand curated datasets (machine/human time, basic & financial math) via exaone models, feeding them through the house-tablet pipeline. |
| Foundational knowledge ingestion | Ingest 74 PDFs (5,988 pages) as always-loaded base knowledge: 4-layer architecture (Form → Meaning → Rules → Meta-Rules), 152 math symbols, 15K words, 1K grammar rules, 500 meta-rules. Symlink pattern achieves 666× compression. Integrates with TRM ternary logic (intelligent sparsity, 16× mask compression) and Vector Dot Maps (quantum field procedural glyphs, multi-modal cross-resonance). Supports pedagogy, eloquence, self-reflection, storytelling. See [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) and [KNOWLEDGE_INGESTION_PLAN_V5_CODEX_READY.md](../TEMP/KNOWLEDGE_INGESTION_PLAN_V5_CODEX_READY.md). |

**Exit criteria:** daily cycles run end-to-end without manual intervention; the model promotes/demotes knowledge and tools based on performance.

## Phase D — Collaborative Habitat & External Interfaces _(Status: Planned)_

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
