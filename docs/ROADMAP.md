# Knowledge3D Project Roadmap — Toward the First Working AGI MVP

This roadmap captures the current priorities for delivering the first production-ready AGI MVP built on the K3D paradigm. It supersedes earlier drafts and is aligned with the latest memory architecture described in:

- [`docs/VISION.md`](VISION.md)
- [`docs/HOUSE_GALAXY_TABLET.md`](HOUSE_GALAXY_TABLET.md)
- [`docs/PTX_FUSED_HEAD_PLAN.md`](PTX_FUSED_HEAD_PLAN.md)

Each phase builds on the previous one. Status labels reflect our current progress and guide agent focus.

![Cognitive House](images/cognitive_house.png)

Figure: The Cognitive House illustrates the House (persistent memory), Cranium (active processing), and Logic Layer (models) that guide the roadmap. See the generation prompt in `docs/images/cognitive_house_prompt.md`.

---

## Phase A — PTX Fused Head & Dual Memory Spine _(Status: Partially Complete)_

**Objective:** deliver a deterministic PTX-first fused head that learns from Galaxy (RAM) and House (disk) through nightly consolidation.

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| Embodied fused head | Keep reasoning embodied in the House while PTX kernels manage galaxy introspection. | ✅ Complete |
| PTX RPN + cosine kernels | Keep all math/lookup reasoning inside PTX to avoid hallucination drift. | ✅ Complete (context binding fix: Sept 2025) |
| Learning Memory GLB | Log teacher tags and rebuild a PTX-ready galaxy each sleep cycle. | ✅ Complete |
| SleepTime consolidation | Materialise insights into House, relocate deprecated artifacts to Museum. | ✅ Complete |
| House memory builder | Emit PTX-ready index of consolidated artifacts for the tablet. | 🔄 In Progress (highest priority) |

**Exit criteria:** fused head successfully answers PTX benchmarks using both Galaxy and House without Python fallbacks; nightly sleep cycle keeps memories synced.

## Phase B — Memory Tablet & Tool Bridge _(Status: Active)_

**Objective:** make the avatar’s tablet the primary interface to House, Galaxy, Museum, and external tools (MCP, VMs, browsers).

| Deliverable | Purpose | Status |
|-------------|---------|--------|
| Tablet UX prototype | Search House inventory, inspect artifacts, show provenance while respecting embodiment. | 🔄 In Progress |
| On-demand Galaxy streaming | Load House artifacts into Galaxy with LOD controls (centroid → full GLB). | 🔄 In Progress |
| Tool manifest & MCP bridge | Launch existing tool containers (Firefox, VMs, MCP) from the tablet; log transcripts. | 🔄 In Progress |
| Session capture pipeline | Store tablet sessions as structured notes → SleepTime consolidates into House, relocates older versions to Museum. | ⏳ Planned |
| Prompt verification loop | Retire prompts once fused head + tablet confirmation succeed twice; move to verification list. | ⏳ Planned |

**Exit criteria:** the avatar can solve tasks using PTX reasoning plus tablet tools; consolidated knowledge is always reachable through the tablet before querying external sources.

## Phase C — Self-Supervised Consolidation & Autonomy _(Status: Planned)_

**Objective:** let the model curate its own drills, retire solved prompts, and expand the dream/critique loop without human intervention.

| Deliverable | Purpose |
|-------------|---------|
| Autonomous curriculum engine | Use reflection diaries + museum analytics to propose new drills and retire stale ones, keeping the avatar embodied in-house while planning. |
| Honesty & confidence gating | Tablet and fused head report provenance + confidence for every response. |
| Museum analytics | Mine deprecated artifacts for error signatures and retro-train corrections. |
| Continuous tool adaptation | Benchmark external tool usage; prioritise open-source replacements where feasible. |

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
