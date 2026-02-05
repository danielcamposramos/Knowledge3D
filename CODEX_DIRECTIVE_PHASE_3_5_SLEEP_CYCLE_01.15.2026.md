# CODEX_DIRECTIVE_PHASE_3_5_SLEEP_CYCLE_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 3.5 - The Router Sleep Cycle (Automated Consolidation)

---

## 1. Context & Objective 💤

**Phase 3.4 Success:** The Router is a Skill (`skill_galaxy_router_v2.jsonl`).
**The Philosophy:** In K3D, memory consolidation happens during "Sleep". The system should wake up smarter than it went to bed.

**Goal:** Create the automated "Sleep Cycle" for the Router.
Instead of running 3-4 manual commands (`seed` -> `train` -> `package` -> `validate`), we need a single entry point that processes the day's experiences (`RouterGalaxy`) into a new, crystallized Router Skill.

---

## 2. Your Mission

### Task 1: Create `scripts/router_sleep_cycle.py`
This script orchestrates the consolidation process.

**Logic Flow:**
1.  **Identify Input:** Find the active `RouterGalaxy` (default `data/router_galaxy_v1.jsonl`).
2.  **Determine Next Version:** Check existing router skills (`skill_galaxy_router_v*.jsonl`). If `v2` exists, next is `v3`.
3.  **Training:** Call `train_router_from_galaxy.py` to produce `data/router_v{N}.pt`.
4.  **Crystallization:** Call `package_router_skill.py` to produce `data/skill_galaxy_router_v{N}.jsonl`.
    *   Metadata: `skill_id="router_gatekeeper_v{N}"`
5.  **Visualization:** Update the constellation.
    *   Call `scripts/visualize_skill_galaxy.py` with ALL skill files (v3, v4, router_v2, router_v{N}).
    *   *Note:* Ensure the visualizer output goes to `viewer/public/skill_galaxy_constellation.gltf`.

**Sovereignty:** Use `subprocess` to call the other scripts (they are robust CLI tools). Do not import their internals directly if it risks circular deps, but direct import is cleaner if feasible. Given they are scripts, `subprocess` or `runpy` logic is safer for orchestration.

### Task 2: Update Visualization Script (Minor Tweak)
*   **File:** `scripts/visualize_skill_galaxy.py`
*   **Request:** Ensure it handles the "router" keyword in filenames for color coding (maybe assign **Red** or **Orange** for routers to distinguish them from Navigation skills).
    *   Current: v1=Blue, v2=Cyan, v3=Green, v4=Magenta.
    *   Add: `if "router" in path.name: color = [1.0, 0.5, 0.0] # Orange`

### Task 3: Execute the First Sleep Cycle
Run the script to generate `router_v3` (even if it's just a clone of v2 for now, it proves the cycle).

---

## 3. Success Criteria

*   `scripts/router_sleep_cycle.py` exists and works.
*   `data/skill_galaxy_router_v3.jsonl` is created automatically.
*   `viewer/public/skill_galaxy_constellation.gltf` is updated and includes the orange Router star.

**Codex, let the system dream.** 🌙
