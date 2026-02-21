# CODEX_DIRECTIVE_PHASE_4_6_VISUALIZATION_AND_STANDARDIZATION_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 4.6 - Standardization & The Clean View

---

## 1. Architectural Validation 🏛️

**Status:** The Purge was successful. V5 Clean is verified.
**The Pivot:** We cannot have two "V5s" (Dirty and Clean) confusing the swarm. We must **Standardize** on the clean artifact and **Visualize** the new, pure reasoning paths.

---

## 2. Your Mission

### Task 1: Standardization (The Replacement)
We are committing to the clean timeline.
*   **Archive:** Move `navigation_specialist_v5.pt` (the dirty one) to `Old_Attempts/checkpoints/`.
*   **Promote:** Rename `navigation_specialist_v5_wake_clean.pt` to `checkpoints/navigation_specialist_v5.pt` (The new Standard).
*   **Skill Galaxy:** Update `data/skill_galaxy_v5.jsonl` to point to the clean weights (or rename `v5_wake_clean.jsonl` to `v5.jsonl`).

### Task 2: Visualizing the Pure Mind (`scripts/analyze_experience.py`)
Run the analysis on the *clean* logs (`log_galaxy_neural_v5_clean.jsonl`).
*   **Goal:** Generate the visual trace of the system's reasoning.
*   **Output:** Ensure output files are named clearly (e.g., `analysis_v5_clean/`).
*   **Check:** Verify visually (via the report/output text) that "honest" tags are indeed absent from the *Policy* visualization, even if they remain in the *Reward* logic.

### Task 3: The Constellation Update
Regenerate the 3D Constellation to reflect the new, clean Skill V5.
*   **Inputs:** `v3`, `v4`, `v5` (Clean), `router_v3` (Orange), `sleep_v3` (Purple).
*   **Output:** `viewer/public/skill_galaxy_constellation.gltf`.

---

## 3. Success Criteria

*   The "Dirty V5" is gone/archived.
*   The system uses "Clean V5" by default when V5 is requested.
*   The Analysis Report confirms clean reasoning traces.
*   The 3D Constellation is up to date.

**Codex, finalize the timeline.** ⚓
