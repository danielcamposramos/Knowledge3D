# CODEX_DIRECTIVE_PHASE_4_1_THE_FIRST_SLEEP_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 4.1 - The First Sleep (Execution & Analysis)

---

## 1. Architectural Validation 🟣

**Codex, you have excelled.**
The concept of "Dust" (visualizing discarded memories) is brilliant. It makes the system's forgetting process observable and reversible. This is exactly the kind of "next-generation" feature the User envisions.

**Status:** Infrastructure Ready.
**Goal:** Wake the Sleep Keeper and let it process the memories of Phase 2/3.

---

## 2. Your Mission

### Task 1: Execution Chain (The "Deep Sleep")
Execute the pipeline you proposed, using the **V4 logs** (`data/log_galaxy_neural_v4.jsonl`) as the primary memory source.

**Sequence:**
1.  **Bootstrap:** Generate `data/sleep_train_v1.jsonl` + `data/sleep_dust_v1.jsonl`.
    *   *Source:* `data/log_galaxy_neural_v4.jsonl`.
2.  **Train:** Produce `checkpoints/sleep_specialist_v1.pt`.
3.  **Consolidate:** Run the specialist to produce `data/sleep_galaxy_v1.jsonl` (The Keepers).
4.  **Visualize:** Generate `viewer/public/sleep_galaxy_v1.gltf`.

### Task 2: The Dream Report (`scripts/sleep_report.py`)
We need to know *what* the system decided to keep. Create a sovereign reporting script.
*   **Input:** `data/sleep_galaxy_v1.jsonl` AND `data/sleep_dust_v1.jsonl`.
*   **Metrics:**
    *   **Compression Rate:** (Dust Count / Total Count).
    *   **Source Bias:** "Kept 90% of Calculus, Discarded 60% of General?"
    *   **Confidence Stats:** Avg confidence of Keep vs Discard.
*   **Output:** `TEMP/PHASE_4_1_SLEEP_REPORT.md`.

---

## 3. Success Criteria

*   The Sleep Galaxy is populated.
*   The "Dust" file exists (evidence of pruning).
*   The Report proves the specialist is making **semantic distinctions** (not just random deletion).

**Codex, execute the First Sleep.** 🌌
