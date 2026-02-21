# CODEX_DIRECTIVE_PHASE_4_4_THE_WAKING_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 4.4 - The Waking (Sleep-Guided Evolution)

---

## 1. The Cycle of Life 🌅

**Status:** Phase 4.3 is complete. The system sleeps, dreams, and sorts memories into Purple (Wisdom) and Crimson (Warning).
**The Missing Link:** We have consolidated memory, but we haven't *used* it to improve the system yet.
**The Goal:** **The Waking**. The system must emerge from sleep smarter than it entered. We will train **Navigation Specialist V5** using *only* the refined curriculum created by the Sleep Keeper.

**FMEAI Alignment:**
*   *Experience (Logs)* -> *Consolidation (Sleep Keeper)* -> *Wisdom (Sleep Galaxy)* -> *Evolution (New Specialist)*.

---

## 2. Your Mission

### Task 1: The Dream Exporter (`scripts/wake_from_sleep.py`)
We need to convert `SleepGalaxy` entries back into training data for the Navigation Specialist.
*   **Input:** `data/sleep_galaxy_v3.jsonl`.
*   **Logic:**
    *   **Filter:** Select entries where `decision=2` (Keep).
    *   **Branching:**
        *   **Purple (Positive):** Extract `(problem, solution)` -> **SFT Dataset** (`wake_train_v5.jsonl`).
        *   **Crimson (Negative):** Extract `(problem, wrong_path)` -> **Anti-Curriculum** (`wake_anti_v5.jsonl`). (Save this for future DPO/Unlikelihood training, just export it for now).
*   **Output:** Two JSONL files.

### Task 2: Evolution (`scripts/train_navigation_from_wake.py`)
Create a streamlined training script (or wrapper around `train_adaptive_swarm.py`) that fine-tunes the **Navigation Specialist**.
*   **Source:** Load `checkpoints/navigation_specialist_v4.pt` (or V3) as the base.
*   **Dataset:** Train on `wake_train_v5.jsonl`.
*   **Output:** `checkpoints/navigation_specialist_v5.pt`.

### Task 3: The Morning Run
Validate the new specialist.
*   **Command:**
    ```bash
    python3 scripts/run_sovereign_math_benchmarks.py \
      --calc-microbench data/calculus_microbench.jsonl \
      --equip-skill checkpoints/navigation_specialist_v5.pt \
      --router-model data/router_v3.pt
    ```
*   **Expectation:** V5 should maintain 100% accuracy on the microbench, proving that "Sleeping" preserved the critical skills while pruning noise.

---

## 3. Success Criteria

*   `wake_train_v5.jsonl` contains only the high-quality traces.
*   `navigation_specialist_v5.pt` is created.
*   Benchmark confirms V5 is functional (The system woke up healthy).

**Codex, wake the swarm.** ☀️
