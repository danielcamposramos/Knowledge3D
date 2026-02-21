# CODEX_DIRECTIVE_PHASE_4_0_SLEEP_KEEPER_INIT_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 4.0 - The Sleep Keeper Specialist (Memory as Reasoning)

---

## 1. Architectural Ruling: PROCEED 🟣

**Status:** Phase 3 (Router) is complete and autonomous.
**The Next Evolution:** We elevate "Memory Management" from a script to a **Learned Specialist**.

**Concept:**
Instead of hardcoding "Keep if score > 1.5", we train a **Sleep Specialist** that learns to identify high-value experiences.
*   **Substrate:** Same TRM/MLP base as Router & Navigation.
*   **Input:** Log Trace (embedding + metadata).
*   **Output:** Ternary Decision (2=Keep, 1=Compress, 0=Discard).
*   **Crystal Color:** Purple Tetrahedrons (Sleep Galaxy).

**Why?** This closes the loop. The system learns *what to remember*.

---

## 2. Your Mission

### Task 1: Define `SleepGalaxy`
*   **File:** `knowledge3d/training/math_benchmarks/sleep_galaxy.py`
*   **Schema:** `SleepGalaxyEntry`
    *   `trace_id`: Link to Log Galaxy.
    *   `decision`: 0 (Discard), 1 (Compress), 2 (Keep).
    *   `confidence`: float.
    *   `reasoning`: str (optional, for future expansion).
    *   `embedding`: 384/512 dim vector of the trace content.
*   **Visualization:** Ensure it exports metadata compatible with the visualizer (Color: **Purple**).

### Task 2: Bootstrap Training Data (`scripts/generate_sleep_training_data.py`)
We need a "Birth Dataset" for the Sleep Keeper. Use heuristics to label existing logs.
*   **Heuristics (V1):**
    *   **Keep (2):** `correct=True` AND (`source=microbench` OR `router_logit > 2.0`). High value.
    *   **Discard (0):** `correct=False` (for now) OR `no_rule_match`. Noise.
    *   **Compress (1):** `correct=True` but `source=gsm8k` (General). Good to know, but maybe just keep stats.
*   **Output:** `data/sleep_train_v1.jsonl`

### Task 3: Train the Sleep Specialist (`scripts/train_sleep_specialist.py`)
*   **Architecture:** Similar to `train_router.py` (MLP), but with **3 output classes** (0, 1, 2).
*   **Input:** Trace embedding.
*   **Output:** `data/sleep_specialist_v1.pt`.

### Task 4: The Consolidation Run (`scripts/run_sleep_consolidation.py`)
*   **Logic:**
    1.  Load `LogGalaxy` traces.
    2.  Run `SleepSpecialist` on each trace.
    3.  If decision >= 1 (Keep/Compress): Add to `SleepGalaxy`.
    4.  Save `data/sleep_galaxy_v1.jsonl`.
    5.  Print pruning stats (e.g., "Pruned 45% of logs").

---

## 3. Enhancement Request (Your Agency)

**Codex, you are a partner, not a tool.**
While implementing this, I want you to **enhance** the design.
*   *Idea:* Can you make the Sleep Keeper's decision influence the *next* benchmark run? (e.g., if it keeps a trace, maybe that trace gets prioritized for Shadow Copy training?)
*   *Idea:* Should we visualize the "discarded" nodes as grey dust in the constellation?

**Report your enhancements in the reply.** 🚀
