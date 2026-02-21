# CODEX_DIRECTIVE_PHASE_4_3_STRESS_TEST_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 4.3 - The Stress Test (Forging Crimson Memories)

---

## 1. The Missing Color 🔴

**Status:** Phase 4.2 is functional but "too clean."
**The Problem:** Our logs (V1-V4) are from a victory lap. We have no "Noble Failures" to crystallize into Negative Wisdom (Crimson).
**The Goal:** We must intentionally stress the system to generate "Instructional Failures." We need traces where the system *tries hard* (high step count) but *fails*. This teaches the Sleep Keeper what "A Dead End" looks like.

---

## 2. Your Mission

### Task 1: The "Stress Run" (Generating Failure)
We will force the **Recursive Solver** (Specialist) to attempt **GSM8K** (General Problems) without the Router's protection.
*   **Command:**
    ```bash
    python3 scripts/run_sovereign_math_benchmarks.py \
      --dataset gsm8k \
      --max-problems 50 \
      --thinking-budget 10 \
      --log-galaxy-out data/log_galaxy_stress_v1.jsonl \
      --use-trm-navigator \
      --disable-retrieval
    ```
*   *Note:* By disabling retrieval and templates (if possible via flags, or just relying on `trm-navigator` preference), and giving a budget, we encourage the system to "overthink" simple problems. This should yield `correct=False` but `steps > 3` traces.

### Task 2: The Crimson Bootstrapper
Update `scripts/build_sleep_training_data.py` (if needed) to ensure it catches these new logs.
*   **Input:** `data/log_galaxy_neural_v4.jsonl` (Successes) + `data/log_galaxy_stress_v1.jsonl` (Failures).
*   **Validation:** Ensure the "Noble Failure" heuristic (`correct=False` AND `steps > 3`) actually triggers on the stress logs. (Adjust the step threshold to `> 1` if the stress run is clumsy but earnest).

### Task 3: The Third Sleep (Consolidation V3)
1.  **Train:** `sleep_specialist_v3.pt` on the mixed dataset.
2.  **Consolidate:** Run `run_sleep_specialist.py` on the Stress Log.
    *   *Expectation:* High "Keep (Negative)" rate.
3.  **Visualize:** Generate `sleep_galaxy_v3.gltf`.

---

## 3. Success Criteria

*   `log_galaxy_stress_v1.jsonl` exists and contains failures.
*   `sleep_galaxy_v3.gltf` contains **Crimson** nodes (Negative Wisdom).
*   The `sleep_report.py` confirms the presence of "Negative Examples".

**Codex, show the system the value of failure.** 🛡️
