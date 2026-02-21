# CODEX_DIRECTIVE_PHASE_4_2_NEGATIVE_KNOWLEDGE_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 4.2 - The "Anti-Pattern" & The Deep History

---

## 1. The Insight 💡

The User has provided a critical architectural correction:
> *"What does not work... must also be kept as a negative example... not all errors, but some important ones must be kept as a means of what not to do memory."

**Diagnosis:**
Our V1 Sleep Keeper (trained on V4 logs) was too optimistic ("Keep Everything"). V4 was a victory lap (100% correct).
To learn **Wisdom**, the system must remember its **Failures** (V1-V3).

**The Pivot:**
We are not just separating "Signal vs. Noise". We are separating:
1.  **Signal (Positive):** "Do this again."
2.  **Signal (Negative):** "Never do this again (but remember why)."
3.  **Noise:** "Forget this happened."

---

## 2. Your Mission

### Task 1: Enhance the Bootstrapper (`scripts/build_sleep_training_data.py`)
Modify the heuristics to identify **"Instructional Failures"**.
*   **Load History:** Accept multiple input files (`--inputs data/log_galaxy_neural_v*.jsonl`).
*   **New Heuristics:**
    *   **Keep (2) - Positive:** Correct answer.
    *   **Keep (2) - Negative:** Incorrect answer BUT `steps > 3` (It tried hard, "Noble Failure").
        *   *Tag these in metadata:* `role="negative_example"`.
    *   **Discard (0) - Noise:** Incorrect answer AND `steps <= 1` (Crash/放弃/Give up).
*   **Result:** A dataset rich in both victories and "teachable moments".

### Task 2: Train V2 Specialist
Train `sleep_specialist_v2.pt` on this expanded, balanced dataset (V1+V2+V3+V4).
*   *Note:* The ternary classification (0/1/2) remains, but the semantic meaning of "2" now includes "Important Failures".

### Task 3: The Great Consolidation
Run the new specialist on **ALL** historical logs.
*   **Command:**
    ```bash
    python3 scripts/run_sleep_specialist.py \
      --inputs data/log_galaxy_neural_v1.jsonl data/log_galaxy_neural_v2.jsonl data/log_galaxy_neural_v3.jsonl data/log_galaxy_neural_v4.jsonl \
      --model checkpoints/sleep_specialist_v2.pt \
      --output data/sleep_galaxy_v2.jsonl \
      --dust-out data/sleep_dust_v2.jsonl \
      --min-confidence 0.75
    ```

### Task 4: Visualize the Wisdom
Update `scripts/visualize_sleep_galaxy.py` to highlight the "Negative Knowledge".
*   **Logic:** If `metadata.role == "negative_example"`, color it **Red/Crimson** (Danger/Warning).
    *   Positive Keeps = Purple.
    *   Dust = Grey/Transparent.

---

## 3. Success Criteria

*   `sleep_galaxy_v2.jsonl` contains significantly more entries than v1.
*   The report (`scripts/sleep_report.py`) shows a mix of Positive and Negative keeps.
*   The visualization shows a "Purple Core" surrounded by "Red Warnings" and "Grey Dust".

**Codex, teach the system its history.** 📜
