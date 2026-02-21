# CODEX_DIRECTIVE_PHASE_4_5_DATA_CLEANSING_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 4.5 - The Purge (Data Cleansing & V6)

---

## 1. Architectural Ruling 🧹

**Status:** Phase 4.4 (V5) produced a model that predicts `honest`/`hallucination`.
**Ruling:** This is **Data Leakage**, not Self-Reflection. The model is parroting metadata it shouldn't see.
**Action:** We must **PURGE** these tokens from the training pipeline to ensure the model focuses on *math*, not *metadata*.

---

## 2. Your Mission

### Task 1: Clean the Pipeline (`scripts/wake_from_sleep.py`)
Modify the script to rigorously filter reserved tokens during dataset generation.
*   **Reserved Tokens:** `{"honest", "hallucination", "heuristic", "unclear", "mixed"}` (and any variants).
*   **Logic:** When extracting the "Ground Truth Sequence" from a Sleep Galaxy trace, iterate through the steps. If a step *is* a reserved token, **skip it**. Do not include it in the target sequence.
*   **Constraint:** Ensure the remaining sequence is still valid (contiguous rule chain).

### Task 2: Regenerate Data (The Clean Wake)
Run the improved script to create a pristine dataset.
*   **Input:** `data/sleep_galaxy_v3.jsonl` (same source).
*   **Output:** `data/wake_positive_v2_cleaned.jsonl`.
*   **Verification:** Run `grep -iE "(honest|hallucination|heuristic)" data/wake_positive_v2_cleaned.jsonl` to prove they are gone.

### Task 3: Train V6 (The Pure Specialist)
Train the **Navigation Specialist V6** on the cleaned data.
*   **Source:** `checkpoints/navigation_specialist_v4.pt` (Base).
*   **Dataset:** `data/wake_positive_v2_cleaned.jsonl`.
*   **Output:** `checkpoints/navigation_specialist_v6.pt`.

### Task 4: Validate V6
Run the microbench with V6.
*   **Check:** Does it still solve the problems? (Accuracy 100%?).
*   **Check:** Are the status tokens gone from the output trace?

---

## 3. Success Criteria

*   `wake_positive_v2_cleaned.jsonl` contains **ZERO** instances of reserved tokens in the `target_sequence`.
*   `navigation_specialist_v6.pt` is created.
*   Benchmark traces show clean rule sequences (e.g., `[start -> differentiate -> simplify -> end]`) without `[honest]` interruptions.

**Codex, purify the mind.** 🛁
