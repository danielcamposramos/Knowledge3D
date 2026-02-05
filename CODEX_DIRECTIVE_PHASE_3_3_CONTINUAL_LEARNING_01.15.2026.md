# CODEX_DIRECTIVE_PHASE_3_3_CONTINUAL_LEARNING_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 3.3 - Closing the Loop (Galaxy → Weights)

---

## 1. Status & Vision 🔄

**Phase 3.2 Success:** The `RouterGalaxy` exists. It captures high-confidence runtime events.
**The Current Limitation:** The Galaxy is sparse (only 12 entries) and the Router (`router_v1.pt`) is still static, trained from a flat file.

**Phase 3.3 Objective:** We must transition to a **Self-Improving Cycle**.
1.  The Galaxy must become the *Single Source of Truth* (containing both original training data and new experiences).
2.  The Router must be trainable directly from the Galaxy.

This proves the **FMEAI Cycle**: *Atomic Cognition (Router) → Action (Benchmark) → Energetic Memory (Galaxy) → Atomic Cognition (Retraining).*

---

## 2. Your Mission

### Task 1: Seed the Memory (`scripts/seed_router_galaxy.py`)
We cannot learn from 12 examples alone. We must ingest the "Birth Knowledge" into the Galaxy.
*   **Input:** `data/router_train.jsonl` (The original bootstrap dataset).
*   **Logic:** Read the JSONL and add each entry to `RouterGalaxy` as a `RouterExperience`.
    *   Set `confidence=1.0` (since it's ground truth).
    *   Set `source="bootstrap"`.
*   **Output:** An updated `data/router_galaxy_v1.jsonl` containing ~24+ entries (Original + Microbench).

### Task 2: The Synapse Builder (`scripts/train_router_from_galaxy.py`)
Create a sovereign training script that learns from the Galaxy, not a flat file.
*   **Imports:** `RouterGalaxy`, `embed_text`, `torch`.
*   **Logic:**
    1.  Load `RouterGalaxy` from disk.
    2.  Iterate through experiences. Filter for valid/high-confidence entries.
    3.  Extract `(problem_text, label)` pairs.
    4.  Train the MLP (same architecture as `train_router.py`).
    5.  Save to `data/router_v2.pt`.
*   **Constraint:** Ensure it handles the unbalanced nature of the Galaxy (e.g., if we have 200 GSM8K and 12 Microbench, maybe weight the loss or just train as-is for now). *For now, training as-is is fine.*

### Task 3: Execute the Loop
Run the sequence to prove the cycle works.
1.  **Seed:** `python3 scripts/seed_router_galaxy.py --input data/router_train.jsonl --galaxy data/router_galaxy_v1.jsonl`
2.  **Train:** `python3 scripts/train_router_from_galaxy.py --galaxy data/router_galaxy_v1.jsonl --output data/router_v2.pt`

---

## 3. Success Criteria

*   `router_v2.pt` exists and is a valid PyTorch model.
*   The training log shows the model learning from the combined dataset (Bootstrap + Experience).
*   **Sovereignty:** No new external libraries.

**Codex, close the loop.** ♾️
