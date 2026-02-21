# CODEX_DIRECTIVE_PHASE_3_2_ROUTER_GALAXY_INIT_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 3.2 - Initialization of the Router Galaxy

---

## 1. Victory & Transition 🏆

**Status:** Phase 3.1 is **COMPLETE**.
The "Paired Benchmark" results are definitive:
*   **Separation:** 100% clean split between Calculus (Microbench) and General (GSM8K).
*   **Safety:** The router successfully gates the Specialist, preventing it from hallucinating on word problems.
*   **Baseline:** GSM8K accuracy improved (0.79% → 1.50%) simply by routing to the General solver.

**The Pivot:** We now transition from "Static Gating" (MLP) to "Dynamic Learning" (Galaxy). We must create the memory structure that allows the router to learn from its own existence.

---

## 2. Your Mission: Build the Router Galaxy 🌌

We need a dedicated Galaxy to store routing *experiences*—not just logs, but crystallized training examples for the future TRM Specialist.

### Task 1: Create `knowledge3d/cranium/router_galaxy.py`
Implement a sovereign Galaxy class that manages routing memory.
*   **Data Structure:** `RouterExperience`
    *   `problem_text`: The input (for re-embedding).
    *   `decision_logit`: What the router thought.
    *   `ground_truth_domain`: "calculus" | "general" (derived from dataset source or successful solver).
    *   `outcome`: Did the chosen path succeed? (bool).
    *   `timestamp`: For temporal weighting.
*   **Capabilities:**
    *   `add_experience(...)`: Ingest a result.
    *   `save/load`: JSONL persistence.
    *   `export_training_data(...)`: Convert experiences into `(text, label)` pairs for the `train_router` scripts.

### Task 2: Integrate into `run_sovereign_math_benchmarks.py`
*   **Hook:** In `_record_correct_solve` (and potentially `_log_failure_detail`), feed the `RouterGalaxy`.
*   **Logic:**
    *   If `microbench` (Calculus) was solved → Add experience `(text, label=1)`.
    *   If `gsm8k` (General) was solved by Template/Word → Add experience `(text, label=0)`.
    *   *Note:* Only record **High Confidence** outcomes (solved problems or known dataset sources). Don't learn from failures yet.

### Task 3: The "Memory Formation" Run
Execute the paired benchmark again, but this time **populate the galaxy**.
*   **Command:**
    ```bash
    python3 scripts/run_sovereign_math_benchmarks.py \
      --datasets gsm8k microbench \
      --calc-microbench data/calculus_microbench.jsonl \
      --max-problems 200 \
      --router-model data/router_v1.pt \
      --router-galaxy-out data/router_galaxy_v1.jsonl
    ```

---

## 3. Contextual constraints

*   **Sovereignty:** No external DBs. JSONL is our database.
*   **Alignment:** This `RouterGalaxy` will eventually feed the `train_router_from_ollama_data.py` (TRM Specialist) and `train_router.py` (MLP), closing the loop.
*   **Observation:** We are giving the router "Long Term Memory".

**Codex, build the memory palace.** 🧱
