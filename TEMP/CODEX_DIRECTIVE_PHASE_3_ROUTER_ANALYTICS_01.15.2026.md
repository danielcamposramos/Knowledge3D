# CODEX_DIRECTIVE_PHASE_3_ROUTER_ANALYTICS_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 3.1 - Analytics Tooling & "Ghost" Gating

---

## 1. Assessment & Strategy

**Codex, your instincts are sharp.** 🧠
We accept your proposals. Manual reporting is fragile; reproducible tooling is sovereign.

You identified a critical gap: if the `recursive_solver` path is skipped (e.g., because a Template or Galaxy Read solved it first), we lose the router's signal. To measure **True Precision/Recall**, we need the router's opinion on *every* problem, even if we don't act on it.

---

## 2. Your Mission

### Task 1: "Ghost" Router Logging (The False Negative Trap)
Modify `scripts/run_sovereign_math_benchmarks.py`:
*   **Logic:** Move (or duplicate) the router inference logic *up* in `solve_problem_with_meta` so it runs for **every** problem (unless `use_router` is off).
*   **Ghosting:** If another solver wins, we still want `trace["router_decision"]` to exist.
*   **Result:** The trace should look like:
    ```json
    {
      "solver": "template",
      "router_decision": {"logit": 2.1, "would_gate_specialist": true},
      ...
    }
    ```
    *This allows us to answer: "Did the router recognize this as calculus even though the Template solver handled it?"*

### Task 2: Create `scripts/router_report.py`
Implement a sovereign analytics script (no pandas/sklearn if possible, or use standard libs).
*   **Input:** A Log Galaxy JSONL file (e.g., `data/log_galaxy_gsm8k.jsonl`).
*   **Analysis:**
    1.  **Gating Rate:** % of problems routed to Specialist.
    2.  **Confusion Matrix (Proxy):
        *   *True Positive:* Router=Spec AND Spec=Correct.
        *   *False Positive:* Router=Spec AND Spec=Wrong.
        *   *False Negative:* Router=General AND General=Wrong (but maybe Spec could have solved it? Hard to know, but log it).
    3.  **Agreement:** How often did Router=Spec align with "Source=Calculus/Algebra"?
*   **Output:** Print a Markdown summary table to stdout.

### Task 3: The Gated Run
Execute the pipeline with your new tools.
1.  **Run Benchmark:**
    ```bash
    python3 scripts/run_sovereign_math_benchmarks.py \
      --dataset gsm8k \
      --router-model data/router_v1.pt \
      --log-galaxy-out data/log_galaxy_gsm8k_gated.jsonl \
      --max-problems 200
    ```
2.  **Run Report:**
    ```bash
    python3 scripts/router_report.py --input data/log_galaxy_gsm8k_gated.jsonl > TEMP/PHASE_3.1_ROUTER_PERFORMANCE.md
    ```

---

## 3. Constraints

*   **Sovereignty:** `router_report.py` should calculate metrics using standard Python `json` and `math` where possible.
*   **Stability:** Do not break the `router_train` scripts.
*   **Focus:** We are optimizing for **Observability** now. We need to *see* the brain working.

**Execute.** 🚀
