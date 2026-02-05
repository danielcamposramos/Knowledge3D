# CODEX_DIRECTIVE_PHASE_3_ROUTER_INTEGRATION_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 3.1 - Router Observability & Benchmark Execution

---

## 1. Context & Objective

We are in **Phase 3.1: The Router/Gatekeeper**.
The infrastructure is in place:
- ✅ `data/router_v1.pt`: Trained binary classifier (Calculus vs General).
- ✅ `scripts/run_sovereign_math_benchmarks.py`: Supports `--router-model` loading and gating.

**Goal:** Execute the "Gated Benchmark" to prove that the router correctly prevents the Calculus Specialist from failing on GSM8K word problems, improving overall system reliability.

**Current Gap:**
The current `solve_problem_with_meta` logic uses the router but **swallows the decision**. We cannot analyze *why* the router gated a problem or verify its precision/recall because the logit/confidence isn't stored in the trace.

---

## 2. Your Mission

### Step 1: Enhance Observability (The "Enhancement")
Modify `scripts/run_sovereign_math_benchmarks.py` (inside `solve_problem_with_meta`) to:
1.  **Capture the Router's Output:** Store the `logit` and the resulting boolean decision (`use_specialist`).
2.  **Inject into Trace:** Add these fields to the `trace` dictionary returned by the solver.
    *   e.g., `trace["router_decision"] = {"logit": 2.5, "gated_specialist": True}`
3.  **Log High-Value Events:** If the router triggers the specialist (logit > 0), print a concise log message (so we can see it working in real-time).

**Why:** This data is required for the "Phase 3.1 Completion Report" to analyze gating effectiveness.

### Step 2: Execute the Benchmark
Run the benchmark on GSM8K with the router enabled.
*   **Command:** `python3 scripts/run_sovereign_math_benchmarks.py --dataset gsm8k --router-model data/router_v1.pt --max-problems 200`
*   *Note:* Start with a subset (200 problems) to verify the pipeline before a full run.

### Step 3: Analyze & Report
Create `TEMP/PHASE_3.1_COMPLETION_REPORT.md` containing:
1.  **Gating Stats:** How many problems did the router send to the specialist? (Expected: Very few for GSM8K).
2.  **Accuracy Delta:** Compare the result to the 0.79% baseline (Specialist-only). With the router, it should be significantly higher (because it falls back to General solvers).
3.  **Failure Analysis:** Did the router block any *actual* calculus problems? (If any exist in the sample).

---

## 3. Constraints

*   **Do NOT** modify `train_router_from_ollama_data.py`. That is for Phase 3.2.
*   **Do NOT** modify `train_router.py`. The model `router_v1.pt` is already trained.
*   **Sovereignty:** Use existing `embed_text` and PyTorch logic. No new dependencies.

**Codex, you have the helm.** Enhance the runner, fire the engine, and show us the metrics. 🚀
