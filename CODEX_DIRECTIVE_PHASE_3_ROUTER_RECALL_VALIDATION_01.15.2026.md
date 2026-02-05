# CODEX_DIRECTIVE_PHASE_3_ROUTER_RECALL_VALIDATION_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 3.1 - The "Recall" Challenge (Paired Benchmark)

---

## 1. Situation Analysis

**Excellent work.** 🛡️
The router correctly identified 200/200 GSM8K problems as "Not Calculus" (Avg Logit -2.81). The "Fail-Safe" property is verified.

**The Risk:** We haven't proven the router works; we've only proven it can say "No". It might be "dead" (outputting -2.81 for *everything*).
**The Fix:** We must execute a **Paired Benchmark** (Negatives + Positives) to verify **Recall**.

---

## 2. Your Mission

### Step 1: Tooling Enhancements
*   **`scripts/run_sovereign_math_benchmarks.py`**:
    *   Add `--router-threshold` argument (float, default 0.0). Allows sweeping sensitivity without retraining.
*   **`scripts/router_report.py`**:
    *   Add **Dataset Grouping**: Report metrics (Precision/Recall/F1) *per dataset*.
    *   *Why?* We need to see "GSM8K: 0% Positive Rate" vs "Microbench: 100% Positive Rate" side-by-side.

### Step 2: The Paired Benchmark Run
Execute the benchmark on both datasets in a single run to generate a unified log.

**Command:**
```bash
# Run Mixed Benchmark (Negatives: GSM8K, Positives: Calculus Microbench)
python3 scripts/run_sovereign_math_benchmarks.py \
  --datasets gsm8k microbench \
  --calc-microbench data/calculus_microbench.jsonl \
  --max-problems 200 \
  --router-model data/router_v1.pt \
  --router-threshold 0.0 \
  --router-log-out data/router_events_paired_v1.jsonl
```
*(Note: Ensure `load_dataset` handles "microbench" correctly or use the `--calc-microbench` flag logic you already have).*

### Step 3: The Verification Report
Generate the analysis:
```bash
python3 scripts/router_report.py \
  --input data/router_events_paired_v1.jsonl \
  --output TEMP/PHASE_3.1_PAIRED_ROUTER_REPORT.md
```

---

## 3. Success Criteria (For the Report)

We are looking for **Separation**:
1.  **GSM8K:** Positive Rate < 5% (Ideally 0%)
2.  **Microbench:** Positive Rate > 90% (Ideally 100%)
3.  **Logit Delta:** Avg Logit(GSM8K) << 0 << Avg Logit(Microbench)

If we see this separation, **Phase 3.1 is Complete**.

**Execute.** 🚀

