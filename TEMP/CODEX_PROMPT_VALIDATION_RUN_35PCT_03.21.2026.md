# Codex Prompt: 35% Stratified Validation Run

**Date:** March 21, 2026
**Priority:** Measure impact of math knowledge expansion + stratified sampling
**Constraint:** One process at a time. No parallel benchmark runs.

---

## What to Run

A single benchmark run using `run_enriched_benchmarks.py` with ~35% of each dataset, using the new stratified sampler. This gives us representative scores across all difficulty levels.

### Target Counts (35% of each dataset)

| Suite | Full Size | 35% Sample | Argument |
|-------|----------|-----------|----------|
| ARC-AGI | 120 | 42 | `--arc-max 42` |
| Math | 12,500 | 500 | `--math-max 500` (existing benchmark slice) |
| GSM8K | 1,319 | 462 | `--gsm8k-max 462` |
| LHE | 100 | 35 | `--lhe-max 35` |
| MMLU | 14,042 | 4,915 | `--mmlu-max 4915` |

**Total: ~5,954 questions** — substantial enough to be meaningful, small enough to finish in a reasonable time.

### Execution

1. **Ensure no running benchmark or sleep-time process:**
   ```bash
   ps aux | grep -E 'run_enriched|benchmark_health|sleep_time' | grep -v grep
   ```

2. **Activate the right environment:**
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   conda activate k3d-cranium
   ```

3. **Run the meaning layer ingest first** (picks up new math rules + symlinks):
   ```bash
   cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
   python3 scripts/ingest_meaning_layer.py --storage-root /K3D/Knowledge3D.local 2>&1 | tail -20
   ```
   Wait for completion. Verify math rules loaded (should be 227+ rule specs, 68+ template programs, 227+ symlinks).

4. **Run the math rules ingest** (if separate from meaning layer):
   ```bash
   python3 scripts/ingest_math_rules.py --storage-root /K3D/Knowledge3D.local 2>&1 | tail -20
   ```

5. **Launch the validation run:**
   ```bash
   python3 scripts/run_enriched_benchmarks.py \
     --full \
     --storage-root /K3D/Knowledge3D.local \
     --arc-max 42 \
     --math-max 500 \
     --gsm8k-max 462 \
     --lhe-max 35 \
     --mmlu-max 4915 \
     2>&1 | tee /tmp/k3d_validation_35pct_03.21.2026.log
   ```

   If the runner doesn't support per-suite max arguments, use the global `--max-problems` or modify the runner to accept them. Check the runner's argument parser first:
   ```bash
   python3 scripts/run_enriched_benchmarks.py --help
   ```

   **Fallback if per-suite args don't exist:** Use `--max-problems 500` which will apply stratified sampling at 500 per suite. This gives us ~2,500 total (a smaller but still representative run).

6. **Monitor progress** (in another terminal):
   ```bash
   tail -f /tmp/k3d_validation_35pct_03.21.2026.log
   ```

7. **After completion, run sleep-time consolidation:**
   Let the normal post-benchmark sleep-time run. Do NOT skip it — the TRM learns from every pass.

8. **Save the run state:**
   ```bash
   cp /K3D/Knowledge3D.local/logs/health_log.full.run_state.json \
      /tmp/k3d_validation_35pct_run_state_03.21.2026.json
   ```

---

## What to Report

After the run completes, create a brief report with:

1. **Per-suite scores:**
   | Suite | Score | % | vs Previous |
   |-------|-------|---|-------------|
   | ARC | ?/42 | | (was 10/120 = 8.33%) |
   | Math | ?/500 | | (was 0/500 = 0.00%) |
   | GSM8K | ?/462 | | (was 30/1319 = 2.27%) |
   | LHE | ?/35 | | (was 8/100 = 8.00%) |
   | MMLU | ?/4915 | | (was 3272/14042 = 23.30%) |

2. **Math breakdown by type** (if available in diagnostics):
   - Algebra, Prealgebra, Number Theory, Geometry, Counting & Probability, Intermediate Algebra, Precalculus

3. **Stratified sampling verification:** Confirm that the sampled problems come from all difficulty levels (not just the first N).

4. **Sleep-time consolidation results:** Updated specialist routes, weights saved.

5. **Galaxy state:** Total entries, math rules loaded, symlinks active.

---

## Success Criteria

- Math > 0/500 (the 0-wall is broken for real with stratified sampling)
- Math shows improvement signal across multiple types (not just Algebra)
- GSM8K holds or improves (same engine, more knowledge)
- MMLU holds at ~23% (no regression from math changes)
- Run completes without crashes
- Sleep-time consolidation commits successfully
