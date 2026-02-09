# CODEX Week 21.3 Architecture-Fixed Full Validation (02.09.2026)

## Scope
Validated benchmark architecture after:
- PTX JIT ranking integration (`ARC_PTX_OPS`)
- Benchmark lazy-embedding policy wiring (`compute|skip|fail`)
- Conda runtime aligned to Debian policy (`/home/daniel/miniforge` + `/K3D/Knowledge3D.local/envs/k3d-cranium`)

## Commands Executed

### Full validation run (100/100/50)
```bash
source /home/daniel/miniforge/etc/profile.d/conda.sh
PYTHONPATH="/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D" \
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
python scripts/run_all_benchmarks.py \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-lhe-questions 50 \
  --arc-enable-contrastive-learning \
  --arc-enable-validity-gates \
  --arc-enable-fuzzy-oracle \
  --arc-fuzzy-oracle-threshold 0.95 \
  --arc-enable-ptx-ranking \
  --arc-embedding-lazy-mode skip \
  --output-dir ../Knowledge3D.local/results/week21_3_architecture_fixed_full \
  --storage-root ../Knowledge3D.local
```

### Lazy-event proof probes
- `skip` mode probe: `../Knowledge3D.local/results/week21_3_architecture_fixed_lazyprobe`
- `fail` mode probe: `../Knowledge3D.local/results/week21_3_architecture_fixed_failprobe`

## Primary Artifacts
- Summary: `../Knowledge3D.local/results/week21_3_architecture_fixed_full/week14_benchmark_summary.json`
- ARC enriched: `../Knowledge3D.local/results/week21_3_architecture_fixed_full/arc_agi_2_enriched.json`
- ARC empty: `../Knowledge3D.local/results/week21_3_architecture_fixed_full/arc_agi_2_empty_mind.json`
- Usage log: `../Knowledge3D.local/logs/benchmark_usage_metrics.jsonl`
- History log: `../Knowledge3D.local/benchmarks/run_all_benchmarks_history.jsonl`

## Results (Full Run)

### Core benchmark scores
- ARC-AGI 2: empty `0.32` vs enriched `0.28` (paradox still present)
- Math: empty `0.00` vs enriched `0.3333`
- LHE: empty `0.50` vs enriched `1.00`

### Historical delta (enriched vs previous enriched)
- ARC: `0.25 -> 0.28` (`+0.03`, improvement)
- Math: `0.40 -> 0.3333` (`-0.0667`, regression)
- LHE: `1.00 -> 1.00` (maintained)

### ARC oracle/ranking diagnostics (enriched)
- `oracle_at_all`: `0.0`
- `oracle_at_3`: `0.0`
- `fuzzy_oracle_at_all`: `0.05`
- `generation_failure_rate`: present in diagnostics (non-zero)
- `validity_reject_rate_mean`: `0.7966666666666665`
- `ptx_ranking_enabled_rate`: `1.0`
- `ptx_ranking_used_rate`: `1.0`
- `ptx_ranking_error_rate`: `0.0`
- `ranking_change_rate`: `0.34`

### ARC generation volume
- `generated_pattern_total`: `686`
- `tasks_with_generated_patterns`: `100`

### Pattern source accuracy (enriched)
- `legacy_pipeline`: `7/35 = 0.20`
- `contrastive_anti`: `15/42 = 0.3571`
- `autonomous_generation`: `6/23 = 0.2609`

## Architecture/Sovereignty Validation

### Confirmed fixed
1. PTX ranking path is active and stable in full run:
   - `ptx_ranking_used_rate = 1.0`
   - `ptx_ranking_error_rate = 0.0`
2. No lazy embedding recomputation under policy enforcement:
   - In `skip` probe log:
     - `[GALAXY LAZY]` count: `0`
     - `Computing missing embeddings` count: `0`
   - In `fail` probe log:
     - `[GALAXY LAZY]` count: `0`
     - `Computing missing embeddings` count: `0`
     - run completes (no lazy dependency)

### Still blocked
- Oracle remains locked (`oracle_at_all = 0.0`) despite high generation volume and active PTX ranking.
- Empty > enriched paradox remains for ARC (`0.32 > 0.28`).

## Interpretation
1. **Generation is no longer zero** (686 patterns), so the original “no generation” blocker is cleared.
2. **Ranking execution is no longer broken** (PTX used 100%, no PTX errors).
3. **Primary bottleneck moved to candidate validity/matching quality**:
   - High reject rate (~0.80)
   - Very low fuzzy oracle (0.05)
   - Exact oracle still zero.

## Targeted Next Pass (Codex recommendation)
1. Tighten train-pair family inference + consistency gates with explicit per-task diagnostics in summary (shape/palette/object/family rates currently absent as `None`).
2. Increase contrastive winner prior weight where source precision is already strongest (`contrastive_anti > autonomous > legacy` in enriched run).
3. Add validity-gate calibration sweeps (strict/medium/relaxed) and report oracle/fuzzy/accuracy triples per sweep.
4. Add stratified fuzzy oracle bins (`0.80/0.85/0.90/0.95/1.00`) directly in top-level summary table for rapid iteration decisions.
5. Keep lazy mode default at `skip` (or `fail` in CI validation profile) to preserve sovereignty and continuity.

