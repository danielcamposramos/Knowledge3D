# Week 21.3 Architecture-Fixed Validation Metrics (Codex)

Date: 2026-02-09
Run env: `/K3D/Knowledge3D.local/envs/k3d-cranium`
Command family: `scripts/run_all_benchmarks.py` with contrastive + validity gates + fuzzy oracle + PTX ranking flags

## 1) Full Validation (100 ARC / 100 Math / 50 LHE)
Output:
- `../Knowledge3D.local/results/week21_3_architecture_fixed/week14_benchmark_summary.json`
- `../Knowledge3D.local/results/week21_3_architecture_fixed/arc_agi_2_enriched.json`
- `../Knowledge3D.local/results/week21_3_architecture_fixed/arc_agi_2_empty_mind.json`

### Topline
- ARC empty mind: `0.32`
- ARC enriched: `0.28`
- Math empty mind: `0.00`
- Math enriched: `0.3333`
- LHE empty mind: `0.50`
- LHE enriched: `1.00`

### ARC enriched diagnostics
- `generated_pattern_total`: `686`
- `tasks_with_generated_patterns`: `100/100`
- `oracle_at_all`: `0.0`
- `oracle_at_3`: `0.0`
- `oracle_at_10`: `0.0`
- `generation_failure_rate`: `1.0`
- `ranking_change_rate`: `0.36`
- `fuzzy_oracle_at_all`: `0.05`
- `oracle_fuzzy_0_80`: `0.31`
- `oracle_fuzzy_0_85`: `0.20`
- `oracle_fuzzy_0_90`: `0.12`
- `oracle_fuzzy_0_95`: `0.05`
- `validity_reject_rate_mean`: `0.7967`
- `family_rejects_mean`: `1.05`

### ARC PTX status in full validation
- `ptx_ranking_enabled_rate`: `1.0`
- `ptx_ranking_used_rate`: `0.0`
- `ptx_ranking_error_rate`: `1.0`
- Error signature (first run): missing `cuda_fp16.h` during CuPy NVRTC compile.

### Continuity/lazy evidence
- `run.log` contains `214` occurrences of `[GALAXY LAZY] Computing ... missing embeddings`.

## 2) CUDA Header Fix Verification
I validated that CuPy compiles if runtime sets:
- `CUDA_PATH=/usr`

Proof command:
- `CUDA_PATH=/usr ... python -c "import cupy as cp; x=cp.arange(...); ..."` works.

## 3) PTX Kernel Repro After Header Fix
After `CUDA_PATH=/usr`, PTX ranking still does not execute because PTX module load fails:
- Repro: `PTX_OPS.sample_dialogue_token(...)`
- Error: `CUDA_ERROR_INVALID_PTX: a PTX JIT compilation failed`
- Direct `cp.RawModule(path='knowledge3d/cranium/ptx/dialogue_sampler.ptx')` also fails with same error.

### ARC enriched-only rerun with `CUDA_PATH=/usr`
Output:
- `../Knowledge3D.local/results/week21_3_ptx_cuda_path_enriched_only_arc100.json`

Metrics:
- ARC accuracy: `0.28`
- generated patterns: `686`
- oracle_at_all: `0.0`
- `ptx_ranking_used_rate`: `0.0`
- `ptx_ranking_error_rate`: `1.0`
- PTX error now consistently: `CUDA_ERROR_INVALID_PTX`

## 4) Source-quality split (enriched)
From `arc_agi_2_enriched.json`:
- `contrastive_anti`: `16/44` (`0.3636`)
- `autonomous_generation`: `6/23` (`0.2609`)
- `legacy_pipeline`: `6/33` (`0.1818`)

Interpretation:
- Contrastive anti-pattern source is strongest among current sources.
- Ranking still cannot surface true correct candidates (`oracle_at_all=0.0`) under exact oracle.

## 5) Current Root Causes (ordered)
1. PTX execution path not active in ARC ranking due invalid PTX module (`dialogue_sampler.ptx`), even after fixing CUDA header path.
2. Candidate-level lazy embedding still appears heavily (`GALAXY LAZY` 214) in ARC path, violating strict preloaded-universe expectation.
3. Summary file (`week14_benchmark_summary.json`) omits deep ARC diagnostics currently present in `arc_agi_2_*.json`, creating visibility gap.

## 6) Recommended next pass (targeted)
1. **Fix PTX module compatibility first**
   - Regenerate/replace `knowledge3d/cranium/ptx/dialogue_sampler.ptx` for this driver/toolchain (RTX 3060, driver 550.163.01).
   - Add startup self-test to fail-fast if PTX JIT fails and include exact kernel name in benchmark summary.
2. **Eliminate candidate-level lazy embedding in ARC legacy path**
   - Precompute once, fail-fast on misses in benchmark runtime.
3. **Promote ARC diagnostic block into unified summary**
   - Copy `oracle_diagnostics` and PTX rates into `week14_benchmark_summary.json` for one-file governance.
4. **Then rerun full Week 21.3**
   - Expectation: PTX used rate > 0, GALAXY LAZY near 0, oracle/fuzzy lift measurable.

---
This run confirms the architecture fixes helped instrumentation and persistence, but the remaining unlock is now precise: PTX kernel compatibility + lazy-embedding removal + summary visibility merge.
