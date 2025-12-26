# CODEX: Run Full Sovereign Benchmark

**Date:** December 14, 2025
**Priority:** HIGH - Get real performance numbers
**Partner:** Claude (Architecture) → Codex (Implementation)

---

## Status

✅ SovereignComposer works (sanity tests passed)
✅ CuPy dependency guarded
✅ Benchmark runner created
✅ Smoke test ran (5 samples - not meaningful)

---

## Task: Run Full Benchmark

Run with meaningful sample sizes:

```bash
# First: 500 samples per dataset (quick validation)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/run_sovereign_math_benchmarks.py --limit 500 2>&1 | tee /tmp/sovereign_500.log

# If that works: Full run (all samples)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/run_sovereign_math_benchmarks.py 2>&1 | tee /tmp/sovereign_full.log
```

---

## Expected Dataset Sizes

| Dataset | Approximate Size |
|---------|------------------|
| GSM8K | ~8,800 |
| MATH | ~12,500 |
| Omni-MATH | ~4,400 |
| AMC-AIME | ~1,400 |
| MMLU | ~1,000 |
| **Total** | ~28,000 |

---

## Report Format

After the run completes, report:

```
SOVEREIGN MATH BENCHMARK RESULTS
================================
Date: [date]
Mode: Pure PTX + Galaxy (No CuPy)

Per-Dataset:
  gsm8k    : XXXX/YYYY = XX.XX%
  math     : XXXX/YYYY = XX.XX%
  omni_math: XXXX/YYYY = XX.XX%
  amc_aime : XXXX/YYYY = XX.XX%
  mmlu     : XXXX/YYYY = XX.XX%

Overall: XXXX/YYYYY = XX.XX%

Notes:
- [Any errors or issues]
- [Which paths produced results: composer vs word_solver vs fallback]
```

---

## Success Criteria

1. Full benchmark completes without crashes
2. GSM8K should be ~90% (word problem rules)
3. Other datasets - we'll see the sovereign baseline

---

## If Errors Occur

If any dataset fails:
1. Note the error
2. Skip that dataset and continue
3. Report partial results

---

**Codex:** Run the 500-sample benchmark first, then full if time permits. Report results.
