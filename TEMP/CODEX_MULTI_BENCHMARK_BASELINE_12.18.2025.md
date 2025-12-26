# CODEX Multi-Benchmark Baseline (Books + TTC)

**Date:** 2025-12-18  
**Seed:** 123  
**Max problems:** 200 per dataset  
**Mode:** `--use-trm-navigator --disable-retrieval --thinking-budget 8 --shadow-readonly --load-all-galaxies --enable-book-galaxies`

## Summary Results

| Dataset | Accuracy | Notes |
|---|---:|---|
| GSM8K | 82/200 = **41.00%** | Retrieval disabled for clean comparison |
| MATH | 5/200 = **2.50%** | Still mostly failing on symbolic/geometry |
| Omni-MATH | 0/200 = **0.00%** | Many problems have no numeric ground truth |
| AMC-AIME | 1/200 = **0.50%** | Mostly symbolic/geometry/number theory gaps |

## Failure Breakdown (from run logs)

- **GSM8K:** `{'no_rule_match': 18, 'wrong_computation': 46, 'multi_step_needed': 33, 'word_problem': 21, 'algebra_needed': 0, 'unknown': 0}`
- **MATH:** `{'no_rule_match': 24, 'wrong_computation': 114, 'multi_step_needed': 12, 'word_problem': 12, 'algebra_needed': 33, 'unknown': 0}`
- **Omni-MATH:** `{'no_rule_match': 26, 'wrong_computation': 104, 'multi_step_needed': 24, 'word_problem': 40, 'algebra_needed': 6, 'unknown': 0}`
- **AMC-AIME:** `{'no_rule_match': 44, 'wrong_computation': 71, 'multi_step_needed': 27, 'word_problem': 30, 'algebra_needed': 27, 'unknown': 0}`

## Logs

- `/tmp/gsm8k_baseline_200_seed123.log`
- `/tmp/math_baseline_200_seed123.log`
- `/tmp/omni_math_baseline_200_seed123.log`
- `/tmp/amc_aime_baseline_200_seed123.log`

## Book Galaxies Available (K3D_LOCAL_DIR)

Root: `/K3D/Knowledge3D.local/galaxies/books`

| Book ID | Domain | Pages | Templates | Title |
|---|---|---:|---:|---|
| `advanced_calculus` | `calculus` | 200 | 1820 | Advanced Calculus |
| `la_done_right` | `linear_algebra` | 200 | 81 | Linear Algebra Done Right |
| `areavol` | `geometry` | 2 | 4 | AreaVol |
| `numbersets` | `number_theory` | 2 | 9 | Number Sets |
| `physquantities` | `units` | 2 | 32 | Physical Quantities |
| `dmoi3` | `discrete_math` | 200 | 0 | Discrete Math (DmoI3) |
| `mathgems` | `competition_math` | 2 | 0 | Math Gems |
| `transition_v104` | `algebra` | 260 | (no template index) | Transition v104 |

## Commands Used

```bash
# All runs (dataset substituted), books enabled
K3D_LOCAL_DIR=/K3D/Knowledge3D.local bash scripts/k3d_env.sh run python3 scripts/run_sovereign_math_benchmarks.py \
  --use-trm-navigator --disable-retrieval --datasets <DATASET> --max-problems 200 \
  --shuffle --shuffle-seed 123 --thinking-budget 8 --shadow-readonly \
  --load-all-galaxies \
  --enable-book-galaxies --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books \
  --book-max-books 16 --book-top-k 5 --verbose \
  2>&1 | tee /tmp/<DATASET>_baseline_200_seed123.log
```

## Immediate Takeaways

- Book ingestion is working mechanically (artifacts load, template indices exist), but cross-benchmark accuracy is still low: the current template extractor is mostly `lhs = rhs` and doesn’t yet capture many theorem-style / LaTeX-heavy patterns (especially in `dmoi3` and `transition_v104`).
- For MATH/AMC, the dominant failures are `wrong_computation` + `algebra_needed` + `no_rule_match`, indicating we still need stronger symbolic bindings and more robust template extraction from book text (still generic; not benchmark-specific).

