# CODEX: Sovereign Math Galaxy - Test & Benchmark

**Date:** December 13, 2025
**Priority:** HIGH - Validate the sovereign approach
**Partner:** Claude (Architecture) → Codex (Implementation)

---

## Status

✅ Math Symbol Galaxy created
✅ Sovereign Composer implemented
✅ Pipeline wired to use Galaxy
✅ Legacy preprocessing deprecated

---

## Task 1: Sanity Test Sovereign Composer

```bash
PYTHONPATH=. python3 -c "
from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
from knowledge3d.training.math_benchmarks.sovereign_composer import SovereignComposer
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

# Test Galaxy lookup
print('=== Galaxy Lookup ===')
for sym in ['\\\\frac', '\\\\binom', '!', '^', '\\\\sqrt']:
    entry = MATH_GALAXY.lookup(sym)
    if entry:
        print(f'{sym:10s} → {entry.rpn_template}')

print()

# Test Composer
print('=== Sovereign Composer ===')
composer = SovereignComposer()
engine = ModularRPNEngine()

tests = [
    ('\\\\frac{24}{4}', 6.0),
    ('\\\\binom{10}{3}', 120.0),
    ('5!', 120.0),
    ('2^10', 1024.0),
    ('\\\\sqrt{16}', 4.0),
    ('3+4*2', 11.0),
    ('10%3', 1.0),
]

for expr, expected in tests:
    rpn_str = composer.compose(expr)
    # Parse RPN string to tokens
    tokens = []
    for tok in rpn_str.split():
        try:
            tokens.append(float(tok))
        except ValueError:
            tokens.append(tok)

    if tokens:
        result = engine.evaluate(tokens)
        status = '✓' if abs(result - expected) < 0.01 else '✗'
        print(f'{status} {expr:20s} → RPN: {rpn_str:30s} → {result} (expected {expected})')
    else:
        print(f'✗ {expr:20s} → NO RPN GENERATED')
"
```

---

## Task 2: Full Benchmark Run

After sanity tests pass:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/train_math_benchmarks.py --epochs 1 2>&1 | tee /tmp/sovereign_benchmark.log
```

Report:
1. Overall accuracy
2. Per-dataset breakdown (gsm8k, math, omni_math, amc_aime, mmlu)
3. Any errors or issues

---

## Expected Results

With sovereign Galaxy composition:
- GSM8K should remain ~90% (word problem rules still work)
- MATH/AMC-AIME should improve (LaTeX symbols now have meanings)
- No more "external preprocessing" - pure Galaxy → RPN → GPU

---

## Success Criteria

| Test | Expected |
|------|----------|
| `\frac{24}{4}` | 6.0 |
| `\binom{10}{3}` | 120.0 |
| `5!` | 120.0 |
| `2^10` | 1024.0 |
| `\sqrt{16}` | 4.0 |

---

**Codex:** Run sanity tests, then full benchmark. Report scores.
