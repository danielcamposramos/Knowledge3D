# Theorem Extraction Investigation Report (01.05.2026)

## Scope
Validate dual-purpose theorem patterns (semantic routing + procedural RPN) and
wire them into TRM navigation for MATH benchmarks.

## What Worked
1) Extraction
- Single-step theorem patterns extracted from books_v5_clean2.
- Extracted patterns: integration_by_parts, quotient_rule, product_rule, pythagorean_identity.

2) Loading
- Role patterns loaded: 132 from books_v5_clean2.
- Theorem patterns loaded: 4 from books_v5_clean2.

3) Navigation
- TRM matched theorem patterns by semantic tags.
- Log evidence shows repeated matches for quotient_rule, product_rule, etc.

## Evidence (Logs)
Log file: /tmp/math_bench_books_theorem_integrated.log

Loaded patterns:
- "Loaded 132 role patterns from books_v5_clean2"
- "Loaded 4 theorem patterns from books_v5_clean2"
- "Theorem patterns: ['integration_by_parts', 'quotient_rule', 'product_rule', 'pythagorean_identity']"
- "Built 4 theorem grammar rules"

Matching (examples):
- "[TRM] matched theorem patterns: ['theorem:quotient_rule']"
- "[TRM] matched theorem patterns: ['theorem:pythagorean_identity']"
- "[TRM] matched theorem patterns: ['theorem:product_rule']"

Execution skips (examples):
- "[TRM] theorem theorem:quotient_rule skipped (unsupported opcodes: ['PUSH_F', 'DERIVATIVE', ...])"
- "[TRM] theorem theorem:pythagorean_identity skipped (unsupported opcodes: ['PUSH_SIN_THETA', 'POW2', ...])"

## What Failed
- Benchmark accuracy stayed at 1.00% (1/100), unchanged from role-only run.
- Theorem pattern execution did not run due to unsupported symbolic opcodes.

## Root Cause (Architectural Gap)
Theorem patterns are symbolic templates (PUSH_F, DERIVATIVE, PUSH_G).
The numeric RPN engine expects concrete numeric tokens (PUSH_3, ADD, MULT, etc).
Missing layer: symbolic binding or instantiation of theorem templates into numeric RPN.

## Architectural Options
Path A: Symbolic Binding Layer (complex)
- Parse problem to extract symbolic functions (f, g, h).
- Bind placeholders to concrete expressions.
- Execute instantiated RPN.

Path B: Direct Numeric Templates (simpler)
- Use theorem patterns for routing only.
- Map to existing numeric grammar templates (already executable).
- Keep theorem patterns as semantic index, not execution programs.

## Outcome
The dual-purpose architecture is validated for routing and matching.
Execution gap is confirmed and now isolated to a symbolic binding layer.

## Files Touched
- knowledge3d/cranium/math_galaxy_population.py
- knowledge3d/training/math_benchmarks/trm_galaxy_reader.py
- scripts/run_sovereign_math_benchmarks.py
- tests/integration/test_theorem_extraction.py
- knowledge3d/training/arc_agi/*.py (compat shims)

## Tests Run
- pytest tests/integration/test_theorem_extraction.py -v

## Decision Point
Need to choose Path A (symbolic binding) vs Path B (routing-only + numeric templates).
