# Week 22 - Contrastive Math Specialist (Sovereign Path) - 02.12.2026

## Scope
- Integrated ternary contrastive forward/backward/fusion selection into the active sovereign math solve path.
- Kept execution path as: `TRMNavigator -> MathSpecialist -> RPN compose -> PTX runtime`.
- No `eval`, `sympy`, or CPU fallback logic introduced.

## Files Changed
- `knowledge3d/knowledgeverse/specialists/math_specialist.py`
- `tests/test_math_specialist.py`

## What Was Added
1. Ternary quality integration
   - `TernaryQualityMemory` wired into `MathSpecialist`.
   - State path resolution:
     - `K3D_MATH_TERNARY_STATE_PATH` override supported.
     - fallback path: `../Knowledge3D.local/checkpoints/math_specialist_ternary_quality.json`.
   - Quality prior updates on success/failure for both Grammar pattern and Math template entries.

2. Contrastive forward/backward/fusion candidate flow
   - `_fuse_contrastive_patterns(...)`
   - `_generate_anti_patterns(...)`
   - `_deduplicate_candidates(...)`
   - Candidate groups:
     - Forward positive (`prior > 0.3`)
     - Backward negative (`prior < -0.3`) -> anti-pattern generation
     - Fusion uncertain (`|prior| <= 0.3`)

3. Problem-type-aware composition (single specialist)
   - `_infer_problem_type(...)`
   - Supported bootstrap problem types:
     - `linear_equation`
     - `arithmetic_add`
     - `arithmetic_subtract`
     - `arithmetic_multiply`
     - `arithmetic_divide`

4. Forward/backward extraction for linear equations
   - `_extract_linear_coefficients_forward_backward(...)`
   - Handles both:
     - `2x + 3 = 11`
     - `11 = 2x + 3`

5. Template/bootstrap expansion
   - Grammar bootstrap entries expanded beyond linear-only.
   - Math template bootstrap entries expanded to high-yield arithmetic templates.

## Tests Added/Updated
- `test_math_specialist_linear_equation_composes_rpn`
- `test_math_specialist_backward_equation_composes_rpn` (new)
- `test_math_specialist_arithmetic_addition_template` (new)

## Validation
- `python3 -m py_compile knowledge3d/knowledgeverse/specialists/math_specialist.py tests/test_math_specialist.py` -> PASS
- `pytest -q tests/test_math_specialist.py tests/test_k3d_daemon.py` -> PASS (`7 passed`)

## Notes
- This patch targets the capability gap without reintroducing fallback drift.
- It improves sovereign candidate generation quality in the math specialist path while preserving PTX-only hot path execution.
