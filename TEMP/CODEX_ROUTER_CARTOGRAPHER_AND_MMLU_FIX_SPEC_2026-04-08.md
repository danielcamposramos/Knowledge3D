# Codex Direction: Router Cartographer + MMLU Fix + Math Knowledge

**Date:** 2026-04-08
**Authority:** docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md §4.3, §10
**Depends on:** Swarm dispatch wired ✅, micro-specialist pool live ✅
**Addresses:** MMLU collapses to GENERAL (0/20), Math at 5% (1/20)

---

## Part 1 — Immediate Bug Fix: MMLU GENERAL Collapse

### Root Cause

`_infer_query_mode` in `knowledgeverse.py` around line 4170:

```python
if declared_surface == "QUESTION":
    if self._looks_like_choice_payload(payload, options):
        return "MMLU_TASK"
    return "GENERAL_TASK"   # ← every MMLU task lands here
```

`_looks_like_choice_payload` checks only for `payload.get("options")`. MMLU format uses
`payload.get("choices")`. So `bool(choice_list)` is always `False` for MMLU → GENERAL_TASK.

### Fix A — Trust the WINE Envelope (Primary Fix)

The WINE layer (`question_wine.py`) already emits `type="QUESTION_TASK"`. The envelope is
the authoritative source. `_infer_query_mode` must trust it rather than override it.

In `_infer_query_mode`, add this at the TOP, before any heuristics:

```python
# Trust the WINE envelope — it already contains the correct surface kind.
# Only run heuristics when the envelope provides no type signal.
if declared_mode in {
    "ARC_TASK", "MATH_TASK", "QUESTION_TASK", "LHE_TASK",
    "MMLU_TASK", "CHAT_TASK", "GRAMMAR_TASK", "GAME_2D",
}:
    return declared_mode
```

This is the correct architecture per KNOWLEDGEVERSE_SPECIFICATION §4.3:
> "Same path for ARC frames, GSM8K word problems, IMO proofs, MMLU questions, user chat.
> The ONLY thing that differs is the I/O adapter that normalizes the external format."

The I/O adapter (WINE envelope) already did its job. Respect it.

### Fix B — Widen `_looks_like_choice_payload` (Fallback Defense)

For tasks that arrive without a typed envelope (legacy paths), also check `choices` and
`answers` fields, which are common formats across datasets:

```python
@staticmethod
def _looks_like_choice_payload(
    task: dict[str, Any] | None,
    options: list[str] | None = None,
) -> bool:
    payload = dict(task or {})
    # Check all known choice-list field names across dataset formats
    for field in ("options", "choices", "answers", "candidates", "alternatives"):
        value = payload.get(field)
        if isinstance(value, list) and value:
            return True
    if options:
        return True
    return False
```

### Fix C — QUESTION surface must never fall to GENERAL

No `QUESTION` surface task should ever produce `GENERAL_TASK`. The distinction between
MMLU (multiple choice) and LHE (multi-hop open) is fine-grained routing — both are still
question tasks. Change:

```python
# BEFORE:
if declared_surface == "QUESTION":
    if self._looks_like_choice_payload(payload, options):
        return "MMLU_TASK"
    return "GENERAL_TASK"   # ← wrong

# AFTER:
if declared_surface == "QUESTION":
    if self._looks_like_choice_payload(payload, options):
        benchmark_hint = " ".join(
            str(payload.get(k, "")).lower()
            for k in ("competition", "benchmark", "dataset", "subject", "domain_hint")
        )
        if "lhe" in benchmark_hint or "logic" in benchmark_hint:
            return "LHE_TASK"
        return "MMLU_TASK"
    return "LHE_TASK"   # open-ended question: multi-hop, not general
```

---

## Part 2 — Architectural Target: Router Cartographer (§10)

The `_infer_query_mode` function (55 lines of Python if-else routing) is the exact
violation identified in KNOWLEDGEVERSE_SPECIFICATION §4.3:

> "CRITICAL: The Knowledgeverse has ONE universal input path for ALL queries.
> There are ZERO `if task_type ==` branches in the hot path."

The sovereign replacement is the **Router Cartographer** — a GPU-side component that
derives task routing from the input embedding and the Meta-Navigation Galaxy (R2).

### Router Cartographer Architecture (from spec §10)

The Router Cartographer lives in Region 2 (Galaxy Universe). It is a meaning star in the
Meta-Navigation Galaxy that the TRM queries at the start of every task:

```
Input query embedding
        ↓
Meta-Navigation Galaxy (R2) — nearest routing star via LED-A*
        ↓
Routing star's behavior_rpn → emit task_type signal (float → integer code)
        ↓
TRM dispatches halting weights and decomposer based on that signal
```

The routing signal is a float in the CAS pool: 1.0=MATH, 2.0=QUESTION, 3.0=GAME_2D, etc.
`OP_SEMANTIC_RESOLVE` can bind a routing code symbol → integer.

### Router Cartographer Stars (Boot-time Seeded)

Add to the boot-time seeding sequence (alongside sas_grammar_bootstrap):

```python
# knowledge3d/cranium/kernels/router_cartographer_bootstrap.py

ROUTING_STARS = [
    MeaningCentricStar(
        star_id="routing:task_type:math",
        meaning_class="routing_signal",
        domain="meta_navigation",
        galaxy_ref="Grammar",          # lives in Grammar / Meta-Navigation
        meaning_rpn="MATH_QUERY_PATTERN",  # embedding prototype for math queries
        behavior_rpn="1.0",            # routing code: MATH
        taxonomy_refs=["routing", "task_type", "math"],
    ),
    MeaningCentricStar(
        star_id="routing:task_type:question",
        meaning_class="routing_signal",
        domain="meta_navigation",
        galaxy_ref="Grammar",
        meaning_rpn="QUESTION_QUERY_PATTERN",
        behavior_rpn="2.0",            # routing code: QUESTION
        taxonomy_refs=["routing", "task_type", "question"],
    ),
    MeaningCentricStar(
        star_id="routing:task_type:spatial",
        meaning_class="routing_signal",
        domain="meta_navigation",
        galaxy_ref="Grammar",
        meaning_rpn="SPATIAL_QUERY_PATTERN",
        behavior_rpn="3.0",            # routing code: GAME_2D / spatial
        taxonomy_refs=["routing", "task_type", "spatial"],
    ),
]
```

The `meaning_rpn` for each is the embedding centroid of that task class, learned from
sleep-time consolidation. At first boot they are placeholder prototypes — they improve as
the system runs and the shadow copy records correct routing decisions.

### Migration Path for `_infer_query_mode`

Do NOT delete `_infer_query_mode` now — it is still needed as boot-time fallback.
The migration is:

1. **Now (Part 1 above)**: Make it trust the WINE envelope first, never override typed tasks
2. **Phase C (daemon)**: Router Cartographer stars seeded at boot; `_infer_query_mode`
   consults Router Cartographer embedding result when envelope has no type
3. **Phase D**: `_infer_query_mode` reduced to a 5-line fallback that calls
   `router_cartographer_query(embedding)` for untyped input only
4. **Phase E**: `_infer_query_mode` deleted entirely; routing is 100% GPU

Each phase shrinks `knowledgeverse.py` toward the ~200-line Python target.

---

## Part 3 — Math Score Improvement: Grammar Knowledge

Math at 1/20 (5%) means routing is correct (MATH=20 in the report) but Grammar Galaxy
only has 7 algebraic rules. Math competition problems need richer knowledge.

### Seed Domain Grammar Rules at Boot

Add to `sas_grammar_bootstrap.py` (or a new `math_grammar_bootstrap.py`):

```python
# Arithmetic rules
("division_inverse",       "a / b", "a * (1/b)"),
("subtraction_as_addition","a - b", "a + (-1 * b)"),
("distributive_mul",       "a * (b + c)", "(a * b) + (a * c)"),
("distributive_div",       "(a + b) / c", "(a / c) + (b / c)"),

# Fraction rules
("fraction_simplify",      "(a * c) / (b * c)", "a / b"),
("fraction_add_same_denom","(a / c) + (b / c)", "(a + b) / c"),

# Exponent rules
("power_product",          "a^m * a^n", "a^(m+n)"),
("power_quotient",         "a^m / a^n", "a^(m-n)"),
("power_of_power",         "(a^m)^n", "a^(m*n)"),

# Numeric identities
("double_negation",        "-(-a)", "a"),
("square_sqrt",            "sqrt(a^2)", "a"),   # assumes a >= 0
```

These are MEANING-named rules. Each maps to a CAS pattern + template in Grammar Galaxy.
`gre_atomic_fission_fusion` decomposes the problem; `gre_temporal_reasoning` applies
these rules in sequence via `OP_RULE_SELECT + OP_CONTEXTUAL_REWRITE`.

### GSM8K as the Math Benchmark Path

Math competitions (IMO, AMC-AIME) require multi-step derivations. GSM8K word problems
(elementary arithmetic reasoning) are better calibrated to what the current 7+N Grammar
rules can solve. Run GSM8K as the primary math benchmark for now:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/gsm8k.py --max-tasks 20 \
  --summary-output /tmp/gsm8k_r1_summary.json
```

Expected score after Part 1 + Part 3: 2-5/20 (10-25%). The fission decomposer handles
the simple arithmetic chain; temporal_reasoning sequences via CAS.

---

## Tests

```bash
# MMLU fix: no task with QUESTION surface should route to GENERAL
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_mmlu_routing_fix.py

# Choice payload detection: choices/options/answers all recognized
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_choice_payload_detection.py

# Router cartographer stars seeded at boot
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_router_cartographer_boot.py

# Benchmark runs
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/mmlu.py --max-tasks 20 \
  --summary-output /tmp/mmlu_r2_summary.json

bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/gsm8k.py --max-tasks 20 \
  --summary-output /tmp/gsm8k_r1_summary.json
```

`test_mmlu_routing_fix.py` must verify:

1. Task with `type="QUESTION_TASK"` → `_infer_query_mode` returns `"QUESTION_TASK"` (trusts envelope)
2. Task with `surface_kind="QUESTION"`, `choices=["A","B","C","D"]` → returns `"MMLU_TASK"`
3. Task with `surface_kind="QUESTION"`, no choices → returns `"LHE_TASK"` (not GENERAL)
4. Task with `surface_kind="QUESTION"`, `options=["A","B","C","D"]` → returns `"MMLU_TASK"`
5. Zero MMLU tasks in 20-task run return `route_family="GENERAL"`

`test_router_cartographer_boot.py` must verify:

1. Grammar Galaxy contains `routing:task_type:math` star after boot
2. Grammar Galaxy contains `routing:task_type:question` star after boot
3. Grammar Galaxy contains `routing:task_type:spatial` star after boot
4. `behavior_rpn` of each star is parseable as a float routing code

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ROUTER_MMLU_REPORT_2026-04-08.md` with:

1. Fix A confirmed: `declared_mode` trusted from envelope (file + line)
2. Fix B confirmed: `_looks_like_choice_payload` now checks `choices`/`options`/`answers`
3. Fix C confirmed: QUESTION surface never produces GENERAL_TASK
4. Router Cartographer stars seeded at boot (yes/no)
5. New Grammar rules count (7 original + N new)
6. MMLU result: `tasks=20, correct=K, score=X%` (expected: non-zero, target 3-8/20)
7. GSM8K result: `tasks=20, correct=K, score=X%` (expected: 2-5/20)
8. Math competitions result if re-run (for comparison)
9. All tests passing: command + count

---

## What NOT to Do

- Do NOT delete `_infer_query_mode` — it is still the fallback until Phase C
- Do NOT add Python arithmetic (eval, int(), float()) in the evaluation path to boost scores
- Do NOT run competition math benchmarks (IMO, AMC-AIME) as the primary metric — GSM8K first
- Do NOT benchmark-name any routing stars — they are `routing:task_type:*`, not `mmlu_router`
- Do NOT touch PID 400282 (encyclopedia ingest)
- Do NOT implement Phase D TRM unification in this pass
