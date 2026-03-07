# Claude Architecture Directive: Benchmark 0% Diagnosis and Fix

**Date:** March 6, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Lead)
**Priority:** CRITICAL -- This blocks all benchmark validation

---

## Diagnosis: Why Benchmarks Show 0%

### Previous Results Were Not Real

| Benchmark | Reported Result | How It Was Achieved |
|-----------|----------------|---------------------|
| LHE 100% | 4/4 correct | 4 hardcoded synthetic questions (lines 123-163 of `benchmarks/last_humanity_exam.py`), not the real 2,500-question HLE corpus |
| Math 33% | ~4 correct | Either answer extraction from GSM8K hash (lines 133-146 of `run_sovereign_math_benchmarks.py`) or matching against synthetic-count entries |
| ARC 28-32% | Variable | Adaptive oracle ranking with inconsistent pattern discovery |

The current 0% is the honest baseline against real external data. This is not a regression -- it's the first truthful measurement.

### Three Compounding Failures in the Current Code Path

**Failure 1: `require_ptx_query = True` blocks all Galaxy queries without GPU**

`galaxy_manager.py:45`: `self.require_ptx_query = _env_true("K3D_REQUIRE_PTX_QUERY", "true")`
`galaxy_manager.py:86-93`: If true, routes to `_query_ptx_implementation` which requires CuPy
`galaxy_manager.py:132-136`: Raises `NotImplementedError` if CuPy unavailable

`run_all_benchmarks.py:1017`: Forces `os.environ["K3D_REQUIRE_PTX_QUERY"] = "true"`

Result: On a machine without CuPy, every `manager.query()` call raises `NotImplementedError`. The MathSpecialist catches this at line 331 and returns `"grammar_query_failed"`. Every single problem fails before it even looks at the Galaxy.

**Every test file sets `K3D_REQUIRE_PTX_QUERY=false` via monkeypatch.** This means tests pass but benchmarks fail -- the test environment and the benchmark environment have different behavior.

**Failure 2: Forward/backward routing not connected to benchmark entry point**

The `NavigatorSpecialist` has `plan_routes()` with forward/backward/fusion/auto strategies (implemented in Weeks 18-19.6). But the benchmark runner calls the navigator differently -- it goes through `navigate_and_compose()` or `execute()` which may not invoke the full forward/backward path for every benchmark task.

The forward/backward logic is Daniel's key insight: problems present data in different orders. "Given a=5, b=3, calculate a+b" vs "Calculate a+b where a=5, b=3". Both should work. Without this, the coefficient extraction step in `MathSpecialist._extract_coefficients()` may fail on problems phrased in unexpected order.

**Failure 3: MathSpecialist bootstrap entries are narrow**

The bootstrap templates cover:
- Linear equations (ax+b=c and variants)
- Basic arithmetic (+, -, *, /)
- Ratios and proportions

But GSM8K is word problems: "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every remaining egg at the farmer's market daily for $2. How much in dollars does she make every day at the farmer's market?"

The MathSpecialist's `_infer_problem_type()` maps everything to basic pattern types (arithmetic_add, linear_equation, etc.) -- but the coefficient extraction (`_extract_coefficients`) expects clean numeric expressions, not word problems. GSM8K requires:
1. Forward/backward reading to parse the narrative
2. Multi-step decomposition (16 - 3 - 4 = 9 eggs, 9 * 2 = $18)
3. Chain RPN composition, not single-template substitution

---

## What Daniel Expects

Daniel's words:

> "Basic math rules should be present at every system load. K3D is not a run -- a task is a run. With basic math rules, math benchmarks should be a no brainer."

> "Recall the forward/backward logic -- parse requests both ways to craft strong start points."

This means:

1. **Math knowledge is FOUNDATIONAL** -- it loads at system init, like the Drawing Galaxy or Character Galaxy. It's not something the system "learns" from benchmarks. It's born knowing arithmetic, algebra, ratios, proportions.

2. **K3D is always-on.** A benchmark task arrives, gets processed, result exits. K3D doesn't start/stop per task. The Galaxy is always populated, the TRM is always ready.

3. **Forward/backward parsing is standard.** Every incoming request gets dual-parsed. This is not optional or experimental -- it's the fixed formula for strong start points.

4. **Sleep-time compute learns AFTER tasks complete.** The task itself doesn't teach K3D basic math. Sleep-time consolidation refines patterns from accumulated experience. But the basics must already be there.

---

## Fix Directive for Codex

### Fix 1: Galaxy Query Must Work (IMMEDIATE)

The `require_ptx_query` gate needs a graceful fallback. When CuPy is unavailable, the query should fall through to the token-matching implementation (which already exists at lines 94-121 of `galaxy_manager.py`).

The current code has TWO query implementations side by side:
- `_query_ptx_implementation` (GPU, requires CuPy)
- Token-matching at lines 94-121 (CPU, always works)

The fix is NOT to remove sovereignty. The fix is: **try PTX first, fall back to token-matching if CuPy is unavailable.** This is exactly what sovereign execution means -- the system runs on what's available, preferring GPU when present.

```
If CuPy available:     GPU query (sovereign PTX)
If CuPy unavailable:   Token-matching query (still sovereign -- no external ML, just string matching on Galaxy entries)
```

Both paths are sovereign. Neither uses numpy/sklearn/torch. The token-matching path is slower but correct. The benchmark runner should NOT force `K3D_REQUIRE_PTX_QUERY=true` -- it should let the system auto-detect.

**Critical:** `run_all_benchmarks.py:1017` must be changed from forcing `true` to letting the system auto-detect GPU availability.

### Fix 2: Foundational Math Bootstrap at System Load (IMMEDIATE)

The `MathSpecialist._ensure_bootstrap_templates()` is called in `__init__`, which is correct. But the bootstrap entries are too narrow for GSM8K word problems.

**What must be present at every system load (foundational, not learned):**

**A. Arithmetic operations** (already present -- add, subtract, multiply, divide)

**B. Multi-step decomposition grammar** (MISSING -- this is the critical gap)

GSM8K word problems require chaining operations. The Grammar Galaxy needs rules for:
- Sequential computation: "result_1 = step1; result_2 = step2(result_1); answer = step3(result_2)"
- As RPN: `{step1_args} OP1 {step2_args} OP2 {step3_args} OP3` -- this is what RPN was built for
- Word problem keywords -> operations mapping (from chain code `MathProceduralizer`):
  - "per day/each/every" -> MULTIPLY
  - "remaining/left/left over" -> SUBTRACT
  - "total/altogether/combined" -> ADD
  - "split/shared/divided among" -> DIVIDE
  - "how much/how many" -> signals the ANSWER variable

**C. Variable binding from natural language** (MISSING)

The coefficient extraction assumes clean equations ("2x + 3 = 7"). Word problems need entity extraction:
- "Janet's ducks lay 16 eggs" -> eggs_per_day = 16
- "She eats three for breakfast" -> eaten = 3
- "bakes muffins with four" -> baked = 4
- "sells remaining" -> remaining = eggs_per_day - eaten - baked
- "$2 each" -> price = 2
- "how much" -> answer = remaining * price

This is the forward/backward reading in action:
- **Forward:** Parse entities as they appear -> build variable map -> identify question -> compose RPN
- **Backward:** Identify question first ("how much does she make") -> identify what's needed (remaining * price) -> parse backwards for values -> compose RPN

**D. Answer comparison** (VERIFY THIS EXISTS)

GSM8K answers are formatted as `#### 18`. The benchmark adapter must extract the numeric answer from K3D's RPN result and compare it to the expected format. This is a Tablet boundary responsibility -- format translation, not solving logic.

### Fix 3: Forward/Backward at the Benchmark Entry Point (NEXT)

The forward/backward routing in `NavigatorSpecialist` is implemented but may not be invoked for benchmark tasks. Ensure:

1. Benchmark task arrives at Tablet boundary
2. Tablet sends to NavigatorSpecialist (not directly to MathSpecialist)
3. NavigatorSpecialist applies `plan_routes()` with forward/backward/fusion/auto
4. Best route reaches MathSpecialist with pre-parsed structure
5. MathSpecialist receives structured input (variables + question), not raw text

This is Daniel's point about the fixed formula: forward/backward is the STANDARD first step, not an optimization applied later.

### Fix 4: Word Problem Grammar Rules (FOUNDATIONAL BOOTSTRAP)

Add to the foundational bootstrap (loaded at init, not learned):

```
Grammar Galaxy entries needed:
- word_problem_sequential: "Extract entities, map keywords to ops, chain as RPN"
- word_problem_rate: "per/each/every -> MULTIPLY"
- word_problem_remainder: "remaining/left -> SUBTRACT from total"
- word_problem_total: "total/altogether -> ADD all"
- word_problem_split: "split/divided among -> DIVIDE"
- word_problem_comparison: "more/less/difference -> SUBTRACT"
- word_problem_percentage: "percent/% -> MULTIPLY by fraction"

Math Galaxy templates needed:
- multi_step_chain: "{step1} {step2_op} {step3_op}" (variable-length RPN chain)
- rate_calculation: "{quantity} {rate} *"
- remainder_calculation: "{total} {used_1} - {used_2} -"
- percentage_calculation: "{base} {percent} 100 / *"
```

These are not exotic -- they're grade-school math. Every system load should have them. Sleep-time compute can then discover MORE patterns from benchmark experience, but these basics must be born-in.

### Fix 5: Foundational Bootstrap Expansion (PRIORITY)

Currently `foundational_operations_bootstrap.py` has geometric transforms (rotate, mirror, transpose) for ARC. It needs equivalent coverage for math:

The existing `_grammar_entries()` and `_math_entries()` functions in `foundational_operations_bootstrap.py` should be expanded with the word problem patterns above. This is the RIGHT place for foundational knowledge -- loaded at system init for every session.

---

## Implementation Order

```
1. Fix Galaxy query fallback (galaxy_manager.py)           -- 0% -> queries work
2. Fix benchmark runner env var (run_all_benchmarks.py)     -- queries reach Galaxy
3. Expand foundational bootstrap (foundational_operations_bootstrap.py + math_specialist.py)
                                                            -- Galaxy has math knowledge
4. Wire forward/backward to benchmark entry point           -- robust parsing
5. Add word problem coefficient extraction                  -- GSM8K entity extraction
6. Run benchmark, measure real baseline                     -- honest number
```

Steps 1-2 are trivial fixes (< 30 minutes). Steps 3-5 are the real work but use existing patterns (bootstrap entries, forward/backward routing, MathProceduralizer from chain code).

---

## Success Criteria

- Galaxy query works without CuPy (fallback to token-matching)
- MathSpecialist solves `7 * (3 + 2) = ?` at system init (no learning needed)
- MathSpecialist solves "If John has 5 apples and buys 3 more, how many does he have?" (word problem)
- Forward/backward parsing handles both "Given X, find Y" and "Find Y where X"
- GSM8K accuracy >= 30% (target from CODEX.md backlog)
- All fixes are sovereign (no numpy/sklearn in hot path)
- Sleep-time compute can still learn new patterns from benchmark runs AFTER the foundational knowledge is present

---

## Daniel's Architecture Principle (Reiterated)

K3D is an always-on system. It is born with foundational knowledge (Drawing, Character, Word, Grammar, Math galaxies). A benchmark task is an external event that enters through the Tablet, gets processed by the sovereign pipeline, and exits with an answer. K3D doesn't start for the task or stop after it. The task is a run. K3D is not.

Basic math is like basic literacy -- you don't learn to read the letter 'A' during a reading test. The Character Galaxy has 'A' at system load. The Math Galaxy must have addition at system load. Sleep-time compute refines and discovers, but the foundation is always there.
