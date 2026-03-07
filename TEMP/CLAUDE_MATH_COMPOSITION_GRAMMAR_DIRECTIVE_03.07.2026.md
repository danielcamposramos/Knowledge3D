# Claude Architecture Directive: Math Composition Grammar (Corrected)

**Date:** March 7, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Lead)
**Supersedes:** CLAUDE_BENCHMARK_DIAGNOSIS_AND_FIX_DIRECTIVE_03.06.2026.md (contained sovereignty violations)

---

## Corrections from Previous Directive

The March 6 directive contained three errors that Daniel caught:

1. **CuPy is NOT sovereign.** I referred to CuPy as a valid PTX query path. Wrong. `import cupy` is an external library dependency -- exactly the same violation as numpy. The `_query_ptx_implementation` in `galaxy_manager.py` that uses CuPy matrix operations (`matrix_gpu.dot(query_gpu)`) is NOT PTX. It's a library pretending to be sovereign. The token-matching path (lines 94-121) is the actual sovereign query path until a real PTX kernel replaces it via ctypes.

2. **Problem-by-problem solving is wrong.** The proposed next step of "attack remaining Math 1/20 failure modes problem by problem" creates deterministic problem-specific solutions. This is an impossible feat for 8,000+ GSM8K problems and fundamentally misunderstands K3D. The TRM must COMPOSE solutions from grammar primitives using reasoning. That's what it's built for.

3. **Number-Word symlinks are missing.** "5" and "five" are the same meaning in different forms. The Number entries must symlink to Word Galaxy entries and vice versa -- exactly as Characters symlink to Drawing primitives. This is the Save Information Principle applied to numerics.

---

## CuPy Sovereignty Violation: Scope

`galaxy_manager.py:16` imports CuPy at module level. This module is hot path -- it's where Galaxy queries happen during inference. This must be flagged and removed.

Current CuPy usage in `galaxy_manager.py`:
- `cp.asarray()` -- CuPy array creation
- `matrix_gpu.dot(query_gpu)` -- CuPy matrix multiplication
- `cp.argpartition()` -- CuPy sorting
- `cp.asnumpy()` -- CuPy to numpy conversion

None of this is PTX. It's CuPy (which wraps CUDA Runtime API, not raw PTX). The sovereign replacement is either:
- The token-matching path (already exists, already sovereign, lines 94-121)
- A real PTX kernel loaded via ctypes for embedding-based query (future work)

**Action:** Remove CuPy from `galaxy_manager.py`. The token-matching query IS the query path. When embedding-based query is needed, it gets a real PTX kernel, not a library wrapper.

---

## The Real Problem: Math Composition Grammar

### What "Basic Math Knowledge" Means

Daniel said: basic math should be a no-brainer at every system load.

This does NOT mean:
- A template for every possible word problem
- Pattern matching on specific question phrasings
- Deterministic mappings from problem text to answers

This DOES mean:
- The Grammar Galaxy contains COMPOSITIONAL RULES for mathematical reasoning
- Numbers, operations, and relationships exist as Galaxy entries with symlinks
- The TRM navigates and composes these to construct solutions
- Any problem expressible in basic arithmetic/algebra is solvable by composition

### Compositional Grammar vs Template Matching

**Current approach (WRONG -- template matching):**
```
"What is 7 * (3 + 2)?" -> match template "arithmetic_multiply" -> "{a} {b} *" -> done
"Janet's ducks lay 16 eggs..." -> match template ??? -> FAIL (no template for this)
```

**Required approach (CORRECT -- compositional grammar):**
```
Grammar Galaxy contains:
  - RULE: "quantity_per_unit" -> "{quantity} {unit_count} MUL"
  - RULE: "consume_from_total" -> "{total} {consumed} SUB"
  - RULE: "chain_result" -> apply rules sequentially, stack carries intermediate
  - RULE: "answer_is_final_stack" -> top of stack after all rules applied

TRM composes for "Janet's ducks lay 16 eggs, eats 3, bakes 4, sells rest at $2":
  1. Navigate -> "quantity_per_unit" matches "16 eggs per day" -> 16 PUSH
  2. Navigate -> "consume_from_total" matches "eats 3" -> 3 SUB (stack: 13)
  3. Navigate -> "consume_from_total" matches "bakes 4" -> 4 SUB (stack: 9)
  4. Navigate -> "quantity_per_unit" matches "sells at $2" -> 2 MUL (stack: 18)
  5. Navigate -> "answer_is_final_stack" -> result: 18
```

The difference: templates are FLAT (one problem -> one template). Composition is SEQUENTIAL (TRM chains multiple grammar rules into an RPN program). RPN is BUILT for this -- the stack naturally carries intermediate results.

### Number Galaxy and Symlinks

The Save Information Principle says: don't duplicate, reference.

**Number Galaxy entries (foundational, loaded at init):**

Every number is a Galaxy entry with dual representation:
```
{
  "id": "num_5",
  "name": "5",
  "domain": "number",
  "category": "integer",
  "rpn_program": "5 PUSH",
  "metadata": {
    "value": 5,
    "word_ref": "word_five",           <- symlink to Word Galaxy
    "ordinal_ref": "word_fifth",       <- symlink to ordinal form
    "char_refs": ["char_u0035"],       <- symlink to Character Galaxy (digit '5')
    "roman_ref": "word_v",             <- symlink to roman numeral form
    "forms": ["5", "five", "fifth", "V", "cinco", "funf"]
  }
}
```

**Word Galaxy entries must cross-reference back:**
```
{
  "id": "word_five",
  "name": "five",
  "domain": "word",
  "category": "lexeme",
  "rpn_program": "WORD five TOKEN",
  "metadata": {
    "char_refs": ["char_u0066", "char_u0069", "char_u0076", "char_u0065"],
    "number_ref": "num_5",             <- symlink back to Number Galaxy
    "is_numeric_word": true
  }
}
```

This means when the TRM encounters "five" in text, it can navigate to the Number Galaxy and get the value 5 for computation. When it encounters "5" in a formula, it can navigate to the Word Galaxy for natural language context. Same meaning, different forms -- the Dual Client Contract in action.

### Mathematical Operation Grammar (Compositional)

These are not templates. They are GRAMMAR RULES that the TRM composes:

**Arithmetic Operations (foundational -- loaded at init):**
```
Grammar: "addition"
  rpn_program: "{a} {b} ADD"
  keywords: ["add", "plus", "sum", "more", "together", "combined", "total", "and"]
  inverse: "subtraction"

Grammar: "subtraction"
  rpn_program: "{a} {b} SUB"
  keywords: ["subtract", "minus", "less", "fewer", "difference", "remaining", "left"]
  inverse: "addition"

Grammar: "multiplication"
  rpn_program: "{a} {b} MUL"
  keywords: ["multiply", "times", "product", "each", "per", "every", "of"]
  inverse: "division"

Grammar: "division"
  rpn_program: "{a} {b} DIV"
  keywords: ["divide", "split", "share", "among", "per", "ratio", "average"]
  inverse: "multiplication"
```

**Compositional Meta-Rules (how to CHAIN operations):**
```
Grammar: "sequential_computation"
  description: "Apply operations in sequence. Stack carries intermediate results."
  rpn_program: "COMPOSE"  -- meta-rule, TRM chains sub-rules
  pattern: "result of step N feeds into step N+1"

Grammar: "consume_from_total"
  description: "Remove quantity from running total"
  rpn_program: "{consumed} SUB"
  pattern: "eats/uses/spends/loses {quantity}"

Grammar: "accumulate_to_total"
  description: "Add quantity to running total"
  rpn_program: "{gained} ADD"
  pattern: "earns/gains/receives/finds {quantity}"

Grammar: "rate_application"
  description: "Apply rate to quantity"
  rpn_program: "{quantity} {rate} MUL"
  pattern: "{quantity} at/for/per {rate}"

Grammar: "percentage"
  description: "Calculate percentage of quantity"
  rpn_program: "{base} {percent} 100 DIV MUL"
  pattern: "{percent}% of {base}"

Grammar: "comparison"
  description: "Compare two quantities"
  rpn_program: "{a} {b} SUB"
  pattern: "how much more/less/difference"
```

**Algebraic Composition Rules:**
```
Grammar: "solve_for_unknown"
  description: "Isolate variable by inverse operations"
  rpn_program: "INVERSE_COMPOSE"  -- meta-rule
  pattern: "solve for x / find x / what is x"
  method: "apply inverse of each operation to both sides"

Grammar: "equation_balance"
  description: "Both sides equal, isolate unknown"
  rpn_program: "{rhs} {known_ops_inverse} APPLY"
  pattern: "expression = value"
```

### How TRM Composes (Not Matches)

For "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every remaining egg at $2. How much does she make every day?"

**Forward reading path:**
1. TRM navigates Number Galaxy: "16" -> num_16, "three" -> (symlink from Word) -> num_3, "four" -> num_4, "$2" -> num_2
2. TRM navigates Grammar Galaxy for each sentence:
   - "lay 16 eggs per day" -> no operation yet, establish base: PUSH 16
   - "eats three" -> "consume_from_total" rule -> 3 SUB
   - "bakes ... with four" -> "consume_from_total" rule -> 4 SUB
   - "sells ... at $2" -> "rate_application" rule -> 2 MUL
   - "how much does she make" -> signals answer = top of stack
3. TRM composes RPN: `16 3 SUB 4 SUB 2 MUL`
4. PTX executes: 16 - 3 = 13, 13 - 4 = 9, 9 * 2 = 18

**Backward reading path:**
1. TRM reads "how much does she make" -> identifies GOAL: compute earnings
2. "sells ... at $2" -> need: remaining_eggs * 2
3. "remaining" -> need: total - consumed
4. "eats three" + "bakes four" -> consumed = 3 + 4
5. "lay 16 eggs" -> total = 16
6. Compose backward: earnings = (16 - 3 - 4) * 2 = 18
7. RPN: `16 3 SUB 4 SUB 2 MUL`

Both paths converge to the same RPN. The forward/backward dual-parse gives robustness -- if one path fails to extract all entities, the other may succeed.

**The TRM composed this. No template existed for "Janet's ducks." The grammar rules are REUSABLE across any problem with the same mathematical structure.**

---

## Implementation Guidance

### What Codex Must Build

1. **Number Galaxy bootstrap** -- Integers 0-1000 as Galaxy entries with Word symlinks. Foundational, loaded at init. Also common fractions (1/2, 1/3, 1/4...) and constants (pi, e). Every entry has `word_ref` symlinks.

2. **Word Galaxy number symlinks** -- Every numeric word ("one" through "thousand", "first" through "hundredth") gets a `number_ref` field pointing to the Number Galaxy entry.

3. **Compositional math grammar rules** -- Not problem templates. Grammar rules for mathematical OPERATIONS and COMPOSITION PATTERNS (sequential computation, consume/accumulate, rate application, percentage, comparison, solve-for-unknown). These are the building blocks the TRM chains.

4. **TRM composition logic in the navigator** -- The navigator must CHAIN grammar rules, not just match one. For word problems, the TRM navigates multiple grammar rules in sequence, composing an RPN program step by step. The stack carries intermediate results naturally.

5. **Remove CuPy from galaxy_manager.py** -- The token-matching query path is sovereign. CuPy is not. When GPU query is needed, build a real PTX kernel via ctypes.

### What Codex Must NOT Build

- Problem-specific templates ("if the problem mentions ducks, use this template")
- Deterministic problem-to-answer mappings
- Answer extraction from question text
- Any import of numpy/cupy/scipy in the Galaxy query hot path
- Per-benchmark special cases

### Chain Code to Leverage

`MathProceduralizer` (CODEX_SOVEREIGN_SWARM_ARCHITECTURE lines 3282-3380) has the right keyword-to-opcode mapping. But it needs to become Grammar Galaxy entries, not a Python class. The patterns become:
- `_operation_patterns` dict -> Grammar Galaxy entries with keyword metadata
- `_extract_quantities` -> Number Galaxy navigation (find numeric entities in text)
- `_build_problem_rpn` -> TRM composition (chain grammar rules into RPN)

---

## Success Criteria

1. Number Galaxy loaded at init with 0-1000 integers + Word symlinks
2. Word Galaxy has number_ref symlinks back to Number Galaxy
3. Grammar Galaxy has compositional math rules (not problem templates)
4. TRM can compose multi-step RPN from chained grammar rules
5. `16 3 SUB 4 SUB 2 MUL` is composed by the TRM, not matched from a template
6. CuPy removed from galaxy_manager.py
7. Forward/backward parsing produces the same RPN for equivalent problems phrased differently
8. GSM8K accuracy improves through COMPOSITION, not through adding more templates
9. No numpy/cupy/scipy anywhere in the hot path (Galaxy query, TRM navigation, PTX execution)

---

## The Principle

K3D is born knowing what "five" means (it's the same as 5, which is the same as V, which is char_u0035 rendered as a glyph). It's born knowing what "add" means (take two things on the stack and combine them). It's born knowing what "remaining" means (subtract what was consumed from the total).

It is NOT born knowing the answer to "Janet's ducks lay 16 eggs." It COMPOSES the answer by navigating its foundational knowledge -- Number Galaxy for values, Grammar Galaxy for operations, Word Galaxy for natural language understanding, and the RPN stack for computation.

This is what the Knowledgeverse was built for. The grammar is the reasoning. The TRM navigates it. The PTX executes it. No libraries needed.
