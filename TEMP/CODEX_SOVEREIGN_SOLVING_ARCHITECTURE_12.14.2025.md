# CODEX: Sovereign Solving Architecture - From Extraction to True Computation

**Date:** December 14, 2025
**Priority:** CRITICAL - Fix fundamental solving architecture
**Partner:** Claude (Architecture Analysis) → Codex (Implementation + Original Ideas)

---

## Root Cause Analysis

### The "100% GSM8K" Was a Lie

**Current benchmark results:**
```
GSM8K:     100.00%  ← FAKE - extracting "#### 72" from answer text
MATH:      15.55%   ← Real (low because not actually solving)
Omni-MATH: 13.66%   ← Real
AMC-AIME:  26.90%   ← Real
MMLU:      22.98%   ← Real (now working with A/B/C/D)
```

**Evidence - GSM8K solve_problem (lines 133-146):**
```python
# GSM8K: answer text contains the numeric solution
if source == "gsm8k":
    hash_match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(solution))
    if hash_match:
        return float(hash_match.group(1))  # ← CHEATING! Just extracting answer
```

**The model isn't solving - it's copying the answer from the solution text.**

---

### Word Solver Has No Rule Chaining

**Test case:**
```
Problem: "Natalia sold clips to 48 friends in April, sold half as many in May.
          How many altogether?"
```

**What word_solver produces:**
```python
{
    'numbers': [48.0],
    'matched_rules': ['wp_division_half', 'wp_addition_total'],  # ← BOTH matched!
    'rpn_program': '48.0 2 /'  # ← But only ONE rule applied!
}
```

**Root cause - `_generate_rpn` line 90:**
```python
if "wp_division_half" in rules:
    return f"{numbers[0]} 2 /"  # ← EARLY RETURN! Never processes wp_addition_total
```

**What SHOULD happen:**
```
Step 1: "48 friends in April" → APRIL = 48
Step 2: "half as many in May" → MAY = APRIL / 2 = 24
Step 3: "altogether" → TOTAL = APRIL + MAY = 48 + 24 = 72

RPN: 48 DUP 2 / +  (or: 48 48 2 / +)
```

---

### Composer Matches English 'e' as Euler's Number

**Test output:**
```python
composer.compose("Natalia sold clips to 48...")
# Returns: '48.0 2.71828182845905 2.71828182845905 2.71828182845905...'
```

Every 'e' in "Natalia sold clips to her friends" becomes Euler's constant!

---

## The User's Key Insight

> "The steps are a model decision, so maybe we now only need to iterate on
> the model logic (weights) learning"

**Current approach (hardcoded):**
- Grammar rules define patterns
- Python code decides WHICH rules to apply
- Python code decides the ORDER
- Python code composes the RPN

**Correct approach (learned):**
- Galaxy symbols define WHAT operations exist
- Grammar rules define HOW to recognize patterns
- **MODEL WEIGHTS learn:**
  - Which rules to apply to which text spans
  - The order of rule application
  - How to chain results from one rule into another
  - How to compose the final RPN program

**The parsing strategy should be LEARNED, not hardcoded.**

---

## Immediate Fixes Needed

### Fix 1: Remove Answer Extraction Cheating

**File:** `scripts/run_sovereign_math_benchmarks.py`

Remove lines 133-146 (GSM8K answer extraction) and lines 176-181 (fallback extraction).
The model should SOLVE, not extract.

```python
# REMOVE THIS:
if source == "gsm8k":
    hash_match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(solution))
    if hash_match:
        return float(hash_match.group(1))  # ← DELETE

# REMOVE THIS:
# Try 3: Parse solution if available (fallback)
numbers = re.findall(r"[-+]?\d*\.?\d+", str(solution))  # ← DELETE
```

### Fix 2: Word Solver Rule Chaining

**File:** `knowledge3d/training/math_benchmarks/word_problem_solver.py`

Replace `_generate_rpn` with multi-rule composition:

```python
def _generate_rpn(self, numbers: List[float], rules: List[str]) -> str:
    """
    Generate RPN by CHAINING multiple rules, not picking one.

    Key insight: Rules should compose, not compete.
    """
    if not numbers:
        return ""

    # Build computation graph from rules
    rpn_parts = []
    base = numbers[0]

    # Check for transformations (half, double, etc.)
    if "wp_division_half" in rules:
        # "half as many" means: take base, divide by 2
        # But we also need the original for "altogether"
        if "wp_addition_total" in rules:
            # Pattern: X + X/2 = X * 1.5 or equivalently: X DUP 2 / +
            return f"{base} DUP 2 / +"
        return f"{base} 2 /"

    if "wp_multiplication_twice" in rules or "wp_multiplication_double" in rules:
        if "wp_addition_total" in rules:
            # X + 2X = 3X
            return f"{base} DUP 2 * +"
        return f"{base} 2 *"

    if "wp_percentage_of" in rules and len(numbers) >= 2:
        pct, base_val = numbers[0], numbers[1]
        return f"{base_val} {pct} * 100 /"

    if "wp_addition_total" in rules:
        # Sum all numbers
        rpn_parts = [str(numbers[0])]
        for n in numbers[1:]:
            rpn_parts.append(str(n))
            rpn_parts.append("+")
        return " ".join(rpn_parts)

    # ... rest of rules ...

    return " ".join(str(n) for n in numbers)
```

### Fix 3: Composer 'e' Guard

**File:** `knowledge3d/training/math_benchmarks/sovereign_composer.py`

Already partially fixed, but needs strengthening:

```python
def _is_euler_constant(self, token: str, context: str) -> bool:
    """
    Determine if 'e' is Euler's constant or just a letter.

    Euler's constant appears in:
    - Standalone: just 'e' as a symbol
    - In exponent: e^x, e^{...}
    - After number: 2.718... (already a number)

    NOT Euler's constant:
    - In English words: "she", "the", "here"
    - Variable names: "element", "edge"
    """
    if token != "e":
        return False

    # Check if preceded/followed by letters (part of word)
    # Check if in mathematical context (\exp, ^, LaTeX)
    # ...
```

---

## Architectural Evolution: Model-Driven Parsing

### Current: Hardcoded Strategy

```
[Problem Text] → [Python regex] → [Early-return rules] → [RPN]
```

### Target: Learned Strategy

```
[Problem Text]
    ↓
[Tokenizer (Grammar Galaxy)] → tokens with symbol IDs
    ↓
[Rule Matcher (Grammar Rules)] → candidate rule applications
    ↓
[MODEL (Learned Weights)] → selects rules, orders them, chains results
    ↓
[RPN Composer] → final RPN program
    ↓
[PTX Engine] → numeric result
```

### Implementation Path

**Phase 1: Fix immediate bugs (this prompt)**
- Remove answer extraction
- Fix rule chaining in word solver
- Fix composer 'e' handling

**Phase 2: Add rule selection scoring**
- Each rule match gets a confidence score
- Rules are sorted by score
- Top-N rules are applied in order

**Phase 3: Train model weights for rule selection**
- Input: problem text + candidate rules
- Output: ordered rule applications + chaining instructions
- Training data: GSM8K/MATH with step-by-step solutions

**Phase 4: Full compositional reasoning**
- Model learns to decompose problems
- Model learns intermediate variable binding
- Model learns to compose multi-step RPN

---

## Task List for Codex

### Task 1: Remove Answer Extraction (Critical)

**File:** `scripts/run_sovereign_math_benchmarks.py`

1. Remove GSM8K hash extraction (lines 133-146)
2. Remove fallback extraction (lines 176-181)
3. GSM8K should go through composer/word_solver like other datasets

**Expected result:** GSM8K accuracy will DROP (probably to ~20-30%), but it will be REAL.

### Task 2: Fix Word Solver Rule Chaining

**File:** `knowledge3d/training/math_benchmarks/word_problem_solver.py`

1. Replace `_generate_rpn` with compositional logic
2. Handle multi-rule patterns:
   - "half" + "altogether" → `X DUP 2 / +`
   - "twice" + "altogether" → `X DUP 2 * +`
   - "percentage" + "remaining" → `X X P * 100 / -`

3. Add DUP opcode support in ModularRPNEngine if not present

### Task 3: Fix Composer 'e' Handling

**File:** `knowledge3d/training/math_benchmarks/sovereign_composer.py`

1. Add context-aware 'e' detection
2. Only treat 'e' as Euler when:
   - Standalone symbol
   - In exponent context (e^x)
   - Preceded by number (2.718e3)
3. NOT when part of English word

### Task 4: Add Rule Confidence Scoring (Enhancement)

**New concept:** Each rule match gets a score based on:
- Pattern specificity (longer patterns = higher score)
- Number of captured groups
- Domain relevance

```python
def score_rule_match(rule, match, problem_text) -> float:
    specificity = len(rule.pattern)
    captures = len(match.groups())
    domain_boost = 1.0  # Adjust based on problem source
    return specificity * 0.1 + captures * 0.5 + domain_boost
```

### Task 5: Codex Original Ideas

**Codex:** Based on this analysis, propose your own enhancements:
- How would you structure multi-step reasoning?
- What additional grammar rules would help?
- How should the model learn rule selection?
- What's missing from the current architecture?

---

## Verification After Implementation

```bash
# Test word solver chaining
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.word_problem_solver import WordProblemSolver
solver = WordProblemSolver()
result = solver.solve('Natalia sold clips to 48 friends in April, sold half as many in May. How many altogether?')
print(f'RPN: {result[\"rpn_program\"]}')
# Expected: '48 DUP 2 / +' or '48 48 2 / +' (result = 72)
"

# Test composer doesn't produce Euler spam
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.sovereign_composer import SovereignComposer
c = SovereignComposer()
print(c.compose('She sold 48 items'))
# Expected: '48' or '48.0', NOT '48.0 2.71828...'
"

# Run benchmark (expect lower GSM8K, but REAL)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/run_sovereign_math_benchmarks.py --limit 100
```

---

## Success Criteria

1. **GSM8K drops to real accuracy** (~20-40%) - no more answer extraction
2. **Word solver chains rules** - "half + altogether" → correct RPN
3. **Composer no 'e' spam** - English text doesn't produce Euler constants
4. **Overall accuracy is REAL** - model is actually solving, not extracting

---

## Key Principle

**The model should SOLVE using Galaxy symbols + Grammar rules + RPN execution.**

- Symbols = vocabulary (what operations exist)
- Grammar = patterns (how to recognize structures)
- Model weights = intelligence (which rules, what order, how to chain)
- RPN engine = computation (execute the program)

**Humans don't copy-paste answers. Neither should our model.**

---

## Codex: Your Turn

Implement Tasks 1-4, then add your original ideas in Task 5.

Consider:
- What multi-step patterns are common in GSM8K?
- How should the model learn to bind intermediate results?
- What's the minimal architecture change to enable compositional reasoning?
- Are there grammar rules missing that would unlock more problems?

The goal is TRUE sovereign solving - not extraction, not hardcoded heuristics, but learned compositional reasoning over Galaxy symbols.
