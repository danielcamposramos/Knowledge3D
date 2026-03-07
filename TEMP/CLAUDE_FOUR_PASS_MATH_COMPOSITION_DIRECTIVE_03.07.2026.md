# Codex Directive: Four-Pass Word Problem Composition

**Date:** March 7, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Supersedes:** All previous math composition approaches in `_build_word_problem_rpn`

---

## The Problem

The current `_build_word_problem_rpn` in `math_specialist.py` (lines 1281-1325) extracts ALL numbers from the problem into a flat list, then guesses which numbers and operations to use based on a problem-type label. This loses the semantic relationship between numbers and their context.

Proof: GSM8K problem 3 -- "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint."
- Numbers extracted: [3, 3, 60] (flat, no context)
- `word_problem_rate` path takes numbers[0] * numbers[-1] = 3 * 60 = 180
- Correct: 3 sprints * 3 times/week * 60 meters/sprint = 540

The flat list destroys the semantic binding between "3" and "sprints", between "3" and "times a week", between "60" and "meters each sprint". Without that binding, composition is impossible.

---

## Daniel's Architecture: Four Passes

This is how humans translate word problems into math. Not one pass. Four passes, each building on the previous.

### Pass 1: Forward Reading (already exists)

Parse left-to-right. Each clause becomes a structured entity.

```
Input:  "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?"

Forward parse:
  clause[0]: "James decides to run 3 sprints 3 times a week"
    -> entities: [{value: 3, role: "count", unit: "sprints"}, {value: 3, role: "frequency", unit: "times a week"}]
    -> operation_hint: MULTIPLY (implied by "X things Y times")

  clause[1]: "He runs 60 meters each sprint"
    -> entities: [{value: 60, role: "rate", unit: "meters per sprint"}]
    -> operation_hint: MULTIPLY (implied by "each")

  clause[2]: "How many total meters does he run a week"
    -> goal: {target: "meters", scope: "a week", aggregation: "total"}
```

### Pass 2: Backward Reading (already exists)

Parse right-to-left. Start from the GOAL and trace what it needs.

```
Backward parse:
  goal: "How many total meters does he run a week"
    -> needs: total_meters_per_week

  clause[1]: "He runs 60 meters each sprint"
    -> provides: meters_per_sprint = 60
    -> needs: number_of_sprints_per_week

  clause[0]: "James decides to run 3 sprints 3 times a week"
    -> provides: sprints_per_session = 3, sessions_per_week = 3
    -> sprints_per_week = sprints_per_session * sessions_per_week
```

### Pass 3: Fusion (Deduplicate + Merge)

Merge forward and backward into a single semantic entity graph. Deduplicate entries that both passes found. This is the third pass Daniel described.

```
Fused entity graph:
  E1: {value: 3, role: "count", semantic: "sprints_per_session", source: [forward_clause_0, backward_clause_0]}
  E2: {value: 3, role: "frequency", semantic: "sessions_per_week", source: [forward_clause_0, backward_clause_0]}
  E3: {value: 60, role: "rate", semantic: "meters_per_sprint", source: [forward_clause_1, backward_clause_1]}
  GOAL: {target: "total_meters_per_week", source: [forward_clause_2, backward_goal]}

  Deduplication: E1 appeared in both passes -> confirmed entity (higher confidence)
  Deduplication: E2 appeared in both passes -> confirmed entity (higher confidence)
  Deduplication: E3 appeared in both passes -> confirmed entity (higher confidence)
```

### Pass 4: Semantic Verification + Formula Construction

This is the critical pass. Attach operations to relationships between entities. Build the formula. Verify dimensional consistency. This is the "words to math" translation.

```
Semantic verification:
  GOAL needs: total_meters_per_week
    = meters_per_sprint * sprints_per_week          [dimensional: meters/sprint * sprints/week = meters/week]
    = meters_per_sprint * (sprints_per_session * sessions_per_week)   [expand]
    = E3 * (E1 * E2)                                [bind entities]
    = 60 * (3 * 3)                                  [substitute values]

  Dimensional check:
    meters/sprint * sprints/session * sessions/week = meters/week  [CONSISTENT]

  RPN construction:
    3 3 MUL 60 MUL    (or equivalently: 60 3 MUL 3 MUL)

  Result: 540
```

---

## How This Maps to Existing Code

### What Exists (Use, Don't Rewrite)

1. **`NavigatorSpecialist._forward_reading_path()`** (line 349): Already splits into clauses and extracts variables. Returns `forward_parse: {context: [...], goal: {...}}`.

2. **`NavigatorSpecialist._backward_reading_path()`** (line 365): Already reverses clauses and extracts dependencies. Returns `backward_parse: {goal: {...}, dependencies: [...]}`.

3. **`NavigatorSpecialist._fusion_reading_path()`** (line 382): Already merges and deduplicates variable assignments. Returns `fusion_parse: {merged_variables: {...}, unified_goal: {...}}`.

4. **`MathSpecialist._extract_parse_bundle()`** (line 832): Already receives all three parse results.

5. **`MathSpecialist._extract_word_problem_entities()`** (line 1235): Already reads from `parse_bundle` clauses. But then FLATTENS everything into `numbers: [all numbers]`. This is where the problem is.

### What Must Change

**Replace `_build_word_problem_rpn` (lines 1281-1325)** -- the flat if/else tree -- with the four-pass composition. The first three passes already exist in the navigator. Pass 4 is new.

**Replace `_extract_word_problem_entities` (lines 1235-1279)** -- stop extracting numbers as a flat list. Instead, extract SEMANTIC ENTITIES: each number bound to its clause, its role (count/rate/frequency/quantity), and its unit.

**Add Pass 4: `_verify_and_construct_formula`** -- this is the new method that:
1. Takes the fused entity graph
2. Identifies what the GOAL needs (dimensional analysis)
3. Chains operations between entities to produce the goal
4. Verifies dimensional consistency
5. Outputs RPN

---

## Detailed Design: Semantic Entity Extraction

### Current (WRONG -- flat list)

```python
def _extract_word_problem_entities(self, question, *, parse_bundle):
    return {
        "numbers": self._extract_numeric_literals(question),  # FLAT LIST -- context lost
        "clauses": clauses,
        "has_rate": any(token in lowered for token in (" each ", " per "...)),
        ...
    }
```

### Required (Semantic entities per clause)

Each clause produces zero or more SEMANTIC ENTITIES. An entity is a number bound to:
- **value**: the numeric value
- **role**: what kind of quantity (count, rate, frequency, amount, total, difference)
- **unit**: what it measures (sprints, meters, dollars, eggs, times)
- **scope**: temporal/spatial qualifier (per day, per week, each, every)
- **clause_index**: which clause it came from
- **operation_context**: what operation the surrounding words imply

Role detection from surrounding words:
- "X things" / "X items" / noun after number -> role=count, unit=noun
- "X per Y" / "X each Y" / "X every Y" -> role=rate, unit=X, scope=per_Y
- "X times" / "X times a Y" -> role=frequency, scope=per_Y
- "$X" / "X dollars" / "X cents" -> role=price, unit=currency
- "remaining" / "left" / "after" -> role=result (from subtraction)
- "total" / "altogether" / "in all" -> role=goal_aggregate

The entities are NOT a flat list. They are a structured graph with relationships.

### Entity Relationship Detection

Once entities are extracted per-clause, detect relationships:

```
"3 sprints 3 times a week" -> E1(3, count, sprints) * E2(3, frequency, per_week)
"60 meters each sprint"    -> E3(60, rate, meters/sprint)
```

Relationship keywords:
- Same clause, two entities, "times/per/each/every" between them -> MULTIPLY
- "left/remaining/after" preceding entity -> SUBTRACT from previous
- "more than" / "added" -> ADD
- "split among" / "divided by" / "shared" -> DIVIDE
- Sequential clauses with dependent units -> CHAIN (output of previous is input to next)

### Unit Chaining (Dimensional Analysis)

This is the key insight that makes composition work for ANY multi-step problem, not just known templates:

```
GOAL: meters/week
  = meters/sprint * sprints/week             [units cancel: sprint/sprint]
  = meters/sprint * sprints/session * sessions/week   [expand]
  = 60 * 3 * 3 = 540
```

If the units don't cancel to produce the goal's unit, the composition is WRONG and should be rejected (contrastive: -1 this path, try another).

---

## Worked Examples

### GSM8K 0 (Currently Correct: Janet's ducks)

```
Pass 1 (Forward):
  C0: "Janet's ducks lay 16 eggs per day"
    -> E1{value:16, role:rate, unit:eggs, scope:per_day}
  C1: "She eats three for breakfast every morning"
    -> E2{value:3, role:consume, unit:eggs, scope:per_day}
  C2: "bakes muffins for her friends every day with four"
    -> E3{value:4, role:consume, unit:eggs, scope:per_day}
  C3: "She sells the remainder at $2 per fresh egg"
    -> E4{value:2, role:price, unit:dollars/egg}
  C4: "How much in dollars does she make every day"
    -> GOAL{target:dollars, scope:per_day}

Pass 2 (Backward):
  GOAL: dollars/day
    needs: remaining_eggs * price_per_egg
  C3: provides price_per_egg = 2
    needs: remaining_eggs
  C2+C1: consumed = 3 + 4 = 7
    needs: total_eggs
  C0: provides total_eggs = 16/day

Pass 3 (Fusion):
  E1{16, rate, eggs/day} -- confirmed both passes
  E2{3, consume, eggs/day} -- confirmed both passes
  E3{4, consume, eggs/day} -- confirmed both passes
  E4{2, price, dollars/egg} -- confirmed both passes

Pass 4 (Verify + Construct):
  dollars/day = (eggs/day - eggs_consumed/day) * dollars/egg
  = (16 - 3 - 4) * 2
  RPN: 16 3 SUB 4 SUB 2 MUL
  Dimensional: eggs/day - eggs/day - eggs/day = eggs/day; eggs/day * dollars/egg = dollars/day [CONSISTENT]
  Result: 18
```

### GSM8K 1 (Currently FAILING: Robe bolts)

"A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"

```
Pass 1 (Forward):
  C0: "A robe takes 2 bolts of blue fiber"
    -> E1{value:2, role:quantity, unit:bolts, qualifier:blue}
  C1: "half that much white fiber"
    -> E2{value:REFERENCE("half", E1), role:quantity, unit:bolts, qualifier:white}
       NOTE: "half that much" is a REFERENCE to E1, not a standalone number
  C2: "How many bolts in total does it take"
    -> GOAL{target:bolts, aggregation:total}

Pass 2 (Backward):
  GOAL: total bolts
    needs: blue_bolts + white_bolts
  C1: white_bolts = half of blue_bolts
    needs: blue_bolts
  C0: blue_bolts = 2

Pass 3 (Fusion):
  E1{2, quantity, bolts, blue} -- confirmed
  E2{HALF_OF(E1), quantity, bolts, white} -- reference resolved: 2/2 = 1

Pass 4 (Verify + Construct):
  total_bolts = blue_bolts + white_bolts
  = E1 + (E1 / 2)
  = 2 + 1
  RPN: 2 2 2 DIV ADD
  (or: 2 DUP 2 DIV ADD)
  Result: 3
```

**Critical:** "half that much" is NOT a number in the text. It's a REFERENCE EXPRESSION. The entity extractor must recognize:
- "half" / "twice" / "triple" / "double" -> multiplier reference to nearest previous quantity
- "that much" / "the same" / "as many" -> explicit back-reference

### GSM8K 3 (Currently FAILING: James sprints)

"James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?"

```
Pass 1 (Forward):
  C0: "James decides to run 3 sprints 3 times a week"
    -> E1{value:3, role:count, unit:sprints, scope:per_session}
    -> E2{value:3, role:frequency, unit:sessions, scope:per_week}
    -> intra_clause_op: E1 * E2 (because "X things Y times" implies multiplication)

  C1: "He runs 60 meters each sprint"
    -> E3{value:60, role:rate, unit:meters, scope:per_sprint}

  C2: "How many total meters does he run a week"
    -> GOAL{target:meters, scope:per_week, aggregation:total}

Pass 2 (Backward):
  GOAL: meters/week
    needs: meters/sprint * sprints/week
  C1: meters/sprint = 60
    needs: sprints/week
  C0: sprints/week = 3 sprints/session * 3 sessions/week = 9

Pass 3 (Fusion):
  E1{3, count, sprints/session} -- confirmed
  E2{3, frequency, sessions/week} -- confirmed
  E3{60, rate, meters/sprint} -- confirmed
  Derived: sprints/week = E1 * E2 = 9

Pass 4 (Verify + Construct):
  meters/week = meters/sprint * sprints/week
  = E3 * (E1 * E2)
  = 60 * (3 * 3)
  RPN: 3 3 MUL 60 MUL
  Dimensional: meters/sprint * sprints/session * sessions/week = meters/week [CONSISTENT]
  Result: 540
```

---

## Implementation Spec

### New Data Structures

```python
@dataclass
class SemanticEntity:
    value: float | None          # numeric value (None if reference expression)
    role: str                    # count, rate, frequency, price, consume, quantity
    unit: str                    # what it measures: eggs, meters, dollars, sprints
    scope: str                   # temporal: per_day, per_week, each, every (or "")
    clause_index: int            # which clause this came from
    raw_text: str                # the original text fragment
    reference: str | None        # "half_of:E1", "twice:E2", etc. (None if standalone)
    confidence: float            # 0-1, higher if confirmed by both passes

@dataclass
class EntityRelation:
    left: str                    # entity id
    right: str                   # entity id
    operation: str               # MUL, ADD, SUB, DIV
    source: str                  # "intra_clause", "inter_clause", "goal_chain"
    clause_index: int

@dataclass
class CompositionPlan:
    entities: list[SemanticEntity]
    relations: list[EntityRelation]
    goal_unit: str               # what the answer should be in
    goal_scope: str              # temporal scope of the answer
    rpn_program: str             # the final composed RPN
    dimensional_check: bool      # True if units cancel correctly
```

### New Methods in MathSpecialist

```
_extract_semantic_entities(clause: str, clause_index: int, parse_context: dict) -> list[SemanticEntity]
    Extract entities with role/unit/scope from a single clause.
    Detect reference expressions ("half that much", "twice as many").
    Use Number Galaxy symlinks for word-form numbers ("three" -> 3).

_detect_entity_relations(entities: list[SemanticEntity], clauses: list[str]) -> list[EntityRelation]
    Detect operations between entities from keyword context.
    Intra-clause: "X things Y times" -> MUL
    Inter-clause: sequential clauses with consume keywords -> SUB
    Goal-chain: trace what the goal needs through entity dependencies.

_fuse_entity_graphs(forward_entities, backward_entities) -> list[SemanticEntity]
    Deduplicate entities found by both passes.
    Boost confidence for entities confirmed by both directions.
    Resolve reference expressions using the fused graph.

_verify_and_construct_formula(entities, relations, goal) -> CompositionPlan
    Build the computation chain from goal backward through relations.
    Check dimensional consistency (units must cancel to goal's unit).
    Output RPN program.
    If dimensions don't check: return None (contrastive -1 this path).

_compose_four_pass(question, parse_bundle) -> CompositionPlan | None
    Orchestrate the four passes:
    1. Forward entity extraction (from parse_bundle forward_parse clauses)
    2. Backward entity extraction (from parse_bundle backward_parse dependencies)
    3. Fusion (deduplicate, resolve references, boost confirmed entities)
    4. Verify + construct (dimensional analysis, RPN output)
```

### What Gets Removed

- `_build_word_problem_rpn` (lines 1281-1325) -- the flat if/else tree
- The flat `numbers` list in `_extract_word_problem_entities`
- The `{rpn_chain}` placeholder templates (`math_template_word_problem_*_v1`)
- All pattern_type-specific branching in word problem composition

### What Gets Kept

- `_infer_word_problem_type` -- still useful as a first signal, but no longer determines the composition strategy
- Grammar Galaxy composition rules (`gsm_consume_from_total`, etc.) -- these become the RELATION types, not the composition driver
- Number Galaxy symlinks -- used by entity extraction for word-form numbers
- Forward/backward/fusion parsing in NavigatorSpecialist -- the first three passes

---

## Role Keywords for Entity Extraction

These are Grammar Galaxy knowledge -- compositional, not problem-specific:

```
COUNT:     "X [noun]", number followed by a noun
RATE:      "X per Y", "X each Y", "X every Y", "$X per Y"
FREQUENCY: "X times", "X times a Y", "Y X times"
PRICE:     "$X", "X dollars", "X cents", "costs X"
CONSUME:   "eats/uses/spends/loses/gives/bakes X"
PRODUCE:   "lays/makes/earns/receives/grows X"
REFERENCE: "half/twice/triple/double/three times" + "that/the/as much/as many"
REMAINDER: "remaining/left/leftover/rest"
TOTAL:     "total/altogether/combined/in all/sum"
COMPARE:   "more than/less than/fewer than/difference"
```

These keywords map numbers to ROLES, not to operations directly. The operations come from the RELATIONSHIPS between roles:
- count * frequency = total_count
- total - consumed = remainder
- remainder * price = revenue
- rate * count = total_amount

---

## Success Criteria

1. GSM8K 0 (Janet's ducks): predicted=18, correct (already works, must still work)
2. GSM8K 1 (Robe bolts): predicted=3 (requires reference resolution: "half that much")
3. GSM8K 3 (James sprints): predicted=540 (requires intra-clause multiplication: "3 sprints 3 times")
4. No flat number list anywhere in the word problem path
5. Every entity carries its role, unit, scope, and clause origin
6. Dimensional check passes for correct compositions and fails for wrong ones
7. Reference expressions ("half", "twice", "double") correctly resolve
8. Forward and backward passes produce independent entity lists that fusion deduplicates
9. The same composition logic handles ANY multi-step word problem, not just known types
10. All new code is sovereign (no numpy/cupy/scipy)

---

## Daniel's Principle (Verbatim)

> "The forward backward passes must be composed into a third one that deduplicates the entries -- then a fourth pass is the one that re-verifies it with semantic attached, and this must consider the operations as well. This is how humans put the problem from words to actual math formulae and then calculate it."

Four passes. Not a flat list. Not templates. Not problem-by-problem. The grammar rules describe RELATIONSHIPS between quantities. The TRM composes them. The PTX executes the resulting RPN.
