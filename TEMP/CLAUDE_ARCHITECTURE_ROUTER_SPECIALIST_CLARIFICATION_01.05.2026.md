# Architecture Clarification: Router Specialist for Theorem Routing

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead)
**Date**: January 5, 2026
**Context**: Clarifying CODEX_DIRECTIVE_INTELLIGENT_THEOREM_ROUTING_01.05.2026.md

---

## Critical Architectural Correction

**User's feedback**: "router as specialist - remember the architecture, the router is supposed to be a base model+routing attachment (specialist in the MoE)"

**What this means**: The "intelligent routing" in the directive is **NOT hardcoded if/else logic** — it's a **ROUTER SPECIALIST** (LoRA adapter) that learns mappings.

---

## Router Specialist Architecture (Existing K3D System)

From [knowledge3d/cranium/router_specialist.py](../knowledge3d/cranium/router_specialist.py):

### Key Insights

1. **Router IS a specialist** (LoRA-style adapter, NOT external infrastructure)
2. **Router learns from routing decisions** (Task features → Specialist weights)
3. **Bootstrap workflow**:
   - Phase 1: Heuristic routing (keyword matching) — collect decisions
   - Phase 2: Train router specialist on successful patterns
   - Phase 3: Switch to learned routing (router specialist)
   - Phase 4: Router self-updates from new routing data (continual learning)

### Router Specialist Components

```python
class RouterBootstrap:
    """Bootstrap router specialist using heuristic routing."""
    def collect_routing_data(tasks, outcome_fn) -> List[RoutingDecision]
    def filter_successful_decisions(min_performance=0.5) -> List[RoutingDecision]

class RouterSpecialistTrainer:
    """Train router specialist from routing decisions."""
    def register_router_specialist(num_specialists, dims=256, rank=16)
    def train_from_history(routing_history, epochs=5)
    def update_from_new_decisions(new_decisions, epochs=1) -> bool  # Continual learning

@dataclass
class RoutingDecision:
    input_data: np.ndarray              # Task input
    task_description: Optional[str]     # Text description
    specialist_weights: Dict[str, float]  # Predicted weights
    outcome_performance: float          # How well did it work? [0-1]
    timestamp: str
```

---

## Theorem Pattern Routing = Router Specialist Task

**Original directive said**: "TRM acts intelligently to map patterns → grammar rules"

**Architectural truth**: The **ROUTER SPECIALIST** learns this mapping (not hardcoded logic)

### How it Works

**Theorem patterns → Grammar rules** is a **routing decision**:
- **Input**: Theorem pattern semantic tags (e.g., `["derivative", "product_rule"]`)
- **Output**: Grammar rule weights (e.g., `{"apply_product_rule": 0.9, "apply_power_rule": 0.1}`)
- **Router specialist** learns: "When I see semantic tags X, route to grammar rule Y"

### Training Data Format

```python
RoutingDecision(
    input_data=np.array([...]),  # Embedded semantic tags from theorem pattern
    task_description="product_rule pattern matched",
    specialist_weights={
        "apply_product_rule": 1.0,  # Target grammar rule
        "apply_quotient_rule": 0.0,
        "apply_power_rule": 0.0,
        # ... other grammar rules
    },
    outcome_performance=0.9,  # If execution succeeded
    timestamp="2026-01-05T..."
)
```

---

## Revised Implementation Strategy

### What `grammar_rule` Field ACTUALLY Is

**NOT**: Hardcoded routing table
**IS**: Training label for router specialist

```python
_THEOREM_PATTERN_DEFS = [
    {
        "pattern_id": "product_rule",
        "semantic_tags": ["derivative", "product_rule"],
        "grammar_rule": "apply_product_rule",  # ← Training label for router
        # ...
    },
]
```

### What TRM Does (Corrected Understanding)

```
Problem: "Find derivative of (x² + 1)(x³ - 2)"
  ↓
TRM semantic search → Matches theorem pattern "product_rule"
  ↓
TRM queries ROUTER SPECIALIST:
  Input: semantic_tags ["derivative", "product_rule"]
  Output: grammar_weights {"apply_product_rule": 0.92, ...}
  ↓
Execute highest-weighted grammar rule (apply_product_rule)
  ↓
Record outcome: RoutingDecision(outcome_performance=0.9)
  ↓
Router specialist updates from success (continual learning)
```

### Key Differences from Original Directive

| Original Directive | Architectural Truth |
|-------------------|-------------------|
| Hardcoded if/else mapping | Router specialist learns mapping |
| `_execute_grammar_rule(rule_name)` | Router specialist outputs weights, executor uses highest |
| One-time mapping | Continual learning from outcomes |
| Static routing | Dynamic routing (improves over time) |

---

## Implementation Tasks (Revised)

### Task 1: Add `grammar_rule` Training Labels

**File**: `knowledge3d/cranium/math_galaxy_population.py`

```python
_THEOREM_PATTERN_DEFS = [
    {
        "pattern_id": "power_rule_polynomial",
        "semantic_tags": ["derivative", "polynomial", "power_rule"],
        "grammar_rule": "apply_power_rule",  # Training label
        # ...
    },
    # ... add to all 9 patterns
]
```

**Purpose**: These labels define TARGET routing decisions for router specialist training.

---

### Task 2: Research Existing Grammar Rules

**Question**: Do these grammar rules exist in Grammar Galaxy?
- `apply_power_rule`
- `apply_product_rule`
- `apply_quotient_rule`
- `apply_chain_rule`
- `apply_sum_rule`
- `apply_constant_multiple_rule`
- `apply_integration_by_parts`
- `apply_fundamental_theorem_calculus`
- `apply_pythagorean_identity`

**Research findings** (from architecture investigation):

**✅ Existing**:
- GrammarRule system in `knowledge3d/training/arc_agi/grammar_galaxy.py`
- GSM8K arithmetic templates in `knowledge3d/training/math_benchmarks/math_templates.py`
- Test case for "en_derivative" in `scripts/test_grammar_galaxy.py`

**❌ Missing**:
- No existing calculus grammar rules (product_rule, quotient_rule, etc.)
- Need to create these as GrammarRule objects with RPN programs

---

### Task 3: Create Calculus Grammar Rules (NEW TASK)

**File**: Create `knowledge3d/training/math_benchmarks/calculus_grammar_rules.py`

**Goal**: Define GrammarRule objects for calculus operations (executable numeric RPN)

**Template**:

```python
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

CALCULUS_RULES = [
    GrammarRule(
        rule_id="apply_power_rule",
        language="math",
        pattern=r"d/dx\s*\(?\s*x\s*\^\s*(\d+\.?\d*)\s*\)?",  # d/dx(x^n)
        rpn_program="{0} PUSH_X {0} 1 - POW *",  # n * x^(n-1)
        domain="calculus",
    ),
    GrammarRule(
        rule_id="apply_product_rule",
        language="math",
        pattern=r"d/dx\s*\(?\s*\(([^)]+)\)\s*\*\s*\(([^)]+)\)\s*\)?",  # d/dx(f*g)
        rpn_program="...",  # f'*g + f*g' (need to compose)
        domain="calculus",
    ),
    # ... 7 more rules
]
```

**Note**: These rules use **numeric RPN** (Tier 1-3 ops: PUSH_X, POW, MULT, ADD), NOT symbolic (no PUSH_F, DERIVATIVE).

---

### Task 4: Bootstrap Router Specialist for Theorem Routing

**File**: Create `scripts/train_theorem_router_specialist.py`

**Goal**: Train router specialist to map theorem patterns → grammar rules

**Workflow**:

```python
from knowledge3d.cranium.router_specialist import (
    RouterBootstrap, RouterSpecialistTrainer, RoutingDecision
)
from knowledge3d.cranium.math_galaxy_population import THEOREM_PATTERNS

# Phase 1: Bootstrap with heuristic routing (pattern_id → grammar_rule exact match)
bootstrap = RouterBootstrap(swarm)
routing_history = []

for pattern in THEOREM_PATTERNS:
    pattern_id = pattern["pattern_id"]
    grammar_rule = pattern.get("grammar_rule")

    if not grammar_rule:
        continue

    # Create training samples: semantic_tags → grammar_rule
    semantic_embedding = embed_semantic_tags(pattern["semantic_tags"])

    decision = RoutingDecision(
        input_data=semantic_embedding,
        task_description=f"{pattern_id} pattern",
        specialist_weights={grammar_rule: 1.0},  # Target rule
        outcome_performance=1.0,  # Assume heuristic works
        timestamp=datetime.now().isoformat()
    )

    routing_history.append(decision)

# Phase 2: Train router specialist
trainer = RouterSpecialistTrainer(swarm)
trainer.register_router_specialist(num_specialists=len(CALCULUS_RULES))
trainer.train_from_history(routing_history, epochs=10)

# Phase 3: Save router weights
swarm.save_specialists("trained_theorem_router.npz")
```

---

### Task 5: Integrate Router Specialist into TRM Navigation

**File**: `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

**Changes**:

```python
class TRMGalaxyReader:
    def __init__(self, ..., router_specialist=None):
        self.router = router_specialist  # Router specialist instance

    def solve(self, problem_text, rpn_engine, max_attempts=3):
        # Match theorem patterns (existing code)
        matched_patterns = self._match_theorem_patterns(problem_text)

        if not matched_patterns:
            return None, {}

        # Router specialist: semantic_tags → grammar_rule weights
        for pattern in matched_patterns:
            semantic_embedding = self._embed_semantic_tags(pattern["semantic_tags"])

            # Query router specialist (NOT hardcoded if/else!)
            grammar_weights = self.router.route_blend(input_data=semantic_embedding)

            # Execute highest-weighted grammar rule
            best_rule = max(grammar_weights.items(), key=lambda x: x[1])[0]
            result = self._execute_grammar_rule(best_rule, problem_text, rpn_engine)

            if result is not None:
                # Record successful routing for continual learning
                outcome = 1.0 if self._validate_result(result) else 0.0
                self._record_routing_decision(
                    pattern_id=pattern["pattern_id"],
                    semantic_embedding=semantic_embedding,
                    grammar_rule=best_rule,
                    outcome_performance=outcome
                )
                return result, {"rule_used": best_rule, ...}

        return None, {}
```

---

## Architecture Principles (Corrected)

### Theorem Patterns = Semantic Knowledge (Training Labels)

- **Purpose**: Teaching material for router specialist
- **NOT**: Hardcoded routing table
- **IS**: Training labels defining target routing decisions

### Grammar Galaxy = Execution Rules

- **Purpose**: Numeric RPN programs (Tier 1-3 ops)
- **Sovereign**: PTX-backed execution
- **Content**: Executable transformation rules

### Router Specialist = Learned Navigation

- **Architecture**: LoRA adapter (rank 16, dims 256)
- **Learns**: semantic_tags → grammar_rule weights
- **Training**: Bootstrap from heuristic, then continual learning
- **Updates**: Self-improves from successful routing outcomes

### Shadow Copy Learning (Existing System)

- **What it is**: Continual learning mechanism for ALL specialists
- **Router shadow**: Accumulates routing decisions, updates weights
- **No weight updates**: Router learns NAVIGATION, not knowledge

---

## Success Criteria (Revised)

### Implementation Quality
- [ ] Calculus grammar rules created (9 rules as GrammarRule objects)
- [ ] Router specialist registered in swarm
- [ ] Router specialist trained on theorem pattern → grammar rule mappings
- [ ] TRM navigation queries router specialist (not hardcoded if/else)
- [ ] Routing decisions recorded for continual learning

### MATH Benchmark Results
- [ ] Accuracy ≥ 2% (improvement from 1% baseline)
- [ ] Logs show router specialist weights for each pattern
- [ ] Grammar rules execute successfully (numeric RPN)
- [ ] Routing decisions recorded with outcome performance

### Router Specialist Metrics
- [ ] Router specialist accuracy ≥ 80% on theorem pattern routing
- [ ] Continual learning improves routing over benchmark run
- [ ] Router weights shift toward successful grammar rules

---

## Key Architectural Insight

**Original directive said**: "TRM acts intelligently"
**Architectural truth**: "Router SPECIALIST acts intelligently (learned adapter, not hardcoded logic)"

The `grammar_rule` field is a **training label** for the router specialist, not a hardcoded routing table. The router specialist LEARNS the mapping: semantic patterns → grammar rules.

This is the K3D way: **Specialists learn navigation, knowledge lives in Galaxy**.

---

## Questions for Codex Before Starting?

1. Do you understand router specialist is a LoRA adapter (NOT hardcoded if/else)?
2. Do you understand `grammar_rule` field is a training label (NOT a lookup key)?
3. Can you create calculus grammar rules as GrammarRule objects (numeric RPN)?
4. Do you understand router specialist needs bootstrap training before use?

**If clear, proceed with**:
1. Task 3 (create calculus grammar rules)
2. Task 4 (bootstrap router specialist training)
3. Task 5 (integrate router into TRM navigation)

---

## Additional Resources

- [router_specialist.py](../knowledge3d/cranium/router_specialist.py) — Full router specialist implementation
- [math_templates.py](../knowledge3d/training/math_benchmarks/math_templates.py) — Example GrammarRule templates
- [grammar_galaxy.py](../knowledge3d/training/arc_agi/grammar_galaxy.py) — GrammarRule class definition

---

**This clarifies the architecture. Router specialist = learned navigation, not hardcoded logic.** 🧠
