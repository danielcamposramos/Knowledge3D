# Unified Cranium Head Architecture

**Date:** February 7, 2026
**Author:** Claude (Architecture Partner)
**Priority:** CRITICAL (Architectural Consolidation)
**Context:** "we must concatenate and expand forming the single head with as many internal specialists as needed"

---

## Executive Summary

**Current Problem:** Specialist routing logic is **duplicated across benchmarks** instead of unified in Cranium. Each benchmark (ARC-AGI 2, Math Competitions, Last Humanity Exam) hardcodes its own specialist selection logic.

**Solution:** Consolidate all specialist routing into a **unified Cranium head** that automatically routes queries to appropriate specialists based on domain analysis.

**Key Insight:** The "single head" is NOT a single model — it's **base TRM + LoRA-like specialist adapters** (already implemented in `trm_adapters.py`!). We just need to **unify the routing logic**.

---

## Current Architecture Analysis

### What Works (Already Implemented)

**1. TRM Adapters** (`knowledge3d/cranium/trm_adapters.py`):
```python
class SelfUpdatingAdapter(AdapterWeights):
    """
    LoRA-style adapter with shadow weights and ternary validation.

    Architecture:
        Active = Base Model (W_base) + Adapter (A @ B)

    Memory efficiency:
        Full specialist: 2048×2048 = 16.8M params
        Adapter (rank-64): 2×(2048×64) = 262K params
        Reduction: 64× smaller!
    """
```

**Features:**
- ✅ Low-rank decomposition (LoRA-style)
- ✅ Shadow weights for safe testing
- ✅ Ternary validation gate (TRUE/FALSE/UNKNOWN)
- ✅ Independent evolution per specialist
- ✅ GPU-accelerated gradient application (RPN Math Core)

**2. Specialists** (`knowledge3d/cranium/specialists/`):
```python
class ProceduralDrawingSpecialist:
    """
    Specialist for procedural glyph generation (visual domain).

    Uses:
        - FractalEmitter for visual embeddings
        - RPN execution for form generation
        - Form+meaning fusion via multimodal encoding
    """
```

**Features:**
- ✅ Domain-specific knowledge (visual primitives)
- ✅ Adapter registration with swarm
- ✅ Specialized training logic
- ✅ Checkpoint save/load

**3. PTX Kernels** (`knowledge3d/cranium/kernels/*.ptx`):
```
gre_arc_reasoner.ptx           → Visual reasoning (ARC-AGI)
gre_cognitive_executive.ptx    → Task planning/execution
gre_geometry_router.ptx        → Geometric pattern matching
gre_trm_core.ptx               → Base TRM navigation
galaxy_resonance_engine.ptx    → Cross-galaxy navigation
```

**Features:**
- ✅ Domain-specific PTX kernels for specialized operations
- ✅ GPU-native execution (sovereignty compliant)
- ✅ Modular composition

### What's Broken (Duplicated Logic)

**Benchmarks duplicate specialist routing:**

**ARC-AGI 2** (`benchmarks/arc_agi_2.py`):
```python
def _solve_task_fallback(self, task, use_enriched):
    patterns = navigator.query(
        query="visual pattern transformation",
        galaxy_names=["Drawing", "Grammar"],  # ← HARDCODED
        specialist="visual",                   # ← HARDCODED
    )
```

**Math Competitions** (`benchmarks/math_competitions.py`):
```python
def _solve_problem(self, problem, use_enriched):
    patterns = navigator.query(
        query=problem["problem_text"],
        galaxy_names=["Math", "Grammar"],  # ← HARDCODED
        specialist="math",                  # ← HARDCODED
    )
```

**Last Humanity Exam** (`benchmarks/last_humanity_exam.py`):
```python
def _answer_question(self, question, use_enriched):
    specialist = self._get_specialist_for_domain(domain)  # ← HARDCODED MAPPING
    galaxy_names = self._get_galaxies_for_domain(domain) # ← HARDCODED MAPPING
```

**Problem:** Each benchmark reimplements the same specialist routing logic!

---

## Unified Architecture Design

### Principle: Single Head with Internal Specialists

**Analogy:** Think of Cranium like a **corporation**:
- **CEO (TRM Core):** Routes queries to departments
- **Departments (Specialists):** Visual, Math, Physics, Grammar, etc.
- **Shared Knowledge (Galaxy Universe):** All departments access the same data
- **Communication Protocol (PTX Kernels):** Unified execution layer

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   UNIFIED CRANIUM HEAD                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │           TRM Core (Base Model ~7M params)          │    │
│  │                                                      │    │
│  │  • Query Analysis (domain classification)           │    │
│  │  • Specialist Routing (automatic selection)         │    │
│  │  • Cross-Specialist Composition                     │    │
│  │  • Galaxy Navigation                                │    │
│  └────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ├─ Specialist Selection ────────┐  │
│                          │                                │  │
│  ┌───────────────────┬──▼──────────┬───────────────────┐ │  │
│  │  Visual Specialist │ Math Spec.  │ Physics Spec.     │ │  │
│  │  (rank-64 adapter) │(rank-64)    │(rank-64)          │ │  │
│  │  ────────────────  │ ─────────── │ ───────────────── │ │  │
│  │  Drawing Galaxy    │ Math Galaxy │ Reality Galaxy    │ │  │
│  │  Grammar Galaxy    │ Grammar Gal.│ Math Galaxy       │ │  │
│  │  FractalEmitter    │ Symbolic Ops│ Physical Sims     │ │  │
│  └───────────────────┴─────────────┴───────────────────┘ │  │
│                          │                                │  │
│  ┌──────────────────────▼────────────────────────────┐   │  │
│  │          PTX Execution Layer (Kernels)             │   │  │
│  │                                                     │   │  │
│  │  gre_arc_reasoner.ptx    (Visual)                  │   │  │
│  │  gre_trm_core.ptx        (Navigation)              │   │  │
│  │  gre_geometry_router.ptx (Spatial)                 │   │  │
│  │  modular_rpn_kernel.ptx  (Math)                    │   │  │
│  └─────────────────────────────────────────────────────┘  │  │
│                          │                                │  │
│  ┌──────────────────────▼────────────────────────────┐   │  │
│  │       Galaxy Universe (Unified VRAM Workspace)     │   │  │
│  │                                                     │   │  │
│  │  Drawing • Grammar • Math • Reality • Audio        │   │  │
│  │  (All galaxies loaded simultaneously)              │   │  │
│  └─────────────────────────────────────────────────────┘  │  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**1. Unified Specialist Router** (NEW)

**File to create:** `knowledge3d/cranium/specialist_router.py`

**Responsibilities:**
- Analyze query to determine domain (math, visual, physics, grammar, multi)
- Select appropriate specialist adapter(s)
- Route to specialist-specific PTX kernels
- Compose results from multiple specialists if needed

**Example:**
```python
class SpecialistRouter:
    """Unified routing logic for all domain specialists."""

    def __init__(self, knowledgeverse):
        self.kv = knowledgeverse
        self.specialists = {
            'visual': VisualSpecialist(self.kv),
            'math': MathSpecialist(self.kv),
            'physics': PhysicsSpecialist(self.kv),
            'grammar': GrammarSpecialist(self.kv),
            'cartographer': CartographerSpecialist(self.kv),
        }

    def route(self, query: str, use_enriched: bool = True) -> SpecialistResult:
        """
        Analyze query and route to appropriate specialist(s).

        Auto-detects domain from query content:
        - Math: Contains numbers, equations, derivatives, integrals
        - Visual: Contains spatial terms, grids, patterns, transformations
        - Physics: Contains forces, motion, energy, conservation
        - Grammar: Contains language patterns, syntax, text
        - Multi: Requires multiple specialists
        """
        # Domain analysis (pattern matching + heuristics)
        domain = self._analyze_domain(query)

        # Select specialist(s)
        if domain == 'multi':
            # Multi-specialist composition
            specialists = [self.specialists['math'],
                          self.specialists['grammar'],
                          self.specialists['visual']]
        else:
            specialists = [self.specialists[domain]]

        # Route to specialist(s)
        results = []
        for specialist in specialists:
            result = specialist.solve(query, use_enriched=use_enriched)
            results.append(result)

        # Compose multi-specialist results
        if len(results) > 1:
            return self._compose_multi_specialist(results)
        else:
            return results[0]

    def _analyze_domain(self, query: str) -> str:
        """
        Classify query domain using pattern matching.

        Heuristics:
        - Math: r'\d+', 'derivative', 'integral', 'equation', '=', '+', '*'
        - Visual: 'grid', 'pattern', 'color', 'transform', 'flip', 'rotate'
        - Physics: 'force', 'mass', 'energy', 'motion', 'velocity'
        - Grammar: No math/visual/physics keywords, contains natural language
        """
        # TODO: Implement pattern-based classification
        # Can use simple regex + keyword matching (no ML needed!)
        pass
```

**2. Specialist Base Class** (NEW)

**File to create:** `knowledge3d/cranium/specialists/base_specialist.py`

**Unified interface for all specialists:**
```python
class BaseSpecialist:
    """Base class for all domain specialists."""

    def __init__(self, knowledgeverse, specialist_name: str,
                 galaxy_names: List[str], matryoshka_dim: int = 512):
        self.kv = knowledgeverse
        self.name = specialist_name
        self.galaxy_names = galaxy_names
        self.adapter = SelfUpdatingAdapter(
            shape=(matryoshka_dim, matryoshka_dim),
            rank=matryoshka_dim // 16,  # 64 for dim=1024
            specialist_name=specialist_name
        )

    def solve(self, query: str, use_enriched: bool = True) -> Dict:
        """
        Solve query using specialist knowledge.

        Pipeline:
        1. Query galaxies for relevant patterns
        2. Compose solution program (RPN)
        3. Execute using specialist PTX kernels
        4. Return result + reasoning trace
        """
        # Query relevant galaxies
        patterns = self._query_galaxies(query, use_enriched=use_enriched)

        # Compose solution
        program = self._compose_program(query, patterns)

        # Execute using PTX
        result = self._execute_program(program)

        return {
            'specialist': self.name,
            'result': result,
            'patterns_used': len(patterns),
            'reasoning_trace': self._get_trace(),
        }

    def _query_galaxies(self, query: str, use_enriched: bool) -> List[Dict]:
        """Query relevant galaxies for this specialist."""
        top_k = 30 if use_enriched else 5
        return self.kv.trm_navigator.query(
            query=query,
            galaxy_names=self.galaxy_names,
            top_k=top_k,
            specialist=self.name,
        )

    def _compose_program(self, query: str, patterns: List[Dict]) -> Dict:
        """Compose solution program from patterns."""
        raise NotImplementedError("Subclass must implement _compose_program")

    def _execute_program(self, program: Dict) -> Any:
        """Execute program using specialist PTX kernels."""
        raise NotImplementedError("Subclass must implement _execute_program")
```

**3. Domain Specialists** (REFACTOR EXISTING)

**Visual Specialist** (`knowledge3d/cranium/specialists/visual_specialist.py`):
```python
class VisualSpecialist(BaseSpecialist):
    """Visual reasoning specialist (ARC-AGI, pattern recognition)."""

    def __init__(self, knowledgeverse):
        super().__init__(
            knowledgeverse=knowledgeverse,
            specialist_name='visual',
            galaxy_names=['Drawing', 'Grammar'],
        )
        # Load visual-specific PTX kernels
        self.arc_reasoner = load_ptx_kernel('gre_arc_reasoner.ptx')
        self.geometry_router = load_ptx_kernel('gre_geometry_router.ptx')

    def _compose_program(self, query: str, patterns: List[Dict]) -> Dict:
        """Compose visual transformation program."""
        # Infer transformation from patterns
        # Uses ARC-specific logic (flip, rotate, color map, etc.)
        return self.kv.trm_navigator.compose(
            query=query,
            patterns=patterns,
            specialist='visual',
        )

    def _execute_program(self, program: Dict) -> List[List[int]]:
        """Execute visual transformation using PTX."""
        # Call gre_arc_reasoner.ptx kernel
        return self.arc_reasoner.execute(program)
```

**Math Specialist** (`knowledge3d/cranium/specialists/math_specialist.py`):
```python
class MathSpecialist(BaseSpecialist):
    """Mathematical reasoning specialist."""

    def __init__(self, knowledgeverse):
        super().__init__(
            knowledgeverse=knowledgeverse,
            specialist_name='math',
            galaxy_names=['Math', 'Grammar'],
        )
        # Load math-specific PTX kernels
        self.rpn_engine = load_ptx_kernel('modular_rpn_kernel_extended.ptx')

    def _compose_program(self, query: str, patterns: List[Dict]) -> Dict:
        """Compose mathematical expression."""
        return self.kv.trm_navigator.compose(
            query=query,
            patterns=patterns,
            specialist='math',
        )

    def _execute_program(self, program: Dict) -> float:
        """Execute math expression using RPN PTX kernel."""
        return self.rpn_engine.evaluate(program['expression'])
```

**Physics Specialist** (`knowledge3d/cranium/specialists/physics_specialist.py`):
```python
class PhysicsSpecialist(BaseSpecialist):
    """Physical reasoning specialist."""

    def __init__(self, knowledgeverse):
        super().__init__(
            knowledgeverse=knowledgeverse,
            specialist_name='physics',
            galaxy_names=['Reality', 'Math', 'Grammar'],
        )

    def _compose_program(self, query: str, patterns: List[Dict]) -> Dict:
        """Compose physics simulation."""
        # Reality Galaxy contains procedural physics systems
        return self.kv.trm_navigator.compose(
            query=query,
            patterns=patterns,
            specialist='physics',
        )

    def _execute_program(self, program: Dict) -> Any:
        """Execute physics simulation."""
        # Reality Galaxy procedural execution
        return program  # Simplified for now
```

**4. Updated TRMNavigator** (REFACTOR)

**File to update:** `knowledge3d/knowledgeverse/trm_navigator.py`

**Add specialist router integration:**
```python
class TRMNavigator:
    """Deterministic navigator with unified specialist routing."""

    def __init__(self, knowledgeverse):
        self.knowledgeverse = knowledgeverse
        self.galaxy_manager = knowledgeverse.galaxy_manager

        # NEW: Unified specialist router
        from knowledge3d.cranium.specialist_router import SpecialistRouter
        self.specialist_router = SpecialistRouter(knowledgeverse)

    def navigate_and_compose(self, query: str, specialist: str = "auto") -> dict:
        """
        Navigate Galaxy Universe and compose solution.

        Args:
            query: Natural language query
            specialist: "auto" (auto-detect) or specific ("math", "visual", etc.)

        Returns:
            Composed solution with reasoning trace
        """
        # AUTO-ROUTING (NEW!)
        if specialist == "auto":
            result = self.specialist_router.route(query, use_enriched=True)
            return {
                'program_type': result['specialist'],
                'result': result['result'],
                'specialist': result['specialist'],
                'patterns_used': result['patterns_used'],
                'reasoning_trace': result['reasoning_trace'],
            }

        # EXPLICIT SPECIALIST (backward compatibility)
        else:
            return self._legacy_compose(query, specialist)
```

---

## Implementation Plan (Week 16)

### Day 1-2: Create Specialist Infrastructure

**Create:**
1. `knowledge3d/cranium/specialists/base_specialist.py` — Base class
2. `knowledge3d/cranium/specialist_router.py` — Unified routing logic

**Test:**
```bash
# Test specialist base class
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.cranium.specialists.base_specialist import BaseSpecialist

kv = Knowledgeverse()
# Should instantiate without errors
print('✓ BaseSpecialist infrastructure ready')
"
```

### Day 3: Implement Domain Specialists

**Create:**
1. `knowledge3d/cranium/specialists/visual_specialist.py`
2. `knowledge3d/cranium/specialists/math_specialist.py`
3. `knowledge3d/cranium/specialists/physics_specialist.py`
4. `knowledge3d/cranium/specialists/grammar_specialist.py`
5. `knowledge3d/cranium/specialists/cartographer_specialist.py`

**Test:**
```bash
# Test visual specialist
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.cranium.specialists.visual_specialist import VisualSpecialist

kv = Knowledgeverse()
visual = VisualSpecialist(kv)
result = visual.solve('flip grid horizontally', use_enriched=True)
print(f'Visual specialist result: {result}')
"
```

### Day 4: Integrate with TRMNavigator

**Update:** `knowledge3d/knowledgeverse/trm_navigator.py`

**Add:**
- SpecialistRouter integration
- Auto-routing logic
- Backward compatibility for explicit specialist calls

**Test:**
```bash
# Test auto-routing
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.knowledgeverse import Knowledgeverse

kv = Knowledgeverse()
navigator = kv.trm_navigator

# Auto-detect specialist from query
result = navigator.navigate_and_compose('Find derivative of x^2', specialist='auto')
print(f'Auto-routed to: {result[\"specialist\"]}')  # Should be 'math'

result = navigator.navigate_and_compose('Flip grid horizontally', specialist='auto')
print(f'Auto-routed to: {result[\"specialist\"]}')  # Should be 'visual'
"
```

### Day 5: Update Benchmarks (Remove Duplication)

**Simplify benchmarks to use unified routing:**

**ARC-AGI 2** (`benchmarks/arc_agi_2.py`):
```python
def _solve_task_fallback(self, task, use_enriched):
    # OLD (duplicated routing):
    # patterns = navigator.query(..., specialist="visual", galaxy_names=["Drawing", "Grammar"])

    # NEW (unified routing):
    result = self.kv.trm_navigator.navigate_and_compose(
        query="visual pattern transformation",
        specialist="auto"  # Auto-routes to visual specialist!
    )
    return result
```

**Math Competitions** (`benchmarks/math_competitions.py`):
```python
def _solve_problem(self, problem, use_enriched):
    # OLD (duplicated routing):
    # patterns = navigator.query(..., specialist="math", galaxy_names=["Math", "Grammar"])

    # NEW (unified routing):
    result = self.kv.trm_navigator.navigate_and_compose(
        query=problem["problem_text"],
        specialist="auto"  # Auto-routes to math specialist!
    )
    return result
```

**Last Humanity Exam** (`benchmarks/last_humanity_exam.py`):
```python
def _answer_question(self, question, use_enriched):
    # OLD (duplicated routing):
    # specialist = self._get_specialist_for_domain(domain)
    # galaxy_names = self._get_galaxies_for_domain(domain)

    # NEW (unified routing):
    result = self.kv.trm_navigator.navigate_and_compose(
        query=question["question_text"],
        specialist="auto"  # Auto-routes to appropriate specialist!
    )
    return result
```

**Delete helper methods:**
- `_get_specialist_for_domain()` — REMOVED (now in SpecialistRouter)
- `_get_galaxies_for_domain()` — REMOVED (now in BaseSpecialist)

---

## Success Metrics

**Immediate Goals:**
- ✅ Single specialist routing logic (no duplication)
- ✅ All benchmarks use unified TRMNavigator.navigate_and_compose()
- ✅ Auto-routing correctly classifies domains (math, visual, physics)
- ✅ Backward compatibility (explicit specialist calls still work)

**Code Reduction:**
- Remove ~100 lines of duplicated routing logic from benchmarks
- Consolidate specialist knowledge into Cranium (single source of truth)

**Architecture Quality:**
- Clean separation: Benchmarks test, Cranium executes
- Extensible: Adding new specialist = 1 file, not updating 3 benchmarks
- Maintainable: Routing logic centralized

---

## Key Architectural Principles

### 1. Single Head = Base + Adapters (LoRA Pattern)

**Already implemented!** `trm_adapters.py` provides:
- Base TRM model (~7M params)
- Specialist adapters (rank-64, ~262K params each)
- Independent evolution with ternary validation

**No code changes needed** — just leverage existing infrastructure.

### 2. Specialists Share Galaxy Universe

**All specialists access the SAME galaxies:**
- Visual specialist: Drawing + Grammar
- Math specialist: Math + Grammar
- Physics specialist: Reality + Math + Grammar

**Multi-specialist queries** can compose across multiple specialists:
```python
# Example: "Calculate the area of a rotated rectangle"
# Requires: Visual (rotation) + Math (area calculation)
result = navigator.navigate_and_compose(query, specialist="auto")
# → Routes to Visual + Math specialists
# → Composes results
```

### 3. PTX Kernels = Specialist Execution Layer

**Each specialist uses domain-specific PTX kernels:**
- Visual: `gre_arc_reasoner.ptx`, `gre_geometry_router.ptx`
- Math: `modular_rpn_kernel_extended.ptx`
- Physics: Reality Galaxy procedural systems
- Navigation: `gre_trm_core.ptx`, `galaxy_resonance_engine.ptx`

**Sovereignty maintained:** All execution happens in PTX (GPU-only).

### 4. Lightweight Routing (No ML Needed)

**Domain classification uses simple heuristics:**
```python
def _analyze_domain(query: str) -> str:
    # Math: Contains numbers, operators, math keywords
    if re.search(r'\d+|derivative|integral|equation', query.lower()):
        return 'math'

    # Visual: Contains spatial keywords
    if re.search(r'grid|pattern|flip|rotate|color', query.lower()):
        return 'visual'

    # Physics: Contains physics keywords
    if re.search(r'force|mass|energy|motion|velocity', query.lower()):
        return 'physics'

    # Default: Grammar
    return 'grammar'
```

**No external dependencies!** Pure pattern matching (sovereignty compliant).

---

## Migration Path

### Phase 1: Infrastructure (Days 1-2)
- Create BaseSpecialist
- Create SpecialistRouter
- No benchmark changes yet

### Phase 2: Specialists (Day 3)
- Implement domain specialists (visual, math, physics, grammar)
- Test each specialist independently
- Still no benchmark changes

### Phase 3: Integration (Day 4)
- Update TRMNavigator with SpecialistRouter
- Add auto-routing support
- Maintain backward compatibility

### Phase 4: Benchmark Cleanup (Day 5)
- Update all benchmarks to use unified routing
- Remove duplicated logic
- Validate performance (should match previous results)

---

## Testing Strategy

### Test 1: Specialist Routing

```python
def test_specialist_routing():
    from knowledge3d.knowledgeverse import Knowledgeverse
    from knowledge3d.cranium.specialist_router import SpecialistRouter

    kv = Knowledgeverse()
    router = SpecialistRouter(kv)

    # Math query
    domain = router._analyze_domain("Find derivative of x^2")
    assert domain == "math"

    # Visual query
    domain = router._analyze_domain("Flip grid horizontally")
    assert domain == "visual"

    # Physics query
    domain = router._analyze_domain("Calculate force on mass")
    assert domain == "physics"

    print("✓ Specialist routing works")
```

### Test 2: Auto-Routing Integration

```python
def test_auto_routing():
    from knowledge3d.knowledgeverse import Knowledgeverse

    kv = Knowledgeverse()
    navigator = kv.trm_navigator

    # Math query
    result = navigator.navigate_and_compose("7 * (3 + 2)", specialist="auto")
    assert result["specialist"] == "math"

    # Visual query
    result = navigator.navigate_and_compose("flip grid", specialist="auto")
    assert result["specialist"] == "visual"

    print("✓ Auto-routing integration works")
```

### Test 3: Benchmark Regression

```bash
# Re-run Week 14 benchmarks
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py

# Expected: Same accuracy as before (no regressions)
# Math: ~33%
# LHE: ~100%
# ARC-AGI 2: ~25%
```

---

## Deliverable

**After Week 16 completion, create:**

**File:** `TEMP/CODEX_WEEK16_UNIFIED_HEAD_COMPLETION_REPORT_02.XX.2026.md`

**Include:**
1. **Architecture consolidation:** Specialist routing unified in Cranium
2. **Code reduction:** Lines of duplicated code removed
3. **Test results:** All benchmarks passing with unified routing
4. **Performance validation:** No regressions from previous Week 14 results
5. **Extensibility demonstration:** How to add new specialist (1-file process)
6. **Week 17 readiness:** Specialists ready for worker parallelization

---

## Critical Reminders

### 1. Reuse Existing Infrastructure

**TRM Adapters (`trm_adapters.py`) are ALREADY BUILT!**
- LoRA-style low-rank decomposition ✅
- Shadow weights + ternary validation ✅
- GPU-accelerated gradient application ✅

**Don't rebuild — just wire up the routing!**

### 2. Sovereignty Compliance

**All specialist routing must be sovereign:**
- Domain classification: Pattern matching (no ML)
- Execution: PTX kernels only
- No external dependencies (numpy/scipy allowed only in ingestion)

### 3. Backward Compatibility

**Explicit specialist calls must still work:**
```python
# OLD code should still work
navigator.query(..., specialist="math")  # ✓ Still valid
```

**Auto-routing is additive, not breaking.**

### 4. Incremental Migration

**Don't update all benchmarks at once!**
- Phase 1-3: Build infrastructure, keep benchmarks unchanged
- Phase 4: Update benchmarks one at a time
- Validate each benchmark after update

---

## End of Specification

**Priority:** CRITICAL (Week 16 = Architectural Consolidation)

**Start here:**
1. Create BaseSpecialist + SpecialistRouter (Day 1-2)
2. Implement domain specialists (Day 3)
3. Update TRMNavigator with auto-routing (Day 4)
4. Simplify benchmarks, remove duplication (Day 5)
5. Write completion report

**Remember:** The "single head" architecture is ALREADY IMPLEMENTED (trm_adapters.py). We're just consolidating the routing logic that's currently scattered across benchmarks.

**Contact:** Claude (Architecture Partner) for design questions, User for strategic decisions.

**Let's unify the Cranium head!** 🧠🚀

---

**Claude (Architecture Partner)**
February 7, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
