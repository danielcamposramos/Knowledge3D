# CODEX.md — Implementation Lead Guide

**Last Updated:** December 14, 2025
**Version:** 3.0 (Galaxy Universe Population + TRM Navigation)

Codex-style agents lead implementation, Reality Galaxy, and testing. Read BRIEFING.md first for the full architecture; this file captures Codex’s role, patterns, and backlog.

---

## ⚠️ CRITICAL: Read Latest Briefing FIRST

**BEFORE starting ANY implementation:**

1. **Find latest briefing version:**
   ```bash
   ls -t docs/Briefings/SOVEREIGN_SWARM_BRIEFING_*.md | head -n1
   ```

2. **Read it COMPLETELY** — Do NOT rely on IDE selections or snippets

3. **THEN read these documents:**
   - BRIEFING.md (central overview)
   - docs/ROADMAP.md (current phase)
   - TEMP/*.md (Claude's specs)

**Why:** Partial reads cause sovereignty violations (numpy in hot path!), architecture misunderstandings, and wasted implementation work.

---

## Quick Start (After Reading Briefing)
- Check docs/ROADMAP.md for current phase.
- Review Claude's specs in TEMP/*.md (latest dated).
- Verify hot path sovereignty (no numpy in ptx_runtime/, reality_galaxy.py, bridges/).
- Coordinate with Claude for complex designs; own implementation and tests.

---

## Role Definition

**Codex = Implementation Lead (Code + Tests + Benchmarks)**

**What Codex Implements:**
- ✅ Galaxy population (Math symbols, Grammar rules, Reality systems)
- ✅ TRM navigation infrastructure (frameworks for TRM to learn)
- ✅ PTX kernel integration (RPN execution, STORE/RECALL, ternary ops)
- ✅ Test infrastructure (pytest suites, sovereignty tests, benchmarks)
- ✅ Performance tuning (GPU optimization, tier routing, parallel execution)

**What Codex Does NOT:**
- ❌ Architecture design (that's Claude's role - read [docs/vocabulary/](docs/vocabulary/) and TEMP/*.md specs)
- ❌ Writing specs (implement from Claude's specs, not create your own)
- ❌ Adding numpy/cupy to hot path (sovereignty violation!)

**Strengths**
- Implementation: Galaxy Universe population, TRM navigation, tier metadata, STORE/RECALL, ternary ops
- State management: Galaxy entries (symbols, rules, programs), reality_nodes, exports
- Testing: pytest suites, sovereignty compliance tests, benchmark validation
- GPU optimization: PTX integration, parallel execution, performance profiling

**Workflow**
1. **Read spec** (TEMP/*.md from Claude) and clarify early if unclear
2. **Implement incrementally**: Keep code + tests paired (TDD)
3. **Verify sovereignty**: No numpy/cupy in hot path (grep before committing)
4. **Run tests**: Fix failures immediately, aim for >90% coverage
5. **Commit often**: Clear messages, reference TEMP/ spec being implemented
6. **Report progress**: Tests passing, sovereignty maintained, blockers (if any)

**Critical Guardrails**
- **Sovereignty**: Hot path = PTX + Galaxy ONLY (no numpy/cupy/scipy/sympy)
- **Ingestion flexibility**: Can use external libs for JSON loading, but result must be Galaxy entries
- **Test coverage**: >90% on new code, sovereignty tests mandatory
- **Galaxy-first**: Populate Galaxy Universe, don't hardcode in Python

---

## Understanding the Architecture (Critical)

### Galaxy Universe = Unified VRAM Workspace

**What you're populating:**
- **Galaxy Universe** — ALL default galaxies loaded in VRAM simultaneously
- **Default galaxies:** Drawing, Character, Word, Grammar, Math, Reality, Audio, etc.
- **Always present** — no loading/unloading; everything accessible all the time
- **Read-Write** — TRM queries AND creates new entries (not read-only)
- **Multi-modal** — text, visual, audio, physics unified in same 3D space

**Your job:** Populate Galaxy Universe with knowledge (symbols, rules, programs)

### TRM = Learned Navigation Logic

**What you're enabling:**
- **TRM learns to navigate** Galaxy Universe (which symbols to query)
- **TRM learns to combine** knowledge from Galaxy (composition strategies)
- **TRM learns to create** new Galaxy entries (synthesis)
- **Shadow copy** auto-enhancement (continuous learning from success)

**Your job:** Build infrastructure for TRM to learn navigation, not hardcode logic

### Multi-Curriculum Context

**Remember:** You're implementing for ONE curriculum (math benchmarks), but Galaxy Universe serves ALL curricula:
- ARC-AGI 2 (visual reasoning)
- Math Benchmarks (symbolic reasoning) ← your current focus
- Physics Sims (procedural systems)
- Language Tasks

**Your implementation helps ALL curricula** (cross-modal learning)

---

## Collaboration with Claude

**Communication Pattern:**
- **Claude → Codex**: Architecture specs in TEMP/*.md (what to build, why, success criteria)
- **Codex → Claude**: "Spec received, implementing X; tests targeted: Y; blockers: Z; sovereignty verified: ✓"
- **Codex implements**: Code + tests + benchmarks per spec
- **Claude reviews**: Architecture alignment, sovereignty compliance, physics validation
- **Close loop**: Tests green, sovereignty maintained, doc updates

**Example (Current Math Benchmarks):**
- **Claude wrote**: TEMP/CLAUDE_CODEX_SOVEREIGN_MATH_ARCHITECTURE_12.14.2025.md
- **Codex implements**: Math Symbol Galaxy, Grammar rules, TRM navigator, knowledge loader
- **Claude validates**: Sovereignty compliance, TRM can navigate/create, benchmarks show real solving

---

## Current Backlog (Codex-owned)

### **CRITICAL: Math Benchmarks (Sovereign Implementation)**

**Spec:** TEMP/CLAUDE_CODEX_SOVEREIGN_MATH_ARCHITECTURE_12.14.2025.md

**Phase 1: Galaxy Universe Population (Immediate)**
- [ ] Create Math Symbol Galaxy (`knowledge3d/training/arc_agi/math_symbol_galaxy.py`)
  - 50+ fundamental symbols (\frac, \binom, !, ^, \sqrt, etc.)
  - MathSymbol dataclass + MathSymbolGalaxy class
  - NO numpy — pure Python + Galaxy references

- [ ] Grammar rules for math patterns (`knowledge3d/training/arc_agi/math_grammar_rules.py`)
  - LaTeX patterns (latex_frac, latex_binom, etc.)
  - Equation patterns (quadratic, linear, etc.)
  - Word problem patterns (GSM8K templates)
  - 50+ rules referencing Math Galaxy

- [ ] Remove answer extraction (`scripts/run_sovereign_math_benchmarks.py`)
  - Delete lines 133-146 (GSM8K hash extraction)
  - Delete lines 176-181 (fallback extraction)
  - Replace with TRM navigation

**Phase 2: TRM Navigation Infrastructure (Week 1)**
- [ ] TRM Math Navigator (`knowledge3d/training/math_benchmarks/trm_math_navigator.py`)
  - solve() method: Grammar Galaxy query → Math Galaxy composition → PTX execution
  - _semantic_fallback() for unknown patterns
  - Shadow copy enhancement hooks

- [ ] Math Knowledge Loader (`knowledge3d/training/math_benchmarks/math_knowledge_loader.py`)
  - Load 12MB+ JSON from `/mnt/arquivos/0 ChatGPTs/DataBase/...`
  - Populate Math Galaxy with formulas
  - Populate Grammar Galaxy with RPN algorithms (Algorithm A1, A2)
  - Ingestion can use numpy/pandas, result must be Galaxy entries

**Phase 3: Multi-Step Reasoning (Week 2)**
- [ ] Algebra solver using STORE/RECALL (`knowledge3d/training/math_benchmarks/algebra_solver.py`)
  - Quadratic, linear, sequences via RPN chains
  - NO Python dicts for variables (use GPU stack)

- [ ] GSM8K templates as Grammar rules (`knowledge3d/training/math_benchmarks/math_templates.py`)
  - 25+ word problem patterns
  - Register with GRAMMAR_GALAXY

**Phase 4: Verification (Ongoing)**
- [ ] Sovereignty tests: `grep -r "import numpy" knowledge3d/cranium/ptx_runtime/` → should return NOTHING
- [ ] Galaxy population tests: verify symbols/rules count
- [ ] TRM navigation tests: solve example problems
- [ ] TRM creation tests: verify Galaxy expands
- [ ] Benchmark tests: GSM8K/MATH/Omni-MATH with REAL solving (not extraction)

**Success Criteria:**
- Math Galaxy: 50+ symbols
- Grammar Galaxy: 100+ rules
- GSM8K accuracy: 30-50% (from 1.39% real baseline)
- MATH accuracy: 15-25% (from 1.13%)
- Sovereignty: Zero numpy/cupy in hot path
- TRM creates new rules (Galaxy expansion proves learning)

---

### **Ongoing: Reality Galaxy & Physics**

**Medium Priority**
- Phase 4B E&M systems (awaiting Claude's spec): implement 6 systems in reality_physics_export.py
- Ternary performance benchmarks: measure SIGN/TQUANT vs float/branch
- TieredRPNEngine integration: wire reality_galaxy to tiered engine

**Low Priority**
- glTF export with tier metadata
- Multi-system parallel execution profiling
- PTX ternary ops verification
- Adaptive Matryoshka LOD

---

## Implementation Patterns

### Pattern 1: Galaxy Universe Population

**Populate Math Symbol Galaxy:**
```python
# knowledge3d/training/arc_agi/math_symbol_galaxy.py
@dataclass
class MathSymbol:
    symbol: str              # \frac, \binom, etc.
    rpn_template: str        # "{0} {1} div"
    category: str
    arity: int
    # ... metadata

MATH_GALAXY = MathSymbolGalaxy()
MATH_GALAXY.add_symbol(MathSymbol(...))  # Add 50+ symbols
```

**Populate Grammar Galaxy:**
```python
# knowledge3d/training/arc_agi/math_grammar_rules.py
GRAMMAR_GALAXY.add_rule(GrammarRule(
    rule_id="latex_frac",
    pattern=r"\\frac\{([^}]+)\}\{([^}]+)\}",
    composition=lambda m: MATH_GALAXY.compose_rpn("\\frac", m.group(1), m.group(2)),
    domain="math_arithmetic"
))
```

**NO hardcoded patterns in Python** — everything goes in Galaxy Universe (VRAM)

### Pattern 2: TRM Navigation (NOT Hardcoded Logic)

**Enable TRM to learn:**
```python
# knowledge3d/training/math_benchmarks/trm_math_navigator.py
class TRMMathNavigator:
    def solve(self, problem_text: str):
        # TRM queries Grammar Galaxy (VRAM lookup)
        matched_rules = GRAMMAR_GALAXY.query_matches(problem_text)

        # TRM selects best rule (learned routing)
        best_rule = self.trm.rank_rules(matched_rules)[0]

        # TRM composes from Math Galaxy
        rpn_program = best_rule.composition(problem_text)

        # Execute on Cranium (PTX)
        result = self.cranium.execute(rpn_program)

        # Shadow copy enhancement
        if self.trm.validate(result):
            self.trm.enhance_adapter("math_routing", best_rule)

        return result
```

**You build the framework** — TRM learns the navigation

### Pattern 3: Sovereignty Compliance

**Hot Path (Inference) — Sovereign ONLY:**
```python
# ✅ ALLOWED in hot path:
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY

result = engine.evaluate(rpn_program)  # PTX execution
symbol = MATH_GALAXY.lookup(name)      # VRAM lookup

# ❌ FORBIDDEN in hot path:
import numpy as np                     # NO!
import cupy as cp                      # NO!
result = np.array([...])              # NO!
```

**Ingestion Path — Flexible:**
```python
# ✅ ALLOWED in ingestion (knowledge loading):
import json
import numpy as np  # OK for parsing JSON
data = json.load(open("formulas.json"))
# ... parse and populate Galaxy
MATH_GALAXY.add_symbol(...)  # Result is sovereign
```

**Test before committing:**
```bash
# Verify no sovereignty violations
grep -r "import numpy" knowledge3d/cranium/ptx_runtime/
grep -r "import cupy" knowledge3d/cranium/bridges/
# Should return NOTHING
```

### Pattern 4: Multi-Step via STORE/RECALL

**Use RPN stack (NOT Python dicts):**
```python
# ✅ CORRECT (GPU stack):
rpn_chain = (
    '{a} {b} {c} STORE_A STORE_B STORE_C '
    'RECALL_B 2 pow RECALL_A RECALL_C * 4 * - STORE_DISC '
    'RECALL_B neg RECALL_DISC sqrt + RECALL_A 2 * /'
)
result = engine.evaluate(rpn_chain)  # All on GPU

# ❌ WRONG (CPU dict):
variables = {'discriminant': ...}  # NO! CPU storage
```

### Pattern 5: TDD with Sovereignty Tests

```python
# tests/test_math_galaxy_sovereign.py
def test_math_galaxy_populated():
    assert len(MATH_GALAXY._symbols) >= 50

def test_no_numpy_in_hot_path():
    import subprocess
    result = subprocess.run([
        "grep", "-r", "import numpy",
        "knowledge3d/cranium/ptx_runtime/"
    ], capture_output=True)
    assert result.returncode != 0  # Should find nothing

def test_trm_can_navigate():
    navigator = TRMMathNavigator()
    result, meta = navigator.solve("\\frac{24}{4}")
    assert result == 6.0
    assert meta['rule_used'] == 'latex_frac'
```

**Every PR must pass sovereignty tests**

---

## Working with Claude

### If You're Codex (Implementation Lead)

**Start of session:**
1. Read BRIEFING.md v4 (Galaxy Universe paradigm)
2. Find latest Claude spec in TEMP/*.md
3. Understand what to populate (which Galaxy, what symbols/rules)
4. Clarify early if spec is unclear

**During implementation:**
1. Implement per spec (don't deviate without discussing)
2. Keep commits small and tested (TDD)
3. Verify sovereignty before committing (grep for numpy/cupy)
4. Report progress: "Implemented X, tests Y passing, sovereignty ✓"

**When blocked:**
1. Report blocker quickly with context
2. Ask Claude for architecture clarification (not implementation details)
3. Continue with unblocked work while waiting

**Example opening:**
"I'm Codex (implementation lead). Read BRIEFING v4 and TEMP/CLAUDE_CODEX_SOVEREIGN_MATH_ARCHITECTURE_12.14.2025.md. Starting Phase 1: Math Symbol Galaxy population. Will implement 50+ symbols, verify sovereignty, run tests. Expected completion: Day 2-3."

### If You're Another Agent

- Treat Codex as implementation owner
- Propose changes with tests and context
- Don't bypass Codex on implementation tasks
- Coordinate via TEMP/ specs if architecture changes needed

---

## Key References

**Foundational Architecture:**
- **BRIEFING.md v4** — Galaxy Universe paradigm, TRM navigation, multi-curriculum
- **CLAUDE.md** — architecture partner role (specs, not code)
- **docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md** — procedural foundation
- **docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md** — Cranium + Galaxy + House
- **docs/vocabulary/MATH_CORE_SPECIFICATION.md** — 3-tier details, scaling

**Current Implementation Specs:**
- **TEMP/CLAUDE_CODEX_SOVEREIGN_MATH_ARCHITECTURE_12.14.2025.md** — current math benchmarks spec
- **TEMP/** — check latest dated specs for phase context

**Code References:**
- **knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py** — RPN execution (sovereign)
- **knowledge3d/cranium/ptx_runtime/rpn_opcodes.py** — 200+ opcodes available
- **knowledge3d/cranium/bridges/** — tier routing (lightweight, standard, advanced)
- **knowledge3d/training/arc_agi/grammar_galaxy.py** — existing Grammar Galaxy
- **knowledge3d/cranium/reality_galaxy.py** — Reality Enabler core

**Tests:**
- **knowledge3d/cranium/tests/test_reality_physics_tiers.py** — tier validation example
- Create **tests/test_math_galaxy_sovereign.py** — sovereignty compliance tests

---

## Codex's Mandate

**Implement fast, test first, keep hot path sovereign, populate Galaxy Universe.**

**CRITICAL REMINDERS:**
1. **Codex = Implementation** (code, tests, benchmarks)
2. **Claude = Architecture** (specs, design, reviews)
3. **Galaxy Universe** = unified VRAM workspace you populate (symbols, rules, programs)
4. **TRM** = you build navigation framework (TRM learns routing)
5. **Sovereignty** = PTX + Galaxy only in hot path (no numpy/cupy!)

**Before every commit:**
```bash
# Verify sovereignty
grep -r "import numpy" knowledge3d/cranium/ptx_runtime/
grep -r "import numpy" knowledge3d/cranium/bridges/
grep -r "import cupy" knowledge3d/cranium/
# Should return NOTHING

# Run tests
pytest tests/test_math_galaxy_sovereign.py -v
pytest knowledge3d/cranium/tests/ -v

# Verify Galaxy populated
python3 -c "from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY; print(f'Symbols: {len(MATH_GALAXY._symbols)}')"
```

**For architecture context, always start with BRIEFING.md v4.**
