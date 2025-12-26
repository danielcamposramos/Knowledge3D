# CLAUDE → CODEX: Math Shadow Copy Extension

**Date:** December 15, 2025
**Priority:** HIGH - Enables TRM Learning for Math Benchmarks
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## Executive Summary

Extend the existing `DualShadowCopy` infrastructure to support Math Galaxy, enabling TRM to learn from successful math problem solving. The infrastructure already exists for ARC-AGI (Drawing + Grammar); we need to wire it for math benchmarks.

**Key Insight:** Don't reinvent the wheel. `DualShadowCopy` already implements:
- Quality-filtered recording
- Deduplication
- Pattern confidence tracking
- Staged commits (Tesla-inspired)
- SleepTime consolidation

We just need to connect TRM Math Navigator to this existing system.

---

## 1. Existing Infrastructure (DO NOT REWRITE)

### 1.1 DualShadowCopy (274 lines)

File: `knowledge3d/training/arc_agi/dual_shadow_copy.py`

**Key Methods:**
```python
class DualShadowCopy:
    def record(self, task_signature, program, program_type, score, **kwargs):
        """Records discovery with quality filtering + deduplication."""

    def _commit_entry(self, entry):
        """Commits to Drawing + Grammar galaxies based on program_type."""

    def update_pattern_confidence(self, pattern_id, confidence):
        """Tracks pattern success rates for calibration."""
```

**Program Types:**
- `"visual"` → commits to Drawing Galaxy
- `"transformation"` → commits to Grammar Galaxy
- `"hybrid"` → commits to both

### 1.2 SleepTimeConsolidator (278 lines)

File: `knowledge3d/training/arc_agi/sleeptime_consolidator.py`

**Key Methods:**
```python
class SleepTimeConsolidator:
    def consolidate(self):
        """Prunes low-quality, promotes canonical patterns."""

    def _promote_canonical_patterns(self, rule_stats):
        """Promotes high-success rules to canonical status."""
```

---

## 2. Required Changes

### 2.1 Add Math Program Type to DualShadowCopy

**File:** `knowledge3d/training/arc_agi/dual_shadow_copy.py`

**Change:** Extend `_commit_entry()` to handle `"math"` program type.

```python
def _commit_entry(self, entry: Dict) -> None:
    program_type = entry["program_type"]
    program = entry["program"]
    signature = entry["task_signature"]

    if program_type == "visual":
        # ... existing code ...

    elif program_type == "transformation":
        # ... existing code ...

    elif program_type == "math":
        # NEW: Commit to Grammar Galaxy with math domain
        rule_id = f"DISCOVERED_MATH_RULE_{len(self.grammar.rules)}"
        self.grammar.rules[rule_id] = GrammarRule(
            rule_id=rule_id,
            language="math",
            domain=entry.get("domain", "math_general"),
            pattern=entry.get("pattern", "discovered"),
            rpn_program=program,
            examples=[signature],
            description=f"Discovered math rule: {signature.get('problem_type', 'unknown')}",
            semantics=entry.get("semantic_context", {}),
        )

    else:  # hybrid
        # ... existing code ...
```

### 2.2 Wire TRM Navigator to Shadow Copy

**File:** `knowledge3d/training/math_benchmarks/trm_math_navigator.py`

**Change:** Replace stub `enhance_adapter()` with real shadow copy recording.

```python
class TRMMathNavigator:
    def __init__(
        self,
        *,
        rule_bank: Sequence[Any],
        math_galaxy: Any,
        rpn_engine: Any,
        trm_engine: Optional[Any] = None,
        shadow_copy: Optional["DualShadowCopy"] = None,  # NEW
    ) -> None:
        self.rule_bank = list(rule_bank)
        self.math_galaxy = math_galaxy
        self.engine = rpn_engine
        self.trm = trm_engine or HeuristicTRMMathEngine()
        self.shadow = shadow_copy  # NEW
```

**Change:** Implement `enhance_adapter()` to record to shadow copy.

```python
def _record_success(
    self,
    rule: Any,
    rpn_program: str,
    result: Any,
    problem_text: str,
    confidence: float,
) -> None:
    """Record successful solve to shadow copy for learning."""
    if self.shadow is None:
        return

    task_signature = {
        "problem_text": problem_text[:200],  # Truncate for storage
        "rule_id": getattr(rule, "rule_id", "unknown"),
        "result": str(result),
        "problem_type": getattr(rule, "domain", "math_general"),
    }

    self.shadow.record(
        task_signature=task_signature,
        program=rpn_program,
        program_type="math",
        score=confidence,
        task_id=f"math_{hash(problem_text) % 10000}",
    )

    # Update pattern confidence for calibration
    pattern_id = getattr(rule, "rule_id", None)
    if pattern_id:
        self.shadow.update_pattern_confidence(pattern_id, confidence)
```

**Change:** Call `_record_success()` in `solve()` method.

```python
def solve(self, problem_text: str) -> Tuple[Any, Dict[str, Any]]:
    # ... existing code up to line 122 ...

    confidence = self.trm.validate_result(result, problem_text)
    if confidence > 0.8:
        # Record to shadow copy (replaces stub enhance_adapter)
        self._record_success(
            rule=best.rule,
            rpn_program=rpn_program,
            result=result,
            problem_text=problem_text,
            confidence=confidence,
        )

    return (result, metadata)
```

### 2.3 Wire Benchmark Runner to Pass Shadow Copy

**File:** `scripts/run_sovereign_math_benchmarks.py`

**Change:** Create shadow copy and pass to TRM navigator.

```python
class SovereignBenchmarkRunner:
    def __init__(self, *, use_trm_navigator: bool = False):
        # ... existing code ...

        self._shadow_copy = None
        if use_trm_navigator:
            # Import shadow copy infrastructure
            from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
            from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
            from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy

            # Create minimal galaxies for math (we only use Grammar)
            drawing_galaxy = DrawingGalaxy()
            grammar_galaxy = GrammarGalaxy()

            self._shadow_copy = DualShadowCopy(
                drawing_galaxy, grammar_galaxy, staged=True
            )

            # Load existing checkpoint if available
            checkpoint_path = Path("/K3D/Knowledge3D.local/checkpoints/math_benchmarks/shadow_copy.json")
            if checkpoint_path.exists():
                self._shadow_copy.load(checkpoint_path)

            # Wire to TRM navigator
            from knowledge3d.training.math_benchmarks.trm_math_navigator import TRMMathNavigator

            self._trm_navigator = TRMMathNavigator(
                rule_bank=math_grammar_rules.SOVEREIGN_MATH_RULES,
                math_galaxy=MATH_GALAXY,
                rpn_engine=self.engine,
                shadow_copy=self._shadow_copy,  # NEW
            )
```

**Change:** Save shadow copy after benchmark run.

```python
def run(self, datasets: List[str], max_problems: int = 100) -> Dict[str, Any]:
    # ... existing code ...

    # Save shadow copy checkpoint
    if self._shadow_copy is not None:
        checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/math_benchmarks")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._shadow_copy.save(checkpoint_dir / "shadow_copy.json")

        # Commit pending discoveries
        self._shadow_copy.commit_pending()

        print(f"\n[SHADOW COPY] Recorded {len(self._shadow_copy.library)} discoveries")

    return results
```

---

## 3. Implementation Checklist

### Phase 2A: Shadow Copy Wiring (Days 1-2)

- [ ] Add `"math"` program type to `DualShadowCopy._commit_entry()`
- [ ] Add `shadow_copy` parameter to `TRMMathNavigator.__init__()`
- [ ] Implement `_record_success()` method in TRM navigator
- [ ] Call `_record_success()` on confidence > 0.8 in `solve()`
- [ ] Wire shadow copy in `SovereignBenchmarkRunner.__init__()`
- [ ] Add checkpoint save/load in benchmark runner

### Phase 2B: Testing (Day 3)

- [ ] Test: Math rule recorded to shadow copy
- [ ] Test: Pattern confidence updated on success
- [ ] Test: Checkpoint persists across runs
- [ ] Test: Discovered rules appear in Grammar Galaxy

### Phase 2C: Consolidation (Day 4)

- [ ] Add math rule analysis to SleepTimeConsolidator
- [ ] Test: Low-quality math rules pruned
- [ ] Test: High-success math rules promoted to canonical

---

## 4. Test Cases

### 4.1 Shadow Copy Recording

```python
def test_trm_navigator_records_to_shadow_copy():
    from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
    from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.math_benchmarks.trm_math_navigator import TRMMathNavigator

    drawing = DrawingGalaxy()
    grammar = GrammarGalaxy()
    shadow = DualShadowCopy(drawing, grammar, staged=True)

    nav = TRMMathNavigator(
        rule_bank=SOVEREIGN_MATH_RULES,
        math_galaxy=MATH_GALAXY,
        rpn_engine=_EchoEngine(),
        shadow_copy=shadow,
    )

    # Solve a problem that should succeed
    result, meta = nav.solve("\\frac{24}{4}")

    # Shadow copy should have recorded the discovery
    assert len(shadow.library) >= 1
    assert shadow.library[0]["program_type"] == "math"
```

### 4.2 Pattern Confidence Tracking

```python
def test_pattern_confidence_updated():
    # ... setup ...

    # Solve same pattern multiple times
    nav.solve("\\frac{10}{2}")
    nav.solve("\\frac{20}{4}")
    nav.solve("\\frac{30}{6}")

    # Pattern confidence should be tracked
    conf = shadow.get_pattern_success_rate("latex_frac")
    assert conf is not None
    assert conf > 0.5
```

### 4.3 Checkpoint Persistence

```python
def test_shadow_copy_persists():
    # ... setup + solve problems ...

    # Save checkpoint
    shadow.save(Path("/tmp/test_shadow_copy.json"))

    # Load in new instance
    shadow2 = DualShadowCopy(DrawingGalaxy(), GrammarGalaxy(), staged=True)
    shadow2.load(Path("/tmp/test_shadow_copy.json"))

    # Should have same entries
    assert len(shadow2.library) == len(shadow.library)
```

---

## 5. Sovereignty Notes

### 5.1 Shadow Copy is NOT Hot Path

Shadow copy operations happen AFTER inference completes:
- `_record_success()` called after RPN execution
- `save()` called at end of benchmark run
- Consolidation runs as separate SleepTime phase

**This means:** numpy/json/file I/O in shadow copy is acceptable.

### 5.2 Hot Path Remains Sovereign

The hot path (inference) is unchanged:
1. TRM Navigator queries Grammar Galaxy → VRAM lookup
2. Composes RPN from Math Galaxy → string manipulation
3. Executes on RPN engine → PTX kernels
4. Returns result → no external deps

Shadow copy recording happens AFTER step 4 returns.

---

## 6. Architecture Diagram

```
[Problem Text]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                  TRM Math Navigator                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Query Rules │───▶│ Compose RPN │───▶│ Execute PTX │  │
│  │ (Grammar)   │    │ (Math Gal.) │    │ (Cranium)   │  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘  │
│                                               │         │
│                                               ▼         │
│                                        [Result + Conf]  │
│                                               │         │
└───────────────────────────────────────────────┼─────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    │ if confidence > 0.8   │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────┐
                              │      DualShadowCopy.record()    │
                              │  - Quality filtering            │
                              │  - Deduplication                │
                              │  - Pattern confidence update    │
                              └─────────────────────────────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────┐
                              │    SleepTime Consolidation      │
                              │  - Prune low-quality            │
                              │  - Promote canonical            │
                              │  - Commit to Grammar Galaxy     │
                              └─────────────────────────────────┘
```

---

## 7. Success Criteria

### 7.1 Functional

- [ ] TRM Navigator records successful solves to shadow copy
- [ ] Pattern confidence tracks success rates
- [ ] Checkpoint persists discoveries across runs
- [ ] SleepTime consolidation prunes/promotes math rules

### 7.2 Metrics

After 100 benchmark problems:
- Shadow copy should have 10+ unique discoveries
- Pattern confidence should track 5+ distinct patterns
- At least 1 high-quality rule should be promoted to canonical

### 7.3 Sovereignty

- [ ] No numpy in TRM Navigator hot path
- [ ] Shadow copy operations happen AFTER inference
- [ ] Checkpoint I/O is separate from solving

---

## 8. What NOT to Do

1. **Don't rewrite DualShadowCopy** - extend it
2. **Don't add numpy to TRM Navigator** - shadow copy is separate
3. **Don't record low-confidence solves** - only confidence > 0.8
4. **Don't block inference on shadow copy** - record asynchronously if needed
5. **Don't duplicate the consolidation logic** - reuse SleepTimeConsolidator

---

## 9. Final Directive

**Codex, your mission:**

1. Add `"math"` program type to `DualShadowCopy._commit_entry()`
2. Wire shadow copy to TRM Navigator via constructor
3. Implement `_record_success()` to record high-confidence solves
4. Wire benchmark runner to create/save shadow copy
5. Add tests for recording, confidence tracking, persistence

**Remember:**
- Existing infrastructure is GOOD - don't reinvent it
- Shadow copy is NOT hot path - numpy/file I/O is OK there
- Hot path remains sovereign (PTX + Galaxy only)
- Patterns learned help future solves (calibration)

**The infrastructure exists. Wire it up.**

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** HIGH - Enables TRM learning loop
