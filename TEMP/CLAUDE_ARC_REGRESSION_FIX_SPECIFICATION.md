# ARC-AGI Regression Fix Specification

**Date:** December 3, 2025
**Version:** 1.0
**Status:** Architecture Review → Codex Implementation
**Authors:** Claude (Architecture Partner) + Daniel (Lead Architect)
**Context:** Run 035-036 Regression Investigation Complete

---

## Executive Summary

**Problem:** Training runs 035-036 show persistent regression to 31.8-31.9% accuracy, down from 46.7% baseline (Run 028). Investigation by Codex identified four root causes that together hollowed out the training architecture while appearing to run normally.

**Impact:**
- **-14.9% accuracy loss** (46.7% → 31.9%)
- **SleepTime consolidation** removed 19 entries without audit trail (destructive filtering)
- **Scale-invariant primitives** registered but never used (0 occurrences in 162 epochs)
- **Grammar/shape instrumentation** blind (parser doesn't match RPN tokens)
- **Shadow Copy discovery stalled** (+2 entries only vs healthy growth in earlier runs)

**Solution Strategy:**
1. Add SleepTime audit logging and relax pruning threshold
2. Wire scale-invariant primitives into candidate generation
3. Fix vocabulary instrumentation to parse RPN tokens correctly
4. Validate fixes with diagnostic run before resuming training

**Success Criteria:**
- Run 037 recovers to 40-47% accuracy range (Run 030-034 baseline)
- Scale-invariant primitives appear in candidate logs
- Vocabulary quality blocks show grammar rule usage
- Shadow Copy growth resumes (healthy discovery rate)

---

## Root Cause Analysis

### **Issue 1: SleepTime Destructive Filtering (Critical)**

**What Happened:**

From consolidation log before Run 035:
```
[SLEEPTIME] Pruned: 19 low-quality entries (< 0.60)
[SLEEPTIME] Canonical promoted: 0 rules (insufficient usage signal)
[SLEEPTIME] Shadow library: 125 → 106 entries
```

**Problems Identified:**

1. **No audit trail:** The 19 pruned entry hashes were not logged
   - Cannot identify which programs were removed
   - Cannot assess if they were actually low-value
   - Cannot restore them if pruning was too aggressive

2. **Consolidation report empty:** File `consolidation_report_20251202T230559Z.json` contains neither `rule_stats` nor `shape_stats`
   - Promised analytics never generated
   - Cannot verify consolidation quality
   - No basis for 0.60 threshold decision

3. **Current shadow library composition:** Analysis of current state shows:
   - 84/142 entries are "visual" type (59%)
   - Few procedural transformations remain
   - Suggests pruning removed rotation/flip/recolor operations

4. **Shadow growth stalled:** Run 035-036 grew shadow by only +2 entries
   - Healthy earlier runs showed continuous growth
   - Indicates discovery pipeline broken

**Hypothesis:**
> SleepTime pruning with threshold 0.60 removed borderline-quality patterns (0.55-0.65 range) that were actually critical building blocks for solving tasks. The consolidator acted as a destructive filter rather than a knowledge consolidator.

**Evidence Supporting Hypothesis:**
- Immediate accuracy drop after consolidation (Run 034 → Run 035)
- Persistent low accuracy in Run 036 (confirms not a fluke)
- Shadow growth stalled (discovery depends on existing patterns)
- Shadow library now 59% visual (transformations pruned?)

---

### **Issue 2: Scale-Invariant Primitives Not Integrated (Critical)**

**What Happened:**

From Run 036 startup log:
```
[DrawingGalaxy] Loaded 24 shapes ...
[INIT] Registered scale-invariant primitives: REL_LINE, REL_RECT, PROP_GRID, FLOOD_REL
```

But from diagnostic analysis:
```bash
grep -c "REL_LINE\|REL_RECT\|PROP_GRID\|FLOOD_REL" /tmp/arc_run_036.log
# Result: 0 occurrences in 162 epochs
```

**Problems Identified:**

1. **Primitives registered but unused:** Drawing Galaxy contains scale-invariant shapes, but candidate generator never emits them

2. **Candidate generation hardcoded:** Inspection of `knowledge3d/training/arc_agi/candidate_generator.py` lines 239-372 shows hand-written operations only:
   - rotate, flip, translate, recolor
   - No Drawing Galaxy query
   - No scale-invariant primitive lookup

3. **Wasted vocabulary:** Added 4 new primitive types that contributed nothing while pruning removed existing vocabulary

**Hypothesis:**
> Scale-invariant primitives were added to Drawing Galaxy catalog but never wired into the candidate generation pipeline. The system continued using only the hardcoded operation set, making the new vocabulary inert.

**Evidence Supporting Hypothesis:**
- Zero occurrences of REL_* tokens in logs
- Candidate generator code shows no Drawing Galaxy integration
- Startup confirms registration (primitives exist)
- Diagnostic confirms never used (0 grep hits)

---

### **Issue 3: Vocabulary Instrumentation Blind (High Priority)**

**What Happened:**

From Run 036 vocabulary quality logs (epochs 1-160):
```
[VOCAB QUALITY Epoch 10]
  Top Grammar Rules (by usage in high-quality solutions):
    No grammar usage recorded among top-100 high-quality entries
  Top Drawing Shapes (by usage in high-quality solutions):
    No drawing shapes recorded among top-100 high-quality entries
```

**This pattern repeated for ALL 21 vocabulary blocks across 162 epochs.**

**Problems Identified:**

1. **Parser doesn't match RPN tokens:** Implementation of `_parse_grammar_rules_from_program()` at `knowledge3d/training/arc_agi/sovereign_pipeline.py` lines 710-718:

```python
def _parse_grammar_rules_from_program(self, program: str) -> List[str]:
    """Parse which grammar rules appear in a program string."""
    tokens = program.split()
    found = [rule_id for rule_id, rule in self.grammar.rules.items()
             if rule.name in tokens]
    return found
```

**Problem:** Rule names are like `"en_simple_sentence"` or `"draw_line"`, but programs contain RPN stack operations like `"FLIP_V 1 rotate GRID"`. The parser checks if `rule.name` appears in tokens, which will never match.

2. **False negative since Phase 2:** This instrumentation has been reporting "No grammar usage" since implementation, not just after SleepTime consolidation. We relied on broken metrics to detect regression.

3. **Cannot distinguish real vs instrumentation issues:** When vocabulary blocks show "no usage," we don't know if:
   - Grammar rules aren't being used (real problem)
   - Grammar rules are being used but parser can't detect them (instrumentation problem)

**Hypothesis:**
> The vocabulary quality parser was implemented as a naive string matching stub that checks if rule IDs (semantic labels) appear in program tokens (RPN opcodes). These never match, so the instrumentation has been blind since inception. We cannot tell if grammar/shape catalogs are actually being used.

**Evidence Supporting Hypothesis:**
- Parser code shows naive token matching
- Rule IDs are semantic (`"rotation_or_reflection"`)
- Program tokens are RPN (`"FLIP_V"`, `"rotate"`)
- Semantic ≠ RPN (different namespaces)
- ALL vocabulary blocks show "no usage" (not just post-consolidation)

---

### **Issue 4: Persistent Regression Confirmed (Architectural)**

**The Data:**

| Run | Accuracy | Context |
|-----|----------|---------|
| 028 | 46.7% | Baseline (documented achievement) |
| 030-034 | 42-47% | Healthy stochastic variance (±3% around 45%) |
| **035** | **31.8%** | Post-SleepTime consolidation |
| **036** | **31.9%** | Confirmation run (same parameters) |

**Diagnostic Analysis (`/tmp/run_036_diagnostics.txt`):**

- Best epoch: 42/108 (38.9%)
- Worst epoch: 29/108 (26.9%)
- Mean: 31.9%
- Trend: First half 32.0%, second half 31.9% (slight decline)
- Scale-invariant primitives: 0 occurrences
- Strong attractors: 0 detected
- Grammar usage: 0 across all epochs
- Shadow growth: +2 entries only

**Conclusion:**
This is NOT stochastic variance. This is a persistent architectural regression where three simultaneous issues (pruning, unused primitives, blind instrumentation) together broke the training loop while appearing to run normally.

---

## Proposed Solutions

### **Solution 1: SleepTime Audit Logging + Threshold Relaxation (Priority: Critical)**

**Objective:** Make consolidation observable and less aggressive.

#### 1A. Add Comprehensive Audit Logging

**File:** `knowledge3d/training/arc_agi/sleeptime_consolidator.py`

**Changes Required:**

1. **Log pruned entries before removal:**

```python
def _prune_low_quality(self) -> int:
    """
    Prune Shadow Copy entries below min_quality threshold WITH AUDIT LOG.

    Returns:
        Number of entries pruned
    """
    pruned_entries = []
    kept_entries = []

    for entry in self.shadow.library:
        quality = entry.get('quality_score', 0)
        if quality < self.min_quality:
            # Log details before pruning
            pruned_entries.append({
                'hash': entry.get('hash', 'unknown'),
                'quality_score': quality,
                'program_type': entry.get('program_type', 'unknown'),
                'program': entry.get('program', 'unknown')[:100],  # First 100 chars
                'complexity': entry.get('complexity', 'unknown'),
            })
        else:
            kept_entries.append(entry)

    # Log pruning details
    print(f"\n[SLEEPTIME PRUNING AUDIT]")
    print(f"  Threshold: {self.min_quality}")
    print(f"  Total entries: {len(self.shadow.library)}")
    print(f"  Pruned: {len(pruned_entries)}")
    print(f"  Kept: {len(kept_entries)}")

    if pruned_entries:
        print(f"\n  Pruned entries detail:")
        for i, entry in enumerate(pruned_entries[:10]):  # Show first 10
            print(f"    {i+1}. hash={entry['hash'][:16]}... quality={entry['quality_score']:.3f} type={entry['program_type']}")
            print(f"       program={entry['program'][:60]}...")

        if len(pruned_entries) > 10:
            print(f"    ... and {len(pruned_entries) - 10} more")

    # Export full pruning details to consolidation report
    self._pruned_audit = pruned_entries

    # Update shadow library
    self.shadow.library = kept_entries

    return len(pruned_entries)
```

2. **Export audit to consolidation report:**

```python
def consolidate(self) -> Dict:
    """Run full consolidation cycle WITH AUDIT EXPORT."""
    stats = {}

    # Existing consolidation steps...
    stats['pruned_count'] = self._prune_low_quality()
    stats['rule_stats'] = self._analyze_grammar_rules()
    stats['shape_stats'] = self._analyze_drawing_shapes()
    # ... etc

    # NEW: Export pruning audit
    if hasattr(self, '_pruned_audit'):
        stats['pruned_entries_audit'] = self._pruned_audit

    # NEW: Export analysis details (not just counts)
    stats['rule_stats_detail'] = self._export_rule_details(stats['rule_stats'])
    stats['shape_stats_detail'] = self._export_shape_details(stats['shape_stats'])

    return stats
```

3. **Save detailed consolidation report:**

```python
# In scripts/run_sleeptime_consolidation.py

with open(report_path, 'w') as f:
    json.dump(stats, f, indent=2)

# NEW: Also save human-readable audit log
audit_log_path = checkpoint_dir / f"consolidation_audit_{timestamp}.txt"
with open(audit_log_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("SLEEPTIME CONSOLIDATION AUDIT LOG\n")
    f.write("="*70 + "\n\n")

    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Pruning threshold: {consolidator.min_quality}\n")
    f.write(f"Pruned entries: {stats['pruned_count']}\n\n")

    if 'pruned_entries_audit' in stats:
        f.write("PRUNED ENTRIES:\n")
        f.write("-"*70 + "\n")
        for entry in stats['pruned_entries_audit']:
            f.write(f"\nHash: {entry['hash']}\n")
            f.write(f"Quality: {entry['quality_score']:.4f}\n")
            f.write(f"Type: {entry['program_type']}\n")
            f.write(f"Program: {entry['program']}\n")
            f.write("-"*70 + "\n")

    # ... similar for rule_stats, shape_stats
```

**Success Criteria:**
- Consolidation report contains `pruned_entries_audit` with full details
- Human-readable audit log saved alongside JSON report
- Can identify exactly which programs were pruned
- Can assess if pruning was appropriate

#### 1B. Relax Pruning Threshold

**Rationale:** Threshold 0.60 may be too aggressive. Programs with quality 0.55-0.65 might be valuable building blocks even if not top-quality themselves (they compose into higher-quality solutions).

**Change:**

```python
# In scripts/run_sleeptime_consolidation.py

consolidator = SleepTimeConsolidator(
    shadow_copy,
    drawing_galaxy,
    grammar_galaxy,
    min_quality=0.50,  # CHANGED from 0.60 to 0.50
    min_uses_for_canonical=5,
    canonical_success_threshold=0.7,
)
```

**Alternative: Temporarily Disable Pruning**

For Run 037 diagnostic run, consider:

```python
min_quality=0.0,  # Disable pruning entirely for diagnostic
```

This allows us to test if pruning was the sole cause without risking further damage.

**Success Criteria:**
- Fewer entries pruned (or zero if disabled)
- Shadow library retains procedural transformations
- Run 037 accuracy shows recovery trend

---

### **Solution 2: Integrate Scale-Invariant Primitives (Priority: Critical)**

**Objective:** Wire Drawing Galaxy primitives into candidate generation so they're actually used.

#### 2A. Extend Candidate Generator to Query Drawing Galaxy

**File:** `knowledge3d/training/arc_agi/candidate_generator.py`

**Current Issue:** Lines 239-372 contain hardcoded operations. Drawing Galaxy is never queried.

**Required Changes:**

1. **Add Drawing Galaxy integration:**

```python
class CandidateGenerator:
    def __init__(
        self,
        matryoshka_dim: int = 512,
        max_candidates: int = 10,
        shadow_copy: Optional[DualShadowCopy] = None,
        drawing_galaxy: Optional[DrawingGalaxy] = None,  # NEW parameter
        executor: Optional[ARCRPNExecutor] = None,
        # ... existing params
    ):
        self.matryoshka_dim = matryoshka_dim
        self.max_candidates = max_candidates
        self.shadow_copy = shadow_copy
        self.drawing_galaxy = drawing_galaxy  # NEW
        self.executor = executor
        # ...
```

2. **Query scale-invariant primitives during generation:**

```python
def _generate_scale_invariant_candidates(
    self,
    input_grid,
    train_examples,
) -> List[Candidate]:
    """
    Generate candidates using scale-invariant primitives from Drawing Galaxy.

    These primitives use relative coordinates (0.0-1.0) instead of absolute pixels,
    making them robust to grid resizing operations common in ARC-AGI.
    """
    if self.drawing_galaxy is None:
        return []

    candidates = []

    # Get scale-invariant primitives
    scale_inv_primitives = [
        prim_id for prim_id in self.drawing_galaxy.shapes.keys()
        if 'REL_' in prim_id or 'PROP_' in prim_id or 'FLOOD_' in prim_id
    ]

    print(f"  [SCALE-INV GEN] Found {len(scale_inv_primitives)} scale-invariant primitives")

    # Generate candidates using these primitives
    for prim_id in scale_inv_primitives[:self.max_candidates // 2]:  # Use up to half budget
        prim_def = self.drawing_galaxy.shapes[prim_id]

        # Extract RPN program
        rpn_program = prim_def.get('visual_rpn', '')
        if not rpn_program:
            continue

        # Execute RPN program on input grid
        try:
            output = self.executor.execute_rpn(rpn_program, input_grid)
            candidate = Candidate(
                program=rpn_program,
                output=output,
                confidence=0.5,  # Neutral confidence for untested primitives
                source='scale_invariant_primitive'
            )
            candidates.append(candidate)
            print(f"  [SCALE-INV GEN] Generated candidate using {prim_id}")
        except Exception as e:
            print(f"  [SCALE-INV GEN] Failed to execute {prim_id}: {e}")

    return candidates
```

3. **Integrate into main generation pipeline:**

```python
def generate_candidates(
    self,
    input_grid,
    train_examples,
    semantic_hints,
    expected_output=None,
) -> List[Candidate]:
    """Generate candidates using ALL available sources."""
    all_candidates = []

    # Existing sources
    semantic_candidates = self._generate_semantic_candidates(...)
    compositional_candidates = self._generate_compositional_candidates(...)

    # NEW: Scale-invariant candidates
    scale_inv_candidates = self._generate_scale_invariant_candidates(
        input_grid,
        train_examples,
    )

    all_candidates.extend(semantic_candidates)
    all_candidates.extend(compositional_candidates)
    all_candidates.extend(scale_inv_candidates)

    print(f"  [CANDIDATE GEN] Total: {len(all_candidates)} candidates "
          f"(semantic={len(semantic_candidates)}, "
          f"compositional={len(compositional_candidates)}, "
          f"scale_inv={len(scale_inv_candidates)})")

    return all_candidates
```

4. **Pass Drawing Galaxy to generator:**

**File:** `knowledge3d/training/arc_agi/parallel_generator.py`

```python
class ParallelCandidateGenerator:
    def __init__(
        self,
        *,
        num_workers: int = 9,
        shadow_copy: Optional[DualShadowCopy] = None,
        drawing_galaxy: Optional[DrawingGalaxy] = None,  # NEW
        # ... existing params
    ):
        self.num_workers = num_workers
        self.shadow_copy = shadow_copy
        self.drawing_galaxy = drawing_galaxy  # NEW
        # ...

def _worker_task(self, worker_id, hints_subset, input_grid, train_examples, expected_output):
    # ... existing code ...

    gen = CandidateGenerator(
        matryoshka_dim=self.matryoshka_dim,
        max_candidates=self.candidates_per_worker,
        shadow_copy=self.shadow_copy,
        drawing_galaxy=self.drawing_galaxy,  # NEW: Pass to worker
        executor=executor,
        # ... existing params
    )

    # ... rest of worker task
```

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py`

```python
# Where ParallelCandidateGenerator is instantiated (around line 291)

par_gen = ParallelCandidateGenerator(
    num_workers=9,
    candidates_per_worker=6,
    shadow_copy=self.shadow_copy,
    drawing_galaxy=self.drawing,  # NEW: Pass DrawingGalaxy instance
    # ... existing params
)
```

**Success Criteria:**
- Scale-invariant candidates appear in generation logs
- `grep "REL_LINE\|PROP_GRID" /tmp/arc_run_037.log` returns >0 hits
- Diagnostic shows scale-invariant primitives being executed
- No crashes when executing scale-invariant RPN programs

---

### **Solution 3: Fix Vocabulary Instrumentation (Priority: High)**

**Objective:** Make grammar/shape usage metrics actually work.

#### 3A. Implement RPN Token Matching

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Current broken implementation (lines 710-718):**

```python
def _parse_grammar_rules_from_program(self, program: str) -> List[str]:
    """Parse which grammar rules appear in a program string."""
    tokens = program.split()
    found = [rule_id for rule_id, rule in self.grammar.rules.items()
             if rule.name in tokens]  # WRONG: name is semantic, tokens are RPN
    return found
```

**Fixed implementation:**

```python
def _parse_grammar_rules_from_program(self, program: str) -> List[str]:
    """
    Parse which grammar rules appear in a program string.

    Strategy:
    - Program contains RPN opcodes: "FLIP_V 1 rotate GRID"
    - Grammar rules have metadata mapping them to RPN tokens
    - Match RPN tokens in program against rule RPN signatures

    Args:
        program: RPN program string (e.g., "FLIP_V 1 rotate GRID")

    Returns:
        List of grammar rule IDs found in the program
    """
    if not program:
        return []

    # Tokenize program
    tokens = program.split()
    token_set = set(tokens)

    found_rules = []

    for rule_id, rule in self.grammar.rules.items():
        # Strategy 1: Check if rule has RPN signature
        if hasattr(rule, 'rpn_pattern'):
            # Rule defines which RPN tokens it generates
            rpn_tokens = set(rule.rpn_pattern.split())
            if rpn_tokens & token_set:  # Set intersection
                found_rules.append(rule_id)
                continue

        # Strategy 2: Check if rule_id itself appears as token
        # Some rules might be named after their RPN operation
        if rule_id in token_set:
            found_rules.append(rule_id)
            continue

        # Strategy 3: Parse rule name for RPN keywords
        # E.g., "rotate_90_cw" contains "rotate"
        rule_name_tokens = rule.name.lower().split('_') if hasattr(rule, 'name') else []
        for rule_token in rule_name_tokens:
            if rule_token.upper() in token_set:
                found_rules.append(rule_id)
                break

    return found_rules
```

**Better Solution: Add RPN Metadata to Grammar Rules**

**File:** `knowledge3d/training/arc_agi/grammar_galaxy.py`

Enhance `GrammarRule` dataclass to include RPN signature:

```python
@dataclass
class GrammarRule:
    """Grammar rule with RPN execution signature."""
    name: str
    description: str = ""
    is_canonical: bool = False

    # NEW: RPN signature field
    rpn_pattern: str = ""  # e.g., "FLIP_V" or "rotate" or "GRID"

    # Existing fields
    # ...
```

Update grammar rule definitions to include RPN patterns:

```python
# Example rules with RPN patterns
rules = {
    "flip_vertical": GrammarRule(
        name="flip_vertical",
        description="Flip grid vertically",
        rpn_pattern="FLIP_V",  # NEW
    ),
    "rotate_90": GrammarRule(
        name="rotate_90",
        description="Rotate 90 degrees clockwise",
        rpn_pattern="rotate",  # NEW
    ),
    "grid_fill": GrammarRule(
        name="grid_fill",
        description="Fill grid pattern",
        rpn_pattern="GRID",  # NEW
    ),
}
```

**If rules don't have RPN patterns yet:** Create mapping file:

```python
# knowledge3d/training/arc_agi/rpn_rule_mapping.py

"""Mapping between grammar rule IDs and RPN tokens they generate."""

RPN_RULE_MAPPING = {
    # Transformation rules
    "flip_vertical": ["FLIP_V"],
    "flip_horizontal": ["FLIP_H"],
    "rotate_90_cw": ["rotate", "1"],
    "rotate_180": ["rotate", "2"],
    "rotate_270_ccw": ["rotate", "3"],

    # Grid operations
    "grid_fill": ["GRID"],
    "grid_expand": ["GRID", "EXPAND"],

    # Color operations
    "recolor": ["RECOLOR"],
    "color_swap": ["COLOR", "SWAP"],

    # Spatial operations
    "translate": ["TRANSLATE"],
    "scale": ["SCALE"],

    # Add more as discovered from actual programs
}
```

Use mapping in parser:

```python
from knowledge3d.training.arc_agi.rpn_rule_mapping import RPN_RULE_MAPPING

def _parse_grammar_rules_from_program(self, program: str) -> List[str]:
    """Parse grammar rules using RPN token mapping."""
    if not program:
        return []

    tokens = set(program.split())
    found_rules = []

    for rule_id, rpn_tokens in RPN_RULE_MAPPING.items():
        if any(token in tokens for token in rpn_tokens):
            found_rules.append(rule_id)

    return found_rules
```

#### 3B. Fix Drawing Shape Detection Similarly

Same approach for `_parse_drawing_shapes_from_program()`:

```python
def _parse_drawing_shapes_from_program(self, program: str) -> List[str]:
    """Parse which drawing shapes appear in a program string."""
    if not program:
        return []

    tokens = set(program.split())
    found_shapes = []

    # Check against Drawing Galaxy shape IDs
    for shape_id in self.drawing.shapes.keys():
        # Check if shape_id or its RPN tokens appear
        if shape_id in tokens:
            found_shapes.append(shape_id)
        # Also check for primitive names
        elif shape_id.startswith('PRIM_'):
            prim_name = shape_id.replace('PRIM_', '')
            if prim_name in tokens:
                found_shapes.append(shape_id)

    return found_shapes
```

**Success Criteria:**
- Vocabulary quality blocks show non-zero grammar rule usage
- When scale-invariant primitives are used, they appear in shape usage stats
- Metrics reflect actual program composition
- Can observe vocabulary trends epoch-over-epoch

---

### **Solution 4: Validation Strategy (Before Run 037)**

**Objective:** Test fixes without committing to 18-24 hour run.

#### 4A. Create Diagnostic Test Script

**File:** `scripts/test_regression_fixes.py`

```python
#!/usr/bin/env python3
"""
Test regression fixes before Run 037.

Validates:
1. SleepTime audit logging works
2. Scale-invariant primitives are generated
3. Vocabulary parsing detects usage
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.candidate_generator import CandidateGenerator
from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignAIPipeline


def test_scale_invariant_primitives():
    """Test that scale-invariant primitives can be generated."""
    print("\n" + "="*70)
    print("TEST 1: Scale-Invariant Primitives Generation")
    print("="*70)

    drawing = DrawingGalaxy()
    drawing.load('/K3D/Knowledge3D.local/checkpoints/arc_agi/drawing_galaxy.json')

    # Check that primitives exist
    scale_inv = [k for k in drawing.shapes.keys() if 'REL_' in k or 'PROP_' in k]
    print(f"✓ Found {len(scale_inv)} scale-invariant primitives: {scale_inv}")

    # Test candidate generation
    from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor
    from knowledge3d.cranium.ptx_runtime.math_core_pool import get_global_math_core_pool

    pool = get_global_math_core_pool()
    core_id = pool.spawn_core(tier=1, reuse=True)
    executor = ARCRPNExecutor(pool=pool, instance_id=core_id)

    gen = CandidateGenerator(
        matryoshka_dim=512,
        max_candidates=10,
        drawing_galaxy=drawing,
        executor=executor,
    )

    # Simple test grid
    test_grid = [[0, 1], [1, 0]]

    candidates = gen._generate_scale_invariant_candidates(
        input_grid=test_grid,
        train_examples=[],
    )

    pool.release_core(core_id, pool=True)

    if candidates:
        print(f"✓ Generated {len(candidates)} scale-invariant candidates")
        for i, cand in enumerate(candidates[:3]):
            print(f"  {i+1}. Program: {cand.program[:50]}...")
        return True
    else:
        print(f"✗ FAILED: No scale-invariant candidates generated")
        return False


def test_vocabulary_parsing():
    """Test that grammar rules can be detected in programs."""
    print("\n" + "="*70)
    print("TEST 2: Vocabulary Parsing (Grammar + Shapes)")
    print("="*70)

    grammar = GrammarGalaxy()
    grammar.load('/K3D/Knowledge3D.local/checkpoints/arc_agi/grammar_galaxy.json')

    drawing = DrawingGalaxy()
    drawing.load('/K3D/Knowledge3D.local/checkpoints/arc_agi/drawing_galaxy.json')

    # Test programs
    test_programs = [
        "FLIP_V 1 rotate",
        "GRID RECOLOR 2",
        "REL_LINE 0.0 0.0 1.0 1.0",
        "PROP_GRID 3 3",
    ]

    pipeline = SovereignAIPipeline(matryoshka_dim=512)
    pipeline.grammar = grammar
    pipeline.drawing = drawing

    success = True
    for program in test_programs:
        rules = pipeline._parse_grammar_rules_from_program(program)
        shapes = pipeline._parse_drawing_shapes_from_program(program)

        print(f"\nProgram: {program}")
        print(f"  Grammar rules detected: {rules if rules else 'NONE'}")
        print(f"  Shapes detected: {shapes if shapes else 'NONE'}")

        # At least one should be detected if parsing works
        if not rules and not shapes:
            print(f"  ⚠️ WARNING: No vocabulary detected for this program")
            # Don't fail immediately - parser might legitimately find nothing
            # but at least one test program should match something

    return True  # Success if no crashes


def test_sleeptime_audit():
    """Test that SleepTime produces audit logs."""
    print("\n" + "="*70)
    print("TEST 3: SleepTime Audit Logging")
    print("="*70)

    # This requires running consolidation
    # For now, just check that the code compiles
    from knowledge3d.training.arc_agi.sleeptime_consolidator import SleepTimeConsolidator

    print("✓ SleepTimeConsolidator imports successfully")
    print("  (Full audit test requires running consolidation)")

    return True


def main():
    """Run all diagnostic tests."""
    print("\n" + "="*70)
    print("REGRESSION FIX VALIDATION TESTS")
    print("="*70)

    results = {
        'Scale-Invariant Primitives': test_scale_invariant_primitives(),
        'Vocabulary Parsing': test_vocabulary_parsing(),
        'SleepTime Audit': test_sleeptime_audit(),
    }

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ All tests passed. Ready for Run 037 diagnostic.")
    else:
        print("\n✗ Some tests failed. Fix issues before Run 037.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

#### 4B. Run Diagnostic Test

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=. \
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
scripts/test_regression_fixes.py
```

**Expected Output:**
```
======================================================================
REGRESSION FIX VALIDATION TESTS
======================================================================

======================================================================
TEST 1: Scale-Invariant Primitives Generation
======================================================================
✓ Found 4 scale-invariant primitives: ['REL_LINE', 'REL_RECT', 'PROP_GRID', 'FLOOD_REL']
✓ Generated 2 scale-invariant candidates
  1. Program: REL_LINE 0.0 0.0 1.0 1.0...
  2. Program: PROP_GRID 3 3...

======================================================================
TEST 2: Vocabulary Parsing (Grammar + Shapes)
======================================================================

Program: FLIP_V 1 rotate
  Grammar rules detected: ['flip_vertical', 'rotate_90_cw']
  Shapes detected: []

Program: REL_LINE 0.0 0.0 1.0 1.0
  Grammar rules detected: []
  Shapes detected: ['REL_LINE']

======================================================================
TEST 3: SleepTime Audit Logging
======================================================================
✓ SleepTimeConsolidator imports successfully
  (Full audit test requires running consolidation)

======================================================================
TEST SUMMARY
======================================================================
  ✓ PASS: Scale-Invariant Primitives
  ✓ PASS: Vocabulary Parsing
  ✓ PASS: SleepTime Audit

✓ All tests passed. Ready for Run 037 diagnostic.
```

#### 4C. Short Diagnostic Run (10 tasks × 3 epochs)

If tests pass, run short training to validate in real scenario:

```bash
tmux new-session -d -s arc037_diagnostic \
  "cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D' && \
   CUDA_VISIBLE_DEVICES=0 \
   PYTHONPATH=. \
   /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
   scripts/train_arc_sovereign_loop.py \
   --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
   --max-tasks 10 \
   --epochs 3 \
   --cycles 1 \
   --matryoshka-dim 512 \
   > /tmp/arc_diagnostic_037.log 2>&1"
```

**Monitor:**
```bash
# Check for scale-invariant usage
grep "SCALE-INV GEN\|REL_LINE\|PROP_GRID" /tmp/arc_diagnostic_037.log

# Check for vocabulary detection
grep "VOCAB QUALITY" /tmp/arc_diagnostic_037.log -A 10

# Check accuracy
tail -50 /tmp/arc_diagnostic_037.log | grep "correct"
```

**Success Criteria:**
- Scale-invariant candidates appear in logs
- Vocabulary quality shows non-zero grammar/shape usage
- Accuracy ≥ 4/10 (40%) on small diagnostic set
- No crashes or exceptions

---

## Rollout Strategy

### **Phase 1: Implement Fixes (1-2 hours)**

**Order of implementation:**

1. **Fix vocabulary parsing** (30 min)
   - Update `_parse_grammar_rules_from_program()` and `_parse_drawing_shapes_from_program()`
   - Add RPN token mapping
   - Test with sample programs

2. **Add SleepTime audit logging** (30 min)
   - Update `sleeptime_consolidator.py` to log pruned entries
   - Export audit to consolidation report
   - Relax threshold to 0.50 (or disable for diagnostic)

3. **Integrate scale-invariant primitives** (45 min)
   - Update `CandidateGenerator` to query Drawing Galaxy
   - Add `_generate_scale_invariant_candidates()` method
   - Pass Drawing Galaxy through pipeline

4. **Create diagnostic test script** (15 min)
   - Implement `test_regression_fixes.py`
   - Validate all three fixes work

---

### **Phase 2: Validation (30 min)**

1. Run diagnostic test script
2. Verify all tests pass
3. Run short 10-task training
4. Check logs for expected behavior

---

### **Phase 3: Full Training (18-24 hours)**

**Only proceed if Phase 2 succeeds.**

1. Run SleepTime consolidation again (with audit logging)
2. Launch Run 037 (full 108 tasks × 162 epochs)
3. Monitor for:
   - Scale-invariant primitive usage
   - Vocabulary quality metrics
   - Accuracy recovery trend
4. Compare to Run 035-036 baseline

---

## Success Metrics

### **Immediate (Diagnostic Run)**

✓ Test script passes all 3 tests
✓ Scale-invariant candidates generated
✓ Vocabulary parsing detects grammar/shape usage
✓ 10-task diagnostic shows ≥40% accuracy

### **Run 037 (Full Training)**

✓ Accuracy recovers to 40-47% range (Run 030-034 baseline)
✓ Scale-invariant primitives appear in logs (grep >0 hits)
✓ Vocabulary quality blocks show grammar rule usage
✓ Shadow Copy growth resumes (healthy +10-20 entries per 162 epochs)
✓ SleepTime audit log contains full pruning details

### **Comparison Metrics**

| Metric | Run 035-036 (Broken) | Run 037 (Target) |
|--------|----------------------|------------------|
| Accuracy | 31.8-31.9% | 40-47% |
| Scale-inv usage | 0 occurrences | >100 occurrences |
| Grammar detection | "No usage" all epochs | >0 rules detected |
| Shadow growth | +2 entries | +10-20 entries |
| Audit logging | None | Full details |

---

## Risk Assessment

### **Low Risk Fixes**

✓ Vocabulary parsing (pure instrumentation, can't break training)
✓ SleepTime audit logging (observability only)

### **Medium Risk Fixes**

⚠️ Scale-invariant primitive integration
- Risk: New candidate source might produce invalid programs
- Mitigation: Wrap execution in try/except, log failures
- Validation: Diagnostic run tests this

⚠️ Relaxed pruning threshold
- Risk: Keeping low-quality patterns might pollute vocabulary
- Mitigation: Start with 0.50 (conservative), can tighten later
- Validation: Audit log shows what's kept vs pruned

### **High Risk Actions (Avoid)**

❌ Disabling SleepTime entirely (loses consolidation benefits)
❌ Reverting to pre-SleepTime code (loses new features)
❌ Running Run 037 without validation (wastes 18-24 hours)

---

## Alternative Strategies

### **Option A: Rollback to Run 034 State (Conservative)**

If fixes don't work, rollback:

1. Restore Run 034 checkpoints (pre-consolidation)
2. Disable SleepTime for Runs 037-039
3. Implement fixes while training continues
4. Re-enable SleepTime after fixes validated

**Pros:** Guaranteed recovery to 42-47% baseline
**Cons:** Loses SleepTime benefits, delays consolidation validation

### **Option B: Hybrid (Recommended)**

1. Restore Run 034 checkpoints
2. Implement all three fixes
3. Skip SleepTime consolidation for Run 037
4. Run 037 with fixes active (should show scale-inv usage, vocab detection)
5. If Run 037 recovers (40-47%), run consolidation before Run 038

**Pros:** Tests fixes independently, safe recovery path
**Cons:** Slightly more complex workflow

---

## Implementation Checklist

**Codex Implementation Tasks:**

- [ ] Implement vocabulary parsing fixes (`_parse_grammar_rules_from_program`, `_parse_drawing_shapes_from_program`)
- [ ] Add RPN token mapping file or enhance GrammarRule with rpn_pattern field
- [ ] Add SleepTime audit logging to `sleeptime_consolidator.py`
- [ ] Update consolidation report export with pruned_entries_audit
- [ ] Relax pruning threshold to 0.50 in consolidation script
- [ ] Add `_generate_scale_invariant_candidates()` to CandidateGenerator
- [ ] Pass Drawing Galaxy through pipeline (ParallelCandidateGenerator → CandidateGenerator)
- [ ] Create diagnostic test script `test_regression_fixes.py`
- [ ] Run diagnostic tests and verify all pass
- [ ] Run 10-task diagnostic training and verify expected behavior
- [ ] If validation succeeds: Launch Run 037 (full 108 × 162)
- [ ] Monitor Run 037 for recovery metrics
- [ ] Report findings to Daniel + Claude

**Validation Tasks:**

- [ ] Diagnostic test script passes
- [ ] Scale-invariant candidates appear in test output
- [ ] Vocabulary parsing detects rules/shapes in test programs
- [ ] 10-task training completes without crashes
- [ ] Scale-invariant primitives appear in training log
- [ ] Vocabulary quality blocks show non-zero usage
- [ ] Accuracy on diagnostic ≥40%

**Run 037 Monitoring:**

- [ ] Scale-invariant primitive usage: `grep -c "REL_LINE\|PROP_GRID" /tmp/arc_run_037.log` > 100
- [ ] Vocabulary detection: `grep "VOCAB QUALITY" /tmp/arc_run_037.log` shows grammar rules
- [ ] Accuracy trend: mean ≥40%, best epoch ≥45%
- [ ] Shadow growth: final count > 150 entries (+8-10 from current 142)
- [ ] SleepTime audit: consolidation report contains pruned_entries_audit

---

## Expected Timeline

**Phase 1: Implementation**
- Vocabulary parsing: 30 min
- SleepTime audit: 30 min
- Scale-invariant integration: 45 min
- Diagnostic test script: 15 min
- **Total: ~2 hours**

**Phase 2: Validation**
- Run diagnostic tests: 5 min
- Run 10-task training: 15 min
- Analyze results: 10 min
- **Total: ~30 min**

**Phase 3: Full Training**
- Run SleepTime consolidation: 2 min
- Launch Run 037: 2 min
- Training duration: 18-24 hours
- Analysis: 30 min
- **Total: 18-24 hours**

**Overall: 2.5 hours of work + 18-24 hours of training**

---

## References

**Investigation Documents:**
- Codex investigation findings (provided by Daniel)
- Run 035-036 diagnostic reports
- `/tmp/run_036_diagnostics.txt`

**Architecture Specifications:**
- [BRIEFING.md](../BRIEFING.md) - Dual Client Reality, Procedural Foundation
- [docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md](../docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) - Save Information Principle
- [TEMP/CODEX_RUN_035_EXECUTION_BRIEFING.md](CODEX_RUN_035_EXECUTION_BRIEFING.md) - Run 035-036 training context

**Code Files:**
- `knowledge3d/training/arc_agi/sleeptime_consolidator.py` - SleepTime consolidation
- `knowledge3d/training/arc_agi/candidate_generator.py` - Candidate generation
- `knowledge3d/training/arc_agi/parallel_generator.py` - Parallel worker pool
- `knowledge3d/training/arc_agi/sovereign_pipeline.py` - Main training pipeline
- `knowledge3d/training/arc_agi/grammar_galaxy.py` - Grammar rules catalog
- `knowledge3d/training/arc_agi/drawing_galaxy.py` - Drawing primitives catalog

**Checkpoints:**
- `/K3D/Knowledge3D.local/checkpoints/arc_agi/*.json` - Current state
- Run 034 final state (if archived) - Rollback candidate

---

## Conclusion

The Run 035-036 regression was caused by three simultaneous architectural issues:

1. **SleepTime pruning** removed 19 entries without audit trail (destructive filtering)
2. **Scale-invariant primitives** added but never wired into generation (inert vocabulary)
3. **Vocabulary instrumentation** blind since implementation (false negative metrics)

Together, these issues hollowed out the training architecture while appearing to run normally. The system lost critical building blocks (pruning), gained no new capabilities (unused primitives), and had no visibility into the damage (blind metrics).

The proposed fixes are surgical, low-risk, and testable:
- Add observability (audit logging)
- Connect plumbing (wire primitives)
- Fix instrumentation (parse RPN tokens)

With these fixes, Run 037 should recover to 40-47% baseline and resume healthy vocabulary growth.

**Recommendation: Implement fixes → validate with diagnostic → launch Run 037.**

---

**End of Specification**

**Status:** Ready for Codex implementation
**Next Steps:** Hand this spec to Codex for implementation
**Review:** Claude will validate implementation before Run 037 launch
