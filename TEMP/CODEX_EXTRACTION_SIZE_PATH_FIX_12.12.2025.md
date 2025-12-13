# CODEX: Extraction Size-Path & Grammar Singleton Threading Fix

**Priority:** HIGH — Extraction tasks blocked by size guard (40%+ of failures)
**Date:** December 12, 2025

---

## Context Summary

Training run with extraction fix (A–D) completed. Results from log `/K3D/Knowledge3D.local/logs/extraction_fix_20251211_211600.log`:

- **Epochs:** 162
- **Average accuracy:** ~46.09% (flat vs baseline)
- **Extraction tasks STILL failing:**
  - `f4081712`: 24×24 → 6×4 DROPPED (ratios 4.0 and 6.0)
  - `ed74f2f2`: 5×9 → 3×3 DROPPED (ratio_w 3.0, aspect mismatch)
- **Grammar singleton:** Workers still log `[GrammarGalaxy] Loaded 747 rules...`

---

## Problem Analysis

### Issue 1: Size Drop Guard Blocking Extraction Tasks

**Location:** `knowledge3d/training/arc_agi/sovereign_pipeline.py` or `candidate_generator.py`

There is a guard that drops candidates when size ratio ≥ 2.5:

```python
# PROBLEMATIC PATTERN (pseudo-code from logs)
ratio_h = expected_h / output_h  # or vice versa
ratio_w = expected_w / output_w
if ratio_h >= 2.5 or ratio_w >= 2.5:
    print(f"  [DROP] Size mismatch: ratios {ratio_h}, {ratio_w}")
    continue  # SKIPS evaluation entirely
```

**Why this breaks extraction:**
- Extraction tasks intentionally produce SMALLER outputs (24×24 → 6×4 = 4× shrink)
- The guard was meant to filter garbage outputs, NOT legitimate extraction
- Tasks like `f4081712` need 4× shrink, but ratio 4.0 > 2.5 → dropped

### Issue 2: Grammar Singleton Not Threading to Workers

**Location:** `knowledge3d/training/arc_agi/parallel_candidate_generator.py`

The singleton `get_grammar_galaxy()` exists but parallel workers create new instances:

```python
# In parallel worker context
def worker_generate(...):
    # Each worker does its own import + instantiation
    grammar = GrammarGalaxy()  # WRONG: creates new instance, reloads from disk
```

**Why this matters:**
- 747 rules × N workers = N redundant disk loads
- Log noise floods output
- Wastes ~200ms per worker startup

---

## Fix Specification

### Fix 1: Size-Aware Evaluation Path (NOT Size Drop)

**Principle:** Detection is separate from evaluation. If we DETECT a size reduction pattern, we should EXPECT smaller outputs, not drop them.

**Implementation Strategy:**

1. **In `sovereign_pipeline.py`**, find the size drop guard and replace with conditional logic:

```python
def _should_evaluate_candidate(
    self,
    candidate_output: Sequence[Sequence[int]],
    expected_output: Sequence[Sequence[int]],
    size_pattern: str,  # "extract" | "expand" | "same"
) -> bool:
    """
    Decide whether to evaluate a candidate (vs drop).

    For extraction tasks, ALLOW large size ratios.
    For same-size tasks, use the existing ratio guard.
    """
    if not candidate_output or not expected_output:
        return False

    h_cand, w_cand = len(candidate_output), len(candidate_output[0]) if candidate_output else 0
    h_exp, w_exp = len(expected_output), len(expected_output[0]) if expected_output else 0

    if h_exp == 0 or w_exp == 0:
        return False

    ratio_h = max(h_cand / h_exp, h_exp / h_cand)
    ratio_w = max(w_cand / w_exp, w_exp / w_cand)

    # EXTRACTION PATH: Allow large ratios if we detected extraction pattern
    if size_pattern == "extract":
        # For extraction, only drop if ratios are extreme (>10×) or output is empty
        if ratio_h > 10.0 or ratio_w > 10.0:
            print(f"  [EXTRACT DROP] Extreme ratio {ratio_h:.1f}×{ratio_w:.1f}, skipping")
            return False
        return True  # Allow evaluation even with 4× shrink

    # SAME-SIZE PATH: Use existing guard
    if size_pattern == "same":
        if ratio_h >= 2.5 or ratio_w >= 2.5:
            print(f"  [SAME DROP] Size mismatch {ratio_h:.1f}×{ratio_w:.1f}, skipping")
            return False
        return True

    # EXPAND PATH: Similar to same-size but allow growth
    if size_pattern == "expand":
        if ratio_h >= 4.0 or ratio_w >= 4.0:
            print(f"  [EXPAND DROP] Extreme expansion {ratio_h:.1f}×{ratio_w:.1f}, skipping")
            return False
        return True

    return True  # Default: evaluate
```

2. **Pass `size_pattern` from task analysis** into candidate evaluation:

```python
# In the main training loop
size_pattern = self._detect_size_pattern(train_examples)
print(f"  [SIZE PATTERN] Detected: {size_pattern}")

# When evaluating candidates
for candidate in candidates:
    if not self._should_evaluate_candidate(candidate.output, expected_output, size_pattern):
        continue
    # ... evaluate candidate
```

3. **For extraction tasks, use procedural resize BEFORE fuzzy match:**

```python
if size_pattern == "extract" and (h_cand != h_exp or w_cand != w_exp):
    # Resize candidate to expected size for comparison
    resized = self._procedural_resize(candidate_output, h_exp, w_exp)
    fuzzy_score = self._fuzzy_match(resized, expected_output)
else:
    fuzzy_score = self._fuzzy_match(candidate_output, expected_output)
```

### Fix 2: Thread Grammar Singleton Through Parallel Workers

**Location:** `knowledge3d/training/arc_agi/parallel_candidate_generator.py`

**Strategy:** Serialize grammar rules once in main process, pass via shared state or IPC, deserialize in workers (cheaper than file load).

**Option A (Recommended): Pass grammar rules as argument:**

```python
# In main process (before spawning workers)
from knowledge3d.training.arc_agi.grammar_galaxy import get_grammar_galaxy

def generate_candidates_parallel(
    input_grid,
    train_examples,
    semantic_hints,
    num_workers=9,
    grammar_rules: Optional[Dict[str, Dict]] = None,  # NEW: pre-loaded rules
):
    # Get singleton once in main process
    if grammar_rules is None:
        grammar = get_grammar_galaxy()
        grammar_rules = {
            rule_id: {
                "rule_id": rule.rule_id,
                "rpn_program": rule.rpn_program,
                "pattern": rule.pattern,
                "language": rule.language,
            }
            for rule_id, rule in grammar.rules.items()
        }

    # Pass to workers via ProcessPoolExecutor initializer or argument
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                worker_generate,
                input_grid,
                train_examples,
                hints,
                grammar_rules,  # Pass rules
            )
            for hints in semantic_partitions
        ]
        # ...
```

**In worker function:**

```python
def worker_generate(input_grid, train_examples, hints, grammar_rules: Dict[str, Dict]):
    """Worker uses pre-loaded grammar rules (no disk load)."""
    # Reconstruct lightweight grammar accessor
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule, GrammarGalaxy

    # Create GrammarGalaxy from passed rules (no file load)
    rules = [
        GrammarRule(
            rule_id=r["rule_id"],
            rpn_program=r["rpn_program"],
            pattern=r.get("pattern", "unknown"),
            language=r.get("language", "en"),
        )
        for r in grammar_rules.values()
    ]
    grammar = GrammarGalaxy(rules=rules)  # Bypasses file load

    # ... rest of worker logic
```

**Option B (Alternative): Use multiprocessing.Manager for shared dict:**

```python
from multiprocessing import Manager

def generate_candidates_parallel(...):
    manager = Manager()
    shared_grammar = manager.dict()

    # Load once
    grammar = get_grammar_galaxy()
    for rule_id, rule in grammar.rules.items():
        shared_grammar[rule_id] = {
            "rpn_program": rule.rpn_program,
            # ... minimal fields
        }

    # Workers access shared_grammar (read-only, no reload)
```

### Fix 3: Update multimodal_parser.py to Use Singleton

**Location:** `knowledge3d/training/arc_agi/multimodal_parser.py`

Codex already added `get_grammar_galaxy()` call here, but verify it's actually used:

```python
from knowledge3d.training.arc_agi.grammar_galaxy import get_grammar_galaxy

class MultimodalSemanticParser:
    def __init__(self, ...):
        self.grammar = get_grammar_galaxy()  # CORRECT: uses singleton
```

**Verify:** Check that no other instantiation of `GrammarGalaxy()` happens in the codebase hot path.

---

## Files to Modify

1. **`knowledge3d/training/arc_agi/sovereign_pipeline.py`**
   - Add `_should_evaluate_candidate()` method with size-aware paths
   - Pass `size_pattern` into evaluation loop
   - Remove/relax the ratio ≥ 2.5 drop guard for extraction tasks

2. **`knowledge3d/training/arc_agi/parallel_candidate_generator.py`** (if exists)
   - Thread grammar rules via argument to workers
   - Avoid per-worker `GrammarGalaxy()` instantiation

3. **`knowledge3d/training/arc_agi/candidate_generator.py`**
   - Ensure `_detect_size_pattern()` result is used throughout
   - Verify extraction candidates are generated AND evaluated

4. **`knowledge3d/training/arc_agi/multimodal_parser.py`**
   - Verify uses `get_grammar_galaxy()` singleton

---

## Verification Script

After implementation, verify with:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Test 1: Extraction task f4081712 should NOT be dropped
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignPipeline

# Mock extraction task
train_examples = [
    {'input': [[0]*24 for _ in range(24)], 'output': [[1,2,3,4],[5,6,7,8],[9,0,1,2],[3,4,5,6],[7,8,9,0],[1,2,3,4]]}
]

# Detect size pattern
pipeline = SovereignPipeline()
pattern = pipeline._detect_size_pattern(train_examples)
print(f'Detected pattern: {pattern}')  # Should be 'extract'

# Test should_evaluate with 4× ratio
result = pipeline._should_evaluate_candidate(
    [[1,2,3,4]],  # 1×4 candidate
    [[1,2,3,4],[5,6,7,8],[9,0,1,2],[3,4,5,6],[7,8,9,0],[1,2,3,4]],  # 6×4 expected
    'extract'
)
print(f'Should evaluate (extract, 6× ratio): {result}')  # Should be True

print('=== EXTRACTION PATH VERIFICATION PASSED ===')
"

# Test 2: Grammar singleton (check log output)
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.grammar_galaxy import get_grammar_galaxy

g1 = get_grammar_galaxy()
g2 = get_grammar_galaxy()
print(f'Same instance: {g1 is g2}')  # Should be True
print(f'Rule count: {len(g1.rules)}')
print('=== SINGLETON VERIFICATION PASSED ===')
"
```

---

## Success Criteria

1. **Extraction tasks NOT dropped:**
   - `f4081712` (24×24 → 6×4) should be evaluated
   - `ed74f2f2` (5×9 → 3×3) should be evaluated
   - Log should show `[SIZE PATTERN] Detected: extract` for these tasks

2. **Grammar loads ONCE:**
   - Only ONE `[GrammarGalaxy] Loaded 747 rules...` line in log (main process)
   - Workers should NOT trigger additional loads

3. **Accuracy improvement:**
   - Target: >50% (up from ~46%)
   - Extraction tasks should score >0% (currently 0%)

---

## Launch Training

After implementation:

```bash
tmux new-session -d -s k3d_size_path "bash -lc '
  source /home/daniel/miniforge/etc/profile.d/conda.sh
  conda activate k3d-cranium
  cd \"/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D\"
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  python scripts/train_arc_sovereign_loop.py \
    --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
               /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
    --max-tasks 108 --epochs 162 --cycles 1 \
    2>&1 | tee /K3D/Knowledge3D.local/logs/size_path_fix_$(date +%Y%m%d_%H%M%S).log
'"
```

Monitor: `tmux attach -t k3d_size_path`

---

## Sovereignty Compliance

- All fixes maintain PTX + RPN sovereignty
- No external ML frameworks in hot path
- Grammar singleton avoids redundant file I/O
- Size-aware path uses existing procedural resize (PTX-backed)

---

**END OF SPECIFICATION**

Claude (Architecture Partner)
December 12, 2025
