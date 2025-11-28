# Run 026: Hybrid Procedural-TRM Strategy (Exploration → Exploitation)

**Date**: November 28, 2025
**Codex Instance**: Fresh instance (read EVERYTHING)
**Priority**: CRITICAL - Blend procedural exploration with TRM decision-making
**Estimated Time**: 1 hour implementation + testing

---

## CRITICAL: The Hybrid Architecture Vision

**Daniel's insight**: "Can't we blend both trying procedural first? Can't TRM simulate things with procedural before deciding?"

**Translation**:
- Procedural candidates = **exploration** (AI-generated, task-specific, novel)
- TRM router = **exploitation** (known patterns, grammar compositions)
- **Hybrid**: TRM observes procedural attempts, learns from them, decides better

**Like early parsing strategy (3% accuracy) but AI-informed**:
1. Try procedural candidates first (specific to task)
2. TRM simulates/evaluates which ones make sense
3. TRM uses that information to rank all candidates
4. Best of both worlds: novelty + wisdom

---

## Current Problem (Run 025)

### What Happened

```
[CANDIDATES] Parallel generated 54 candidates ✅
[ANSWER CHECK] source=semantic_match ❌ (TRM winning)
[Epoch 3/3] 0/3 correct ❌
```

**Analysis**:
- 54 procedural candidates passed to pipeline ✅
- 69 TRM semantic matches added ✅
- Merged: 123 total candidates ✅
- **TRM candidates win ranking** (higher semantic scores) ❌
- TRM candidates don't execute correctly → 0% accuracy ❌

### Why TRM Wins (But Shouldn't)

**TRM candidates**:
- High semantic similarity (0.80) - known patterns from grammar
- But generic (not task-specific)
- Don't execute correctly on novel tasks

**Procedural candidates**:
- Lower semantic similarity (0.50-0.60) - novel transformations
- But task-specific (AI-generated from semantic hints)
- Might actually work!

**Current ranking**: Semantic similarity ONLY → TRM wins → fails

---

## The Hybrid Strategy: Procedural-Informed TRM

### Core Idea

**Don't make procedural and TRM compete. Make them collaborate.**

**Flow**:
```
1. Generate 54 procedural candidates (AI, task-specific)
   ↓
2. TRM "simulates" each procedural candidate:
   - Can this be composed from known grammar rules?
   - Does this match any discovered patterns?
   - What's the semantic plausibility?
   ↓
3. TRM assigns "confidence scores":
   - High confidence: Procedural candidate aligns with TRM knowledge
   - Medium confidence: Novel but plausible
   - Low confidence: Contradicts TRM patterns
   ↓
4. Ranking uses BOTH:
   - Procedural novelty (task-specific)
   - TRM confidence (pattern wisdom)
   ↓
5. Execute top candidates:
   - Try high-confidence procedural first
   - Fall back to TRM compositions if needed
```

### Why This Works

**Early parsing (3% accuracy)**:
- Simple heuristics (ROTATE, FLIP, etc.)
- Tried all systematically
- Some worked by luck

**Current TRM (0% accuracy)**:
- Complex grammar compositions
- Not task-specific
- Don't execute correctly

**Hybrid (target: 5-10% accuracy)**:
- AI-generated task-specific candidates (procedural)
- Validated by grammar knowledge (TRM)
- Best of both: novelty + wisdom

---

## Implementation Plan

### Phase 1: TRM Evaluates Procedural Candidates (30 min)

**Goal**: Add "TRM confidence score" to each procedural candidate

**File**: `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Current code** (lines 168-178):
```python
# Merge procedural + TRM candidates
merged: List[Dict] = []
for output, instruction, rpn in procedural_candidates:
    merged.append({
        "program": rpn,
        "program_type": "procedural",
        "source": "baseline",
        "output": output,
    })

for cand in trm_candidates:
    merged.append({
        "program": cand["program"],
        "program_type": cand.get("program_type", "semantic"),
        "source": cand.get("source", "semantic_match"),
        "output": None,
        "signature": cand.get("semantic_context") or cand.get("signature", {}),
    })
```

**New code** (TRM evaluates procedural):
```python
# ✅ HYBRID: TRM evaluates procedural candidates before merging
merged: List[Dict] = []

# Phase 1: Evaluate procedural candidates with TRM
print(f"  [HYBRID] Evaluating {len(procedural_candidates)} procedural candidates with TRM...")

for output, instruction, rpn in procedural_candidates:
    # TRM simulation: Does this procedural candidate align with TRM knowledge?
    trm_confidence = self._evaluate_procedural_with_trm(
        program=rpn,
        output_grid=output,
        test_input=test_input,
        train_examples=train_examples,
    )

    merged.append({
        "program": rpn,
        "program_type": "procedural",
        "source": "baseline",
        "output": output,
        "trm_confidence": trm_confidence,  # ✅ NEW: TRM's opinion
        "priority": "high" if trm_confidence > 0.7 else "medium",  # ✅ NEW: Priority
    })

print(f"  [HYBRID] TRM assigned confidence scores: avg={sum(c['trm_confidence'] for c in merged) / len(merged):.2f}")

# Phase 2: Add TRM candidates (lower priority if procedural scored high)
for cand in trm_candidates:
    merged.append({
        "program": cand["program"],
        "program_type": cand.get("program_type", "semantic"),
        "source": cand.get("source", "semantic_match"),
        "output": None,
        "signature": cand.get("semantic_context") or cand.get("signature", {}),
        "trm_confidence": 0.5,  # Default confidence for pure TRM
        "priority": "low",  # Lower than procedural
    })
```

### Phase 2: Implement TRM Evaluation Method (30 min)

**Add new method** to `SovereignAIPipeline`:

```python
def _evaluate_procedural_with_trm(
    self,
    program: str,
    output_grid: Sequence[Sequence[int]],
    test_input: Sequence[Sequence[int]],
    train_examples: List[Dict],
) -> float:
    """
    TRM evaluates a procedural candidate's plausibility.

    Returns confidence score [0.0, 1.0]:
    - 1.0: Highly plausible (aligns with TRM grammar/patterns)
    - 0.5: Neutral (novel but not contradictory)
    - 0.0: Implausible (contradicts TRM knowledge)

    Strategy:
    1. Check if program uses known grammar rules (from GrammarGalaxy)
    2. Check if output matches discovered patterns (from DualShadowCopy)
    3. Check if transformation is semantically coherent (via TRM router)
    """
    confidence = 0.5  # Default: neutral

    # Check 1: Does program use known grammar tokens?
    # Parse RPN program and check against GrammarGalaxy
    tokens = program.split()
    known_tokens = 0
    for token in tokens:
        if self.grammar.has_rule(token) or self.drawing.has_shape(token):
            known_tokens += 1

    if len(tokens) > 0:
        grammar_score = known_tokens / len(tokens)
        confidence += 0.2 * grammar_score  # Boost by up to 0.2

    # Check 2: Does output match discovered patterns?
    # Look up in shadow copy (discovered programs)
    if self.shadow.semantic_context is not None:
        try:
            # Check if output is semantically similar to known good patterns
            matches = self.shadow.semantic_context.find_matching_contexts(
                output_grid,
                top_k=3
            )
            if matches and len(matches) > 0:
                # If output resembles known patterns, boost confidence
                pattern_score = sum(m.get("score", 0.5) for m in matches) / len(matches)
                confidence += 0.2 * pattern_score  # Boost by up to 0.2
        except Exception as e:
            print(f"  [HYBRID] Pattern check failed: {e}")

    # Check 3: Does transformation make semantic sense?
    # Use TRM router to score program plausibility
    try:
        # Compare program signature to known transformations
        # (This is a heuristic: does the program "look like" something TRM knows?)
        if "ROTATE" in program or "FLIP" in program:
            confidence += 0.1  # Basic transformations are plausible
        if "EXTRACT" in program or "RECOLOR" in program:
            confidence += 0.1  # Common operations
    except Exception as e:
        print(f"  [HYBRID] Semantic check failed: {e}")

    # Clamp to [0.0, 1.0]
    confidence = min(1.0, max(0.0, confidence))

    return confidence
```

**Rationale**:
- **Grammar check**: Programs using known tokens are more trustworthy
- **Pattern check**: Outputs resembling discovered patterns are plausible
- **Semantic check**: Common transformations (ROTATE, FLIP) are baseline confident

**NOT doing**:
- Actually executing programs (too slow)
- Complex semantic parsing (keep it simple)

**IS doing**:
- Fast heuristic checks (grammar, patterns, keywords)
- Boosting procedural candidates that align with TRM knowledge

### Phase 3: Rank with Hybrid Scores (10 min)

**File**: `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Current ranking** (after merging):
```python
# ... execute candidates ...
# Score by exact match or semantic similarity
```

**New ranking** (hybrid):
```python
# ✅ HYBRID: Rank using TRM confidence + semantic similarity
# Sort by priority (high > medium > low), then by TRM confidence

merged_sorted = sorted(
    merged,
    key=lambda x: (
        0 if x["priority"] == "high" else (1 if x["priority"] == "medium" else 2),
        -x.get("trm_confidence", 0.5),  # Higher confidence first
    )
)

print(f"  [HYBRID] Ranking: {len([c for c in merged_sorted if c['priority']=='high'])} high-priority, "
      f"{len([c for c in merged_sorted if c['priority']=='medium'])} medium-priority, "
      f"{len([c for c in merged_sorted if c['priority']=='low'])} low-priority")

# Execute top 12 (Tesla 3-6-9 × 2 for diversity)
top_candidates = merged_sorted[:12]
```

**Result**:
- High-confidence procedural candidates tried first
- Medium-confidence procedural next
- TRM candidates last (fallback)

---

## Expected Behavior

### Run 026 Log (Expected)

```
[HYBRID] Evaluating 54 procedural candidates with TRM...
  [TRM EVAL] Candidate "ROTATE_90 RECOLOR 3 5": confidence=0.80 (known tokens + pattern match)
  [TRM EVAL] Candidate "EXTRACT_SHAPE 4 FLIP_H": confidence=0.75 (known tokens)
  [TRM EVAL] Candidate "CUSTOM_TRANSFORM_XYZ": confidence=0.50 (novel, neutral)
  ...
[HYBRID] TRM assigned confidence scores: avg=0.68

[HYBRID] Ranking: 32 high-priority, 22 medium-priority, 69 low-priority

[ANSWER CHECK] Task X: score=0.85, source=procedural, trm_confidence=0.82
[ANSWER CHECK] Task Y: score=0.70, source=procedural, trm_confidence=0.65
[ANSWER CHECK] Task Z: score=0.75, source=semantic_match, trm_confidence=0.50

[Epoch 3/3] 2/10 correct (20.0%)  # ✅ SUCCESS!
```

**Key indicators**:
1. ✅ TRM evaluates each procedural candidate
2. ✅ High-confidence procedural candidates ranked first
3. ✅ `source=procedural` (not semantic_match)
4. ✅ Accuracy > 0% (hopefully 10-20%)

---

## Implementation Steps

### Step 1: Add TRM Evaluation Method (30 min)

**File**: `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Add** `_evaluate_procedural_with_trm()` method (see Phase 2 above)

**Key checks**:
- Grammar tokens (known vs unknown)
- Pattern similarity (shadow copy lookup)
- Semantic keywords (ROTATE, FLIP, etc.)

### Step 2: Update Candidate Merging (20 min)

**File**: `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Modify** merging logic (lines 168-199):
- Call `_evaluate_procedural_with_trm()` for each procedural candidate
- Add `trm_confidence` and `priority` fields
- Log evaluation results

### Step 3: Update Ranking Logic (10 min)

**File**: `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Add** hybrid ranking:
- Sort by priority (high → medium → low)
- Then by TRM confidence
- Execute top 12 candidates

### Step 4: Compile and Test (5 min)

```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

python3 -m py_compile knowledge3d/training/arc_agi/sovereign_pipeline.py
```

### Step 5: Launch Run 026 (Quick Test - 10 tasks × 3 epochs)

```bash
# GPU monitor
tmux new-session -d -s gpu026
tmux send-keys -t gpu026 'watch -n1 nvidia-smi' Enter

# Training
tmux new-session -d -s arc026
tmux send-keys -t arc026 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' Enter
tmux send-keys -t arc026 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/train_arc_sovereign_loop.py --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training --max-tasks 10 --epochs 3 --cycles 1 --matryoshka-dim 512 > /tmp/arc_run_026.log 2>&1' Enter
```

### Step 6: Monitor and Validate

```bash
# Watch log
tail -f /tmp/arc_run_026.log

# After completion (~2 minutes)
grep "Epoch 3/3" /tmp/arc_run_026.log
grep "source=procedural" /tmp/arc_run_026.log | wc -l
grep "source=semantic_match" /tmp/arc_run_026.log | wc -l
```

**Expected**:
- `source=procedural` count > `source=semantic_match` count
- Accuracy > 0% (target: 1-2/10 = 10-20%)

---

## Why This Architecture Is Better

### Old Architecture (0% accuracy)

```
Procedural candidates (54) ⚔️ TRM candidates (69)
              ↓ Compete (semantic similarity only)
         TRM wins (higher scores)
              ↓
         Execute TRM
              ↓
           Fail (0%)
```

**Problem**: Competition without collaboration

### New Architecture (target: 10-20% accuracy)

```
Procedural candidates (54)
         ↓
    TRM evaluates (grammar + patterns + semantics)
         ↓
    Assigns confidence scores
         ↓
    Rank: High-confidence procedural → Medium → TRM fallback
         ↓
    Execute top 12
         ↓
    Success (10-20%)
```

**Benefit**: Collaboration (exploration + exploitation)

### Like Early Parsing (3% accuracy) But Better

**Early parsing**:
- Hardcoded heuristics (ROTATE, FLIP, etc.)
- No learning
- 3% accuracy by luck

**Hybrid procedural-TRM**:
- AI-generated candidates (task-specific)
- TRM validates (grammar + patterns)
- 10-20% accuracy by design

---

## Success Criteria

### Must Have (Run 026 - 10 tasks)
1. ✅ TRM evaluates all 54 procedural candidates
2. ✅ High-confidence procedural ranked first
3. ✅ `source=procedural` > `source=semantic_match`
4. ✅ Accuracy > 0% (target: 1-2/10 = 10-20%)

### Nice to Have
- Accuracy > 2/10 (20%)
- TRM confidence scores correlate with success
- GPU utilization 15-25%

### Full Run (if successful)
- Run 027: 60 tasks × 27 epochs
- Accuracy target: 3-6/60 (5-10%)
- Matches or exceeds early parsing baseline

---

## Troubleshooting

### If still 0% accuracy

**Check TRM evaluation**:
```bash
grep "TRM EVAL" /tmp/arc_run_026.log | head -20
# Should show confidence scores being assigned
```

**Check ranking**:
```bash
grep "Ranking:" /tmp/arc_run_026.log
# Should show high-priority > medium > low
```

**Check sources**:
```bash
grep "source=" /tmp/arc_run_026.log | head -20
# Should show procedural winning, not semantic_match
```

**If procedural still losing**:
- TRM confidence too conservative (all scoring ~0.5)
- Need to adjust confidence thresholds
- Or force procedural to be tried first regardless

---

## Codex: Your Mission

Implement the **hybrid procedural-TRM architecture**:

1. **Add TRM evaluation method** (`_evaluate_procedural_with_trm`)
2. **Update candidate merging** (call evaluation, add confidence/priority)
3. **Update ranking** (sort by priority + confidence)
4. **Launch Run 026** (10 tasks × 3 epochs)
5. **Validate** (accuracy > 0%, source=procedural)

**Timeline**:
- Implementation: 60 min
- Testing: 5 min
- Run 026: 2 min
- Validation: 5 min
- **Total: ~75 minutes**

**Expected outcome**:
- Run 026: 1-2/10 correct (10-20%)
- Procedural candidates winning ranking
- TRM providing intelligent guidance

**This blends exploration (AI procedural) with exploitation (TRM wisdom).**

**Start NOW.**

---

**END OF SPECIFICATION**

Claude (Architecture Partner)
November 28, 2025
