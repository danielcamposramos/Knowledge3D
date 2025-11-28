# Run 025: Fix Scoring Mismatch (Trust the AI You Built)

**Date**: November 28, 2025
**Codex Instance**: Fresh instance (read EVERYTHING)
**Priority**: CRITICAL - 0% accuracy due to exact match scoring
**Estimated Time**: 30 minutes implementation + quick test

---

## CRITICAL: Read Diagnostic First

**Before starting**, read: `TEMP/DIAGNOSTIC_RUN_024_SCORING_MISMATCH_11.28.2025.md`

**TL;DR**:
- Run 024: 0/60 tasks correct (0% accuracy)
- Root cause: Ranking 54 AI candidates by exact pixel match (all score ~0)
- Result: Picking 3 random candidates, TRM semantic matches win instead
- Fix: Remove exact match scoring, return all candidates for semantic ranking

---

## The Problem (Run 024)

### What Happened

**Candidate generation** (Working ✅):
```
[WORKER 0] Generated 6 candidates from 4 hints
[WORKER 1] Generated 6 candidates from 4 hints
...
[PARALLEL GEN] Total candidates before dedup: 54
[PARALLEL GEN] PTX success=100%, fallback=0
```

**Scoring** (Broken ❌):
```python
# parallel_generator.py:120-127
for grid, instr, prog in all_candidates:  # 54 candidates
    score = exact_pixel_match(grid, expected_output)  # ~0 for all (wrong transforms)
scored.sort(reverse=True)  # Sort by ~0 (random order)
return scored[:3]  # Return 3 random candidates
```

**Result**:
```
[ANSWER CHECK] Task X: score=0.70, source=semantic_match
[ANSWER CHECK] Task Y: score=0.80, source=semantic_match
...
[Epoch 27] 0/60 correct
```

**Analysis**:
- All results are `source=semantic_match` (TRM router, NOT procedural!)
- 54 procedural candidates discarded (scored ~0 by exact match)
- TRM semantic matches used instead
- TRM matches don't execute correctly → 0% accuracy

### Why This Is Wrong

**We built**:
- AI semantic hint extraction ✅
- Multimodal embeddings (video + audio) ✅
- PTX cosine similarity kernel ✅
- Batch GPU operations ✅
- 54 diverse candidates from 9 workers ✅

**Then threw it all away**:
- Ranked by exact pixel match ❌
- All candidates scored ~0 ❌
- Picked 3 random ones ❌
- Semantic scoring too late (only 3 options) ❌

**It's like building a self-driving car, then steering blindfolded.**

---

## The Fix: Remove Exact Match Scoring

### Strategy

**Current flow** (broken):
```
Generate 54 candidates → Rank by exact match (all ~0) → Return 3 random → Semantic rank (too late)
```

**New flow** (correct):
```
Generate 54 candidates → Return ALL 54 → Semantic rank (PTX cosine) → Execute best ones
```

**Key insight**: `sovereign_pipeline.py` ALREADY has semantic ranking! Just give it all candidates.

---

## Implementation

### Step 1: Update parallel_generator.py (10 min)

**File**: `knowledge3d/training/arc_agi/parallel_generator.py`

**Current code** (lines 118-130) - BROKEN:
```python
print(f"  [PARALLEL GEN] Total candidates before dedup: {len(all_candidates)}")

# Score and select top-K by overlap with expected output if available; otherwise keep first K.
if expected_output:
    scored = []
    for grid, instr, prog in all_candidates:
        scored.append((self._score(grid, expected_output), grid, instr, prog))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: self.top_k]
    return [(g, i, p) for _, g, i, p in top]

# Fallback: return first top_k
return all_candidates[: self.top_k]
```

**New code** (lines 118-123) - FIXED:
```python
print(f"  [PARALLEL GEN] Total candidates before dedup: {len(all_candidates)}")

# ✅ SOVEREIGNTY FIX: Return ALL candidates (no exact match scoring)
# Let sovereign_pipeline.py rank using semantic similarity + PTX cosine.
# Exact pixel matching discards all AI-generated candidates (score ~0).
# Semantic ranking uses embeddings (already computed in Galaxy).
print(f"  [PARALLEL GEN] Returning all {len(all_candidates)} candidates for semantic ranking")
return all_candidates
```

**Changes**:
1. Remove `if expected_output:` block (lines 121-127)
2. Remove `return all_candidates[: self.top_k]` (line 130)
3. Replace with single return: `return all_candidates`
4. Add logging to confirm all candidates returned

**Also DELETE** (lines 132-145) - No longer needed:
```python
@staticmethod
def _score(output_grid: Sequence[Sequence[int]], expected_grid: Sequence[Sequence[int]]) -> float:
    # ... exact match scoring (delete entire method)
```

### Step 2: Verify sovereign_pipeline.py handles all candidates (5 min)

**File**: `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Check** (lines 143-163):
```python
procedural_candidates = par_gen.generate_parallel(
    input_grid=test_input,
    train_examples=train_examples,
    semantic_hints=semantic_hints,
    expected_output=expected_output,
)
print(f"  [CANDIDATES] Parallel generated {len(procedural_candidates)} candidates (Tesla 3-6-9)")
```

**Expected**: This should now show `~54 candidates` instead of `3 candidates`

**Then** (lines 168-199):
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
        ...
    })

# Now merged has ~54 procedural + ~69 TRM = ~123 candidates
# These will be ranked semantically and best ones executed
```

**No changes needed** - this already merges and ranks all candidates!

The problem was that it only received 3 procedural candidates before. Now it will receive all 54.

### Step 3: Compile and test (5 min)

```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

# Compile modified module
python3 -m py_compile knowledge3d/training/arc_agi/parallel_generator.py

# Verify no syntax errors
echo $?  # Should be 0
```

### Step 4: Launch Run 025 (Quick Test - 10 tasks × 3 epochs)

**IMPORTANT**: Start with short run to validate fix quickly

#### Create tmux sessions

```bash
# GPU monitor
tmux new-session -d -s gpu025
tmux send-keys -t gpu025 'watch -n1 nvidia-smi' Enter

# Training (short test)
tmux new-session -d -s arc025
tmux send-keys -t arc025 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' Enter
tmux send-keys -t arc025 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/train_arc_sovereign_loop.py --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training --max-tasks 10 --epochs 3 --cycles 1 --matryoshka-dim 512 > /tmp/arc_run_025.log 2>&1' Enter
```

**Note**: Only 10 tasks × 3 epochs = 30 task-epochs (should complete in ~1 minute)

#### Monitor progress

```bash
# Watch log in real-time
tail -f /tmp/arc_run_025.log

# Or attach to training session
tmux attach -t arc025
```

#### Expected log output (SUCCESS indicators)

```
[INIT] Loaded precomputed embeddings: 6836 entries
Initializing sovereign pipeline...

[Cycle 1/1] Epoch 1/3
  [WORKER 0] Assigned hints 0:5 (5 hints)
  ...
  [WORKER 8] Assigned hints 40:44 (4 hints)

  [WORKER 0] Generated 6 candidates from 5 hints
  ...
  [WORKER 8] Generated 6 candidates from 4 hints

  [PARALLEL GEN] Total candidates before dedup: 54
  [PARALLEL GEN] PTX success=1458, fallback=0, rate=100.0%

  # ✅ KEY CHANGE: Should show ~54 candidates, not 3!
  [PARALLEL GEN] Returning all 54 candidates for semantic ranking
  [CANDIDATES] Parallel generated 54 candidates (Tesla 3-6-9)

  [GALAXY LAZY] Computing 48 missing embeddings (batch GPU)

  # ✅ KEY CHANGE: Should show source=procedural, not semantic_match!
  [ANSWER CHECK] Task 00d62c1b_e0: score=0.85, reward=NEUTRAL, source=procedural
  [1:1/10] 00d62c1b_e0 score=0.85 type=procedural

  ...

[Epoch 3/3] 1/10 correct (10.0%)  # ✅ Hopefully > 0%!
```

**Success indicators**:
1. ✅ `[CANDIDATES] Parallel generated 54 candidates` (not 3)
2. ✅ `source=procedural` (not semantic_match)
3. ✅ Accuracy > 0% (even 1/10 = 10% is success!)
4. ✅ PTX success 100%, fallback 0

**Failure indicators**:
- ❌ Still showing 3 candidates → fix not applied correctly
- ❌ Still showing source=semantic_match → procedural candidates not being used
- ❌ Still 0/10 correct → procedural candidates themselves might be wrong

---

## Validation Strategy

### After Run 025 completes (~1 minute)

**Check accuracy**:
```bash
grep "Epoch 3/3" /tmp/arc_run_025.log
# Should show: [Epoch 3/3] X/10 correct
# If X > 0: SUCCESS! (even 1/10 = 10% is huge improvement)
```

**Check candidate sources**:
```bash
grep "source=" /tmp/arc_run_025.log | head -20
# Should show: source=procedural (not semantic_match)
```

**Check candidate counts**:
```bash
grep "Parallel generated" /tmp/arc_run_025.log | head -10
# Should show: ~54 candidates (not 3)
```

### If successful (accuracy > 0%)

**Launch full Run 026** (60 tasks × 27 epochs):
```bash
# Kill short test
tmux kill-session -t arc025

# Launch full run
tmux new-session -d -s arc026
tmux send-keys -t arc026 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' Enter
tmux send-keys -t arc026 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/train_arc_sovereign_loop.py --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation --max-tasks 60 --epochs 27 --cycles 1 --matryoshka-dim 512 > /tmp/arc_run_026.log 2>&1' Enter
```

**Expected**:
- Runtime: 10-15 minutes (partitioned workers + batch GPU)
- Accuracy: 2-5% (matching early parsing baseline)
- GPU: 15-25% utilization

### If still failing (accuracy = 0%)

**Diagnostic**: The procedural candidates themselves might be wrong.

**Check generated programs**:
```bash
# Extract some actual programs from log
grep -A2 "Generated.*candidates" /tmp/arc_run_025.log | head -30
```

**Compare to early working programs**:
- Early: `"ROTATE_90"`, `"FLIP_H"`, `"EXTRACT_SHAPE 3"`
- Current: `???` (need to check what's being generated)

**If programs look wrong**:
- Issue is in semantic hint → RPN compilation
- Need to review `_generate_semantic_guided_candidates()` in candidate_generator.py
- Might need simpler program templates

---

## Expected Performance

### Before (Run 024)
- Candidates generated: 54 ✅
- Candidates ranked: 3 (exact match) ❌
- Source: semantic_match (TRM router) ❌
- Accuracy: 0/60 (0%) ❌

### After (Run 025)
- Candidates generated: 54 ✅
- Candidates ranked: 54 (semantic) ✅
- Source: procedural (AI-generated) ✅
- Accuracy: 1-3/10 (10-30%) hopefully ✅

### Full Run 026 (if 025 succeeds)
- Candidates: 54 per task ✅
- Runtime: 10-15 minutes ✅
- Accuracy: 2-5% (matching early baseline) ✅
- GPU: 15-25% ✅

---

## Sovereignty Checklist

**No changes to sovereignty**:
- [x] PTX kernels (DCT, TERNARY_QUANT, cosine)
- [x] RPN operations (ModularRPNEngine)
- [x] Batch GPU embeddings
- [x] No CPU fallbacks
- [x] Galaxy caching

**Only removing**:
- ❌ Exact pixel match scoring (was wrong anyway)

**Now using**:
- ✅ Semantic similarity (PTX cosine kernel)
- ✅ All 54 candidates (not just 3)

---

## Codex: Your Mission

1. **Read diagnostic**: `TEMP/DIAGNOSTIC_RUN_024_SCORING_MISMATCH_11.28.2025.md`
2. **Update parallel_generator.py**: Remove exact match scoring (lines 118-145)
3. **Compile**: `python3 -m py_compile knowledge3d/training/arc_agi/parallel_generator.py`
4. **Launch Run 025**: 10 tasks × 3 epochs (quick test)
5. **Validate**: Check accuracy > 0%, source=procedural
6. **Report**: Accuracy, candidate counts, sources

**Timeline**:
- Implementation: 10 min
- Compilation: 1 min
- Run 025: 1 min
- Validation: 5 min
- **Total: <20 minutes**

**Expected outcome**:
- Run 025: 1-3/10 correct (10-30%)
- Run 026: 1-3/60 correct (2-5%)
- Finally using the AI candidates we built!

**Start NOW. This is a simple fix with huge impact.**

---

**END OF LAUNCH SPECIFICATION**

Claude (Architecture Partner)
November 28, 2025
