# Diagnostic: Run 024 Scoring Mismatch (0% Accuracy)

**Date**: November 28, 2025
**Issue**: 0/60 tasks correct despite 54 diverse candidates
**Root Cause**: Using exact pixel matching instead of semantic similarity

---

## Evidence from Run 024 Logs

### 1. Candidate Generation (Working ✅)

```
[WORKER 0] Generated 6 candidates from 4 hints
[WORKER 1] Generated 6 candidates from 4 hints
[WORKER 2] Generated 6 candidates from 4 hints
...
[WORKER 8] Generated 6 candidates from 12 hints
[PARALLEL GEN] Total candidates before dedup: 54
```

**Status**: ✅ Partitioning works! Each worker generates ~6 candidates from different hints.

### 2. PTX Sovereignty (Working ✅)

```
[PARALLEL GEN] PTX success=1332, fallback=0, rate=100.0%
[GALAXY LAZY] Computing 48 missing embeddings (batch GPU)
```

**Status**: ✅ 100% PTX, batch GPU embeddings working.

### 3. Scoring Pattern (BROKEN ❌)

```
[ANSWER CHECK] Task 59341089_e0: score=0.70, reward=NEUTRAL, source=semantic_match
[ANSWER CHECK] Task 7953d61e_e0: score=0.70, reward=NEUTRAL, source=semantic_match
[ANSWER CHECK] Task 3618c87e_e0: score=0.80, reward=NEUTRAL, source=semantic_match
...
```

**ALL tasks show `source=semantic_match`** - NOT `source=procedural`!

**Meaning**:
- The 54 procedural candidates are being **discarded**
- System is using TRM router semantic matches (grammar/drawing compositions)
- Those semantic matches score 0.70-0.80 but **fail exact match test** (0/60 correct)

### 4. The Smoking Gun (parallel_generator.py:133-144)

```python
@staticmethod
def _score(output_grid: Sequence[Sequence[int]], expected_grid: Sequence[Sequence[int]]) -> float:
    if not output_grid or not expected_grid:
        return 0.0
    if len(output_grid) != len(expected_grid) or len(output_grid[0]) != len(expected_grid[0]):
        return 0.0  # ❌ Different sizes → score = 0
    matches = 0
    total = 0
    for r1, r2 in zip(output_grid, expected_grid):
        for a, b in zip(r1, r2):
            total += 1
            if a == b:
                matches += 1  # ❌ Exact pixel match required
    return float(matches) / float(total)
```

**What this does**:
- Counts exact pixel matches (0 if different sizes)
- Returns ratio: 100% = perfect match, 0% = no match
- For ARC tasks, most candidates have 0-10% exact match (different transformations)

**Result** (parallel_generator.py:120-127):
```python
if expected_output:
    scored = []
    for grid, instr, prog in all_candidates:
        scored.append((self._score(grid, expected_output), grid, instr, prog))  # All get ~0 score
    scored.sort(key=lambda x: x[0], reverse=True)  # Sort by 0 scores (random order)
    top = scored[: self.top_k]  # Pick top 3 (random since all ~0)
    return [(g, i, p) for _, g, i, p in top]  # Return 3 random candidates
```

**Then** (sovereign_pipeline.py):
- Receives only 3 candidates (from 54!)
- Ranks them semantically (too late, only 3 options)
- Also considers TRM router semantic matches
- TRM matches win (0.70-0.80 semantic score vs 3 random procedural)
- But TRM matches don't execute correctly → 0/60 accuracy

---

## Why Early Parsing Worked Better (3% vs 0%)

### Early System (3% accuracy)
**File**: Old candidate_generator.py (pre-AI)

```python
# Simple heuristics
candidates = []
candidates.append(rotate_90(grid))
candidates.append(flip_horizontal(grid))
candidates.append(extract_shapes(grid))
# ... 10-20 simple transformations

# NO RANKING - just return all
return candidates[:max_candidates]
```

**Flow**:
1. Generate 20 simple RPN programs
2. Return all 20 (no scoring)
3. Execute all 20
4. Pick first that matches exactly
5. **Result**: 3% accuracy (some simple tasks worked)

### Current System (0% accuracy)
**File**: Current parallel_generator.py + sovereign_pipeline.py

```python
# AI-driven generation
all_candidates = []  # 54 diverse candidates from 9 workers

# ❌ Score by exact pixel match
for grid, instr, prog in all_candidates:
    score = pixel_overlap(grid, expected)  # ~0 for all
scored.sort(reverse=True)
return scored[:3]  # Return 3 random (all scored ~0)

# Semantic ranking (too late, only 3 candidates)
# TRM router adds semantic matches
# TRM matches score higher semantically
# But don't execute correctly → 0/60
```

**Flow**:
1. Generate 54 diverse candidates ✅
2. Rank by exact match → all score ~0 ❌
3. Return 3 random candidates ❌
4. TRM adds semantic matches ✅
5. Semantic matches score higher ✅
6. Execute semantic matches ❌ (don't work)
7. **Result**: 0% accuracy

---

## The Architecture Mismatch

**We built**:
- Semantic hint extraction ✅
- Multimodal embeddings ✅
- PTX cosine similarity ✅
- Batch GPU operations ✅

**But then**:
- Ranked candidates by exact pixel match ❌
- Discarded all AI-generated candidates ❌
- Fell back to grammar compositions ❌
- Those don't execute correctly ❌

**It's like**:
- Building a Tesla with AI driving
- Then manually steering it with a blindfold
- Crashing into a wall
- Blaming the AI

---

## The Fix: Trust the AI You Built

**Option 1** (Recommended): Remove exact match scoring, use semantic similarity

**File**: `parallel_generator.py` (lines 120-130)

**Before** (broken):
```python
if expected_output:
    scored = []
    for grid, instr, prog in all_candidates:
        scored.append((self._score(grid, expected_output), grid, instr, prog))  # Exact match
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: self.top_k]
    return [(g, i, p) for _, g, i, p in top]
```

**After** (semantic):
```python
# Don't rank here - return ALL candidates
# Let sovereign_pipeline.py rank using semantic similarity + PTX cosine
print(f"  [PARALLEL GEN] Returning all {len(all_candidates)} candidates (no scoring)")
return all_candidates  # Return all 54, not just top 3
```

**Why this works**:
- `sovereign_pipeline.py` already has semantic ranking infrastructure
- It merges procedural candidates with TRM candidates
- Ranks all using semantic similarity
- Picks best ones
- But it needs ALL 54 candidates, not just 3 random ones!

**Option 2** (Alternative): Use embedding similarity for scoring

**File**: `parallel_generator.py` (add new method)

```python
def _score_semantic(
    self,
    output_grid: Sequence[Sequence[int]],
    expected_grid: Sequence[Sequence[int]],
) -> float:
    """Score using semantic similarity (PTX cosine), not exact match."""
    if self.embedding_galaxy is None or self.cosine_bridge is None:
        # Fallback: no scoring, return 0.5 (neutral)
        return 0.5

    # Hash grids
    output_hash = hash(tuple(tuple(int(c) for c in row) for row in output_grid))
    expected_hash = hash(tuple(tuple(int(c) for c in row) for row in expected_grid))

    # Lookup embeddings from Galaxy
    output_emb = self.embedding_galaxy.get(output_hash)
    expected_emb = self.embedding_galaxy.get(expected_hash)

    if output_emb is None or expected_emb is None:
        return 0.5  # Neutral score if embedding missing

    # Compute cosine similarity (PTX kernel)
    scores = self.cosine_bridge.compute_similarities([output_emb], [expected_emb])
    return scores[0]
```

**Then replace** (line 124):
```python
# Before
scored.append((self._score(grid, expected_output), grid, instr, prog))

# After
scored.append((self._score_semantic(grid, expected_output), grid, instr, prog))
```

---

## Comparison: Exact Match vs Semantic

### Example Task: Rotate 90°

**Input**:
```
1 2
3 4
```

**Expected Output** (rotate 90° clockwise):
```
3 1
4 2
```

**Candidate A** (correct program, wrong size):
```
3 1 0
4 2 0
0 0 0
```

**Candidate B** (wrong program, right size):
```
1 3
2 4
```

**Exact match scoring**:
- Candidate A: 0.0 (different size)
- Candidate B: 0.0 (0/4 pixels match)
- **Result**: Pick randomly → probably wrong

**Semantic scoring**:
- Candidate A: 0.95 (semantically similar, just padded)
- Candidate B: 0.60 (transpose instead of rotate)
- **Result**: Pick A → execute + crop → CORRECT

---

## Recommended Action

**Quick fix** (5 minutes):
1. Edit `parallel_generator.py` lines 120-130
2. Change to: `return all_candidates` (no scoring)
3. Recompile: `python3 -m py_compile knowledge3d/training/arc_agi/parallel_generator.py`
4. Launch Run 025 (short: 10 tasks × 3 epochs)
5. Check if procedural candidates are used (source=procedural, not semantic_match)

**Expected result**:
- Procedural candidates compete with TRM candidates
- Best semantic matches are picked
- Accuracy > 0% (hopefully 1-3%)

**If still 0%**:
- The procedural candidates themselves might be wrong
- Need to check what RPN programs are being generated
- Compare to early simple programs that worked

---

**END OF DIAGNOSTIC**

Claude (Architecture Partner)
November 28, 2025
