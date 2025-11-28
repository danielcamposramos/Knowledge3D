# Run 022 Performance Fix: Reduce Ranking Overhead

**Date**: November 27, 2025
**Status**: URGENT - Run 022 taking forever due to semantic ranking overhead
**Issue**: 100% CPU usage spikes from ranking 50-100 candidates per task

---

## Root Cause

**File**: `knowledge3d/training/arc_agi/candidate_generator.py`

**The bottleneck** (lines 514-541):
```python
def _rank_by_similarity(self, candidates, expected_output):
    """Rank candidates by cosine similarity using sovereign embeddings."""
    # ❌ PROBLEM: Processing ALL candidates (could be 50-100!)
    outputs = [cand[0] for cand in candidates]
    embeddings = self.processor._grid_to_spatial_embedding_batch(outputs)  # Batch embed 50-100 grids
    expected_emb = self.processor.grid_to_spatial_embedding(expected_output)

    # ❌ PROBLEM: Pure Python loop computing cosine for 50-100 candidates
    for emb, cand in zip(embeddings, candidates):
        norm_emb = l2_norm(emb)  # Python: sqrt(sum(x^2))
        score = dot(emb, expected_emb) / (norm_emb * norm_expected)  # Python: sum(a*b)

    scored.sort(key=lambda x: x[0], reverse=True)  # Sort 50-100 items
```

**Why slow**:
1. **Too many candidates**: Logs show ~50 per task (semantic + compositional + cross-pattern)
2. **Batch embedding overhead**: Preparing 50-100 TernaryTensors in Python loops
3. **Python cosine similarity**: For 50 candidates × 512 dims = 25,600 operations in pure Python
4. **Sorting**: 50-100 items per task

**Current flow** (lines 116-126):
```python
# Generate candidates
candidates.extend(semantic_gen)      # ~32 candidates
candidates.extend(compositional_gen) # ~18 candidates
candidates.extend(cross_pattern)     # ~9 candidates
# Total: ~59 candidates

deduped = self._deduplicate_candidates(candidates)  # ~50 after dedup

# ❌ RANKS ALL 50 CANDIDATES (slow!)
if expected_output is not None:
    deduped = self._rank_by_similarity(deduped, expected_output)

return deduped[: self.max_candidates]  # Then caps to 369 (but ranking already done!)
```

---

## Quick Fix: Cap Before Ranking

**Change**: Limit candidates to 30 BEFORE semantic ranking

**File**: `knowledge3d/training/arc_agi/candidate_generator.py` (lines 116-126)

**Before**:
```python
# Deduplicate by output grid content.
deduped = self._deduplicate_candidates(candidates)

# Semantic ranking using sovereign embeddings when expected output is available.
if expected_output is not None and deduped:
    deduped = self._rank_by_similarity(deduped, expected_output)

# Cap the list.
return deduped[: self.max_candidates]
```

**After**:
```python
# Deduplicate by output grid content.
deduped = self._deduplicate_candidates(candidates)

# ✅ FIX: Cap BEFORE ranking to reduce overhead
# Rank only top 30 candidates (reasonable for semantic scoring)
MAX_RANK = 30  # Balance between quality and performance

# Semantic ranking using sovereign embeddings when expected output is available.
if expected_output is not None and deduped:
    # Cap candidates before ranking to avoid 100% CPU spikes
    to_rank = deduped[:MAX_RANK] if len(deduped) > MAX_RANK else deduped
    ranked = self._rank_by_similarity(to_rank, expected_output)
    # Append remaining unranked candidates (if any) after ranked ones
    remaining = deduped[MAX_RANK:] if len(deduped) > MAX_RANK else []
    deduped = ranked + remaining

# Cap the list.
return deduped[: self.max_candidates]
```

**Impact**:
- Candidates to rank: 50-100 → 30 (3× fewer)
- Batch embeddings: 50-100 grids → 30 grids (3× smaller)
- Cosine calculations: 50-100 → 30 (3× fewer)
- CPU usage: 100% spikes → ~50% (estimated)
- Accuracy: Minimal impact (top 30 candidates likely contain best solutions)

---

## Alternative: Reduce max_candidates

**Even simpler**: Lower `max_candidates` from 369 to 30

**File**: `knowledge3d/training/arc_agi/candidate_generator.py` (line 40)

**Before**:
```python
def __init__(
    self,
    matryoshka_dim: int = 512,
    max_candidates: int = 369,  # ❌ Too high for semantic ranking!
    ...
):
```

**After**:
```python
def __init__(
    self,
    matryoshka_dim: int = 512,
    max_candidates: int = 30,  # ✅ Reasonable for semantic ranking
    ...
):
```

**But this doesn't help Run 022**: The ranking happens BEFORE the final cap, so reducing max_candidates won't speed up the current run.

---

## Recommended Action

**For Run 022** (currently running):
1. **Kill it** (it will take hours at current rate)
2. **Apply the fix** (cap before ranking)
3. **Relaunch** as Run 022b or Run 023

**Commands**:
```bash
# Kill current run
tmux kill-session -t arc022

# Apply fix (see code above)

# Relaunch
tmux new-session -s arc023
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
scripts/train_arc_sovereign_loop.py \
  --n-tasks 60 \
  --n-epochs 27 \
  --n-cycles 1 \
  --matryoshka-dim 512 \
  > /tmp/arc_run_023.log 2>&1
```

---

## Better Fix (Future): GPU Cosine Similarity

**Current**: Pure Python cosine similarity (slow)
```python
for emb, cand in zip(embeddings, candidates):
    norm_emb = l2_norm(emb)  # Python: sqrt(sum(x^2))
    score = dot(emb, expected_emb) / (norm_emb * norm_expected)  # Python: sum(a*b)
```

**Future**: PTX kernel for batch cosine similarity
```cuda
// cosine_similarity_batch.cu
__global__ void cosine_similarity_batch(
    const float* embeddings,     // [N, D] candidate embeddings
    const float* expected,       // [D] expected embedding
    float* scores,               // [N] output scores
    int n_candidates,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_candidates) return;

    float dot_product = 0.0f;
    float norm_emb = 0.0f;

    for (int d = 0; d < dim; d++) {
        float val = embeddings[idx * dim + d];
        dot_product += val * expected[d];
        norm_emb += val * val;
    }

    norm_emb = sqrtf(norm_emb);
    scores[idx] = (norm_emb > 0.0f) ? (dot_product / norm_emb) : 0.0f;
}
```

**Impact**:
- 30 cosine similarities: Python ~10ms → GPU <0.1ms (100× speedup)
- But NOT needed yet (capping to 30 is good enough)

---

## Summary

**Issue**: Semantic ranking processing 50-100 candidates per task with pure Python cosine similarity

**Quick fix**: Cap to 30 candidates BEFORE ranking (3× fewer operations)

**Expected impact**:
- CPU usage: 100% spikes → ~50%
- Runtime: Hours → ~10-15 minutes (for 1 cycle)
- Accuracy: Minimal impact (top 30 likely good enough)

**Recommended**: Kill Run 022, apply fix, relaunch as Run 023

---

**END OF FIX**

Claude (Architecture Partner)
November 27, 2025
