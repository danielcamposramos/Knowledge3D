# URGENT FIX: Remove CPU Fallbacks (Sovereignty Violation)

**Date**: November 27, 2025
**Priority**: CRITICAL - Sovereignty violation detected
**Estimated Time**: 15 minutes

---

## The Violation

**Daniel's directive**: "No CPU FALLBACKS Claude, we fail and fix - no fallbacks!! that's actually a violation!"

**What this means**:
- If Galaxy doesn't have an embedding → FAIL LOUDLY (don't compute on-the-fly)
- If parallel generation fails → FAIL LOUDLY (don't fall back to sequential)
- Sovereignty = NO COMPROMISES (fail fast, fix architecture)

**Why**: Fallbacks hide problems and create silent performance degradation. We want to know IMMEDIATELY if preprocessing didn't work or if architecture is broken.

---

## Files to Fix

### 1. candidate_generator.py - Remove On-The-Fly Fallbacks

**File**: `knowledge3d/training/arc_agi/candidate_generator.py`

**Current code** (lines 519-551) - VIOLATES SOVEREIGNTY:
```python
def _rank_by_similarity(
    self,
    candidates: List[Candidate],
    expected_output: Sequence[Sequence[int]],
) -> List[Candidate]:
    """Rank candidates by cosine similarity using sovereign embeddings."""
    if not candidates:
        return candidates

    # Lookup or compute expected embedding.
    expected_hash = self._hash_grid(expected_output)
    expected_emb = None
    if self.embedding_galaxy is not None:
        expected_emb = self.embedding_galaxy.get(expected_hash)
    if expected_emb is None:  # ❌ FALLBACK!
        expected_emb = self.processor.grid_to_spatial_embedding(expected_output)

    # Collect candidate embeddings.
    embeddings: List[List[float]] = []
    for grid, _, _ in candidates:
        h = self._hash_grid(grid)
        emb = None
        if self.embedding_galaxy is not None:
            emb = self.embedding_galaxy.get(h)
        if emb is None:  # ❌ FALLBACK!
            emb = self.processor.grid_to_spatial_embedding(grid)
        embeddings.append(emb)

    # GPU cosine similarity.
    scores = self.cosine_bridge.compute_similarities(embeddings, expected_emb)
    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cand for _, cand in scored]
```

**SOVEREIGN CODE** (NO FALLBACKS):
```python
def _rank_by_similarity(
    self,
    candidates: List[Candidate],
    expected_output: Sequence[Sequence[int]],
) -> List[Candidate]:
    """Rank candidates by cosine similarity using SOVEREIGN Galaxy + PTX (NO FALLBACKS)."""
    if not candidates:
        return candidates

    # ✅ SOVEREIGN: Galaxy MUST be available (fail if not)
    if self.embedding_galaxy is None:
        raise RuntimeError(
            "SOVEREIGNTY VIOLATION: embedding_galaxy is None. "
            "Run preprocessing: python scripts/preprocess_arc_embeddings.py"
        )

    # ✅ SOVEREIGN: Look up expected embedding (fail if missing)
    expected_hash = self._hash_grid(expected_output)
    expected_emb = self.embedding_galaxy.get(expected_hash)
    if expected_emb is None:
        raise RuntimeError(
            f"SOVEREIGNTY VIOLATION: Expected output embedding not found in Galaxy (hash={expected_hash}). "
            "Preprocessing incomplete or grid not in training/test set."
        )

    # ✅ SOVEREIGN: Collect candidate embeddings (fail if any missing)
    embeddings: List[List[float]] = []
    for grid, _, _ in candidates:
        h = self._hash_grid(grid)
        emb = self.embedding_galaxy.get(h)
        if emb is None:
            raise RuntimeError(
                f"SOVEREIGNTY VIOLATION: Candidate embedding not found in Galaxy (hash={h}). "
                "This grid was not in preprocessing set. Check candidate generation."
            )
        embeddings.append(emb)

    # ✅ SOVEREIGN: GPU batch cosine similarity (PTX kernel)
    scores = self.cosine_bridge.compute_similarities(embeddings, expected_emb)
    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cand for _, cand in scored]
```

**Why this is better**:
- Fails LOUDLY if Galaxy missing → tells us preprocessing didn't run
- Fails LOUDLY if embedding missing → tells us which grids weren't preprocessed
- NO silent performance degradation (100% sovereign or 0%)
- Forces us to fix architecture instead of hiding problems

---

### 2. sovereign_pipeline.py - Remove Sequential Fallback

**File**: `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Current code** (lines 143-163) - VIOLATES SOVEREIGNTY:
```python
try:
    from knowledge3d.training.arc_agi.parallel_generator import ParallelCandidateGenerator

    par_gen = ParallelCandidateGenerator(
        num_workers=9,
        candidates_per_worker=6,
        top_k=3,
        matryoshka_dim=self.router.matryoshka_dim,
        shadow_copy=self.shadow,
        codec_embedder=self.codec_embedder,
        embedding_galaxy=self.embedding_galaxy,
        cosine_bridge=self.cosine_bridge,
    )
    procedural_candidates = par_gen.generate_parallel(...)
    print(f"  [CANDIDATES] Parallel generated {len(procedural_candidates)} candidates (Tesla 3-6-9)")
except Exception as e:
    print(f"  [PIPELINE] Parallel generation failed ({e}); falling back to sequential")  # ❌ FALLBACK!
    gen = CandidateGenerator(
        matryoshka_dim=self.router.matryoshka_dim,
        shadow_copy=self.shadow,
        codec_embedder=self.codec_embedder,
        # ❌ MISSING: embedding_galaxy, cosine_bridge
    )
    procedural_candidates = gen.generate_candidates(...)
    print(f"  [CANDIDATES] Generated {len(procedural_candidates)} procedural candidates (max={gen.max_candidates})")
```

**SOVEREIGN CODE** (NO FALLBACKS):
```python
# ✅ SOVEREIGN: Use parallel generation (fail if broken)
from knowledge3d.training.arc_agi.parallel_generator import ParallelCandidateGenerator

par_gen = ParallelCandidateGenerator(
    num_workers=9,
    candidates_per_worker=6,
    top_k=3,
    matryoshka_dim=self.router.matryoshka_dim,
    shadow_copy=self.shadow,
    codec_embedder=self.codec_embedder,
    embedding_galaxy=self.embedding_galaxy,
    cosine_bridge=self.cosine_bridge,
)
procedural_candidates = par_gen.generate_parallel(
    input_grid=test_input,
    train_examples=train_examples,
    semantic_hints=semantic_hints,
    expected_output=expected_output,
)
print(f"  [CANDIDATES] Parallel generated {len(procedural_candidates)} candidates (Tesla 3-6-9)")
```

**Why this is better**:
- No try/except hiding problems
- If parallel generation fails → training fails LOUDLY
- Forces us to fix MathCorePool or worker issues
- NO silent fallback to slower sequential code

---

## Implementation Steps

### Step 1: Fix candidate_generator.py (5 min)

Replace `_rank_by_similarity()` method (lines 519-551) with sovereign version above.

**Test**:
```python
# Should FAIL if Galaxy not loaded
gen = CandidateGenerator(embedding_galaxy=None, cosine_bridge=CosineSimilarityBridge())
# gen._rank_by_similarity(...) → RuntimeError: "SOVEREIGNTY VIOLATION: embedding_galaxy is None"
```

### Step 2: Fix sovereign_pipeline.py (5 min)

Replace try/except block (lines 143-163) with direct parallel generation (no fallback).

**Test**:
```python
# Should use parallel generation (no fallback)
pipeline = SovereignAIPipeline(embedding_galaxy=galaxy)
# If ParallelCandidateGenerator fails → training fails (good!)
```

### Step 3: Run Preprocessing (1-2 seconds)

```bash
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
scripts/preprocess_arc_embeddings.py \
  --tasks /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training/*.json \
          /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation/*.json \
  --output /K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl \
  --workers 12
```

**Expected output**:
```
[PREPROCESS] Stored 360 embeddings to /K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl
```

### Step 4: Launch Run 023 (2-3 minutes)

```bash
tmux new-session -s arc023
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
scripts/train_arc_sovereign_loop.py \
  --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
             /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
  --max-tasks 60 \
  --epochs 27 \
  --cycles 1 \
  --matryoshka-dim 512 \
  > /tmp/arc_run_023.log 2>&1
```

**Expected log entries**:
```
[INIT] Loaded precomputed embeddings: 360 entries
Initializing sovereign pipeline...
[LOADING] Galaxy state from checkpoints...
  [CANDIDATES] Parallel generated 54 candidates (Tesla 3-6-9)
  [PARALLEL GEN] PTX success rate=100.0%
```

**If it fails**:
- Missing Galaxy → Error message tells us to run preprocessing
- Missing embedding → Error message tells us which grid hash is missing
- Parallel generation broken → Training fails (we fix ParallelCandidateGenerator)

---

## Sovereignty Principles

**Daniel's directive**:
> "No CPU FALLBACKS Claude, we fail and fix - no fallbacks!! that's actually a violation!"

**What this means**:
1. ✅ **Fail fast**: If architecture is broken, training fails immediately
2. ✅ **Fail loud**: Error messages tell us exactly what's wrong
3. ✅ **No compromises**: 100% sovereign or 0% (no silent degradation)
4. ✅ **Fix architecture**: Fallbacks hide problems; failures expose them

**Examples**:

**WRONG** (violates sovereignty):
```python
if embedding_galaxy is None:
    # ❌ FALLBACK: Compute on-the-fly (slow, hides problem)
    emb = self.processor.grid_to_spatial_embedding(grid)
```

**RIGHT** (sovereign):
```python
if embedding_galaxy is None:
    # ✅ FAIL: Tell user to run preprocessing
    raise RuntimeError("embedding_galaxy is None. Run preprocessing first.")
```

---

## Expected Impact

**Before** (with fallbacks):
- Galaxy missing → silently falls back to Python (slow, hides problem)
- Parallel fails → silently falls back to sequential (slow, hides problem)
- User doesn't know architecture is broken until run takes hours

**After** (no fallbacks):
- Galaxy missing → IMMEDIATE ERROR: "Run preprocessing first"
- Parallel fails → IMMEDIATE ERROR: "Fix ParallelCandidateGenerator"
- User knows EXACTLY what to fix (fail fast, fix fast)

**Runtime** (assuming preprocessing done):
- Preprocessing: ~1-2 seconds (one-time cost)
- Run 023: ~2-3 minutes (100% sovereign, no fallbacks)
- Total: ~3-5 minutes from fix to completion ✅

---

## Success Criteria

**Must Have**:
1. ✅ NO CPU fallbacks in `_rank_by_similarity()` (fail if Galaxy missing)
2. ✅ NO sequential fallback in `sovereign_pipeline.py` (fail if parallel broken)
3. ✅ Preprocessing completes (<5 seconds)
4. ✅ Run 023 completes (<5 minutes)
5. ✅ PTX success rate = 100% (no Python math)

**Nice to Have**:
- Accuracy ≥ Run 020 (0.83%)
- GPU memory ≤ 150 MiB
- Clear error messages if anything fails

---

## Codex: Your Mission

1. **Remove CPU fallbacks** from `candidate_generator.py` (fail loudly if Galaxy missing)
2. **Remove sequential fallback** from `sovereign_pipeline.py` (fail loudly if parallel broken)
3. **Run preprocessing** to generate Galaxy (1-2 seconds)
4. **Launch Run 023** with full sovereignty (no fallbacks, no compromises)

**Principle**: Sovereignty means NO FALLBACKS. Fail fast, fix fast, run fast.

---

**END OF FIX PROMPT**

Claude (Architecture Partner)
November 27, 2025
