# Training Bottleneck Analysis

**Date**: November 26, 2025
**Status**: 🔴 CRITICAL ISSUES IDENTIFIED
**Result**: 0% accuracy after 10,206 attempts

---

## Problems Found

### 1. ⚠️ **Generation Constraints TOO TIGHT**

**Location**: `candidate_generator.py`

```python
# CURRENT (TOO RESTRICTIVE):
def __init__(self, matryoshka_dim: int = 512, max_candidates: int = 69):
    # Only 69 candidates max!

def generate_candidates(...):
    for example in train_examples[:3]:  # Only uses 3 examples!
        inferred = self._infer_from_example(example, input_grid)
```

**Problem**:
- Max 69 candidates per task
- Only 3 train examples used
- Limited primitive search (rotations, flips, translations only)
- Gets capped further to 12 by pipeline!

---

### 2. ⚠️ **top_k VALUES TOO SMALL**

**Locations**:
- `sovereign_trm_router.py:116` → `top_k=3` (default)
- `sovereign_pipeline.py:64` → `top_k=12` (pipeline)
- `sovereign_trm_router.py:127` → `top_k*2=6` (semantic matches)

**Current Flow**:
```
CandidateGenerator: up to 69 candidates
    ↓ (capped)
Pipeline top_k: 12 candidates selected
    ↓
Router top_k: 3 grammar rules
    ↓
Semantic matches: 6 contexts
```

**Problem**: With only 12-69 total attempts per task, chances of finding correct solution are ~0%!

---

### 3. 🔴 **SHADOW COPY NOT FEEDING BACK TO WEIGHTS**

**Location**: `dual_shadow_copy.py` + `sovereign_trm_router.py`

**What's Working**:
```python
# dual_shadow_copy.py:80-84
if self.staged:
    self.library.append(entry)  # ✅ Recording discoveries
    self._pending.append(entry)
else:
    self.library.append(entry)  # ✅ Recording discoveries
    self._commit_entry(entry)   # ✅ Committing to galaxies
```

**What's MISSING**:
```python
# sovereign_trm_router.py:53
self.base_trm = MatryoshkaTRM(max_dims=self.matryoshka_dim, min_dims=64)
# ❌ TRM weights are NEVER updated with discovered patterns!
# ❌ Shadow copy discoveries are isolated from routing logic!
```

**Impact**: The model learns patterns but doesn't USE them for routing!

---

### 4. ⚠️ **SEMANTIC LAYER PARTIALLY WIRED**

**What's Working**:
- ✅ Semantic context records word refs (line 64-66 in dual_shadow_copy.py)
- ✅ Semantic matching happens (line 124-141 in sovereign_trm_router.py)
- ✅ Words are stored with symlink pattern

**What's INCOMPLETE**:
- ⚠️ Only `top_k*2 = 6` semantic matches retrieved
- ⚠️ Words map intents but don't GUIDE generation
- ⚠️ Vocabulary isn't used to expand search space

**Current Semantic Flow**:
```
Task → SemanticSignature.extract(grid)
    → SemanticContext.find_matching_contexts(grid, top_k=6)
    → Returns 6 matches with word refs
    → Words resolve to meanings
    → BUT: Meanings don't instruct new candidate generation!
```

---

## Architectural Gaps

### Gap 1: Discovery → Routing Feedback Loop MISSING

```
┌─────────────────┐
│ Discoveries     │
│ (Shadow Copy)   │  ← Records patterns
└────────┬────────┘
         │
         │ ❌ NOT CONNECTED!
         │
         ↓
┌─────────────────┐
│ TRM Routing     │  ← Uses static weights
│ (Router)        │
└─────────────────┘
```

**Should Be**:
```
┌─────────────────┐
│ Discoveries     │
│ (Shadow Copy)   │
└────────┬────────┘
         │
         │ ✅ UPDATE WEIGHTS!
         ↓
┌─────────────────┐
│ TRM Routing     │  ← Learns from discoveries
│ (Router)        │
└─────────────────┘
```

---

### Gap 2: Semantic Words → Generation Instruction MISSING

```
Words: "rotation_task", "sparse_grid", "color_change_task"
    ↓
Meanings: Known and stored
    ↓
❌ NOT USED to generate new candidates!
```

**Should Be**:
```
Words: "rotation_task" detected
    ↓
Meanings: "Task involves rotation"
    ↓
✅ GENERATE: More rotation variants (45°, 135°, etc.)
✅ GENERATE: Combined rotation + recolor
✅ GENERATE: Multi-step rotation sequences
```

---

## Immediate Fixes Needed

### Fix 1: Increase Generation Diversity

**File**: `candidate_generator.py`

```python
# CHANGE:
def __init__(self, matryoshka_dim: int = 512, max_candidates: int = 369):  # ← 369 (Tesla 3-6-9)

def generate_candidates(...):
    for example in train_examples[:9]:  # ← Use 9 examples (Tesla 3-6-9)
```

---

### Fix 2: Increase top_k Values

**Files**: `sovereign_pipeline.py`, `sovereign_trm_router.py`

```python
# sovereign_pipeline.py:64
top_k: int = 69,  # ← Increase from 12 to 69

# sovereign_trm_router.py:116
def route(..., top_k: int = 27):  # ← Increase from 3 to 27

# sovereign_trm_router.py:127
matches = self.semantic_context.find_matching_contexts(grid, top_k=top_k * 3)  # ← 81 matches
```

---

### Fix 3: Wire Shadow Copy → TRM Weights

**File**: `sovereign_trm_router.py`

Add method to update routing based on discoveries:

```python
def update_from_discoveries(self, shadow_copy):
    """Update router weights from discovered patterns."""
    for entry in shadow_copy.library[-100:]:  # Last 100 discoveries
        program = entry["program"]
        quality = entry["quality_score"]
        # TODO: Update self.base_trm weights
        # TODO: Update self.router_adapter with pattern
```

Call this after each epoch!

---

### Fix 4: Wire Semantic Words → Generation

**File**: `candidate_generator.py`

Add semantic-guided generation:

```python
def _generate_semantic_guided_candidates(self, grid, semantic_words):
    """Generate candidates based on semantic word hints."""
    candidates = []

    for word in semantic_words:
        if "rotation" in word:
            # Generate MORE rotation variants
            for angle in [45, 90, 135, 180, 225, 270, 315]:
                candidates.append(self._rotate(grid, angle))

        if "sparse" in word:
            # Generate fill patterns
            candidates.extend(self._generate_fill_patterns(grid))

        if "color_change" in word:
            # Generate recoloring variants
            candidates.extend(self._generate_recolor_variants(grid))

    return candidates
```

---

## Root Cause Summary

**Why 0% accuracy?**

1. **Too few candidates** (12-69) to explore solution space
2. **No learning feedback** from discoveries to routing
3. **Semantic layer observes but doesn't act** on word meanings
4. **Generation too conservative** (only basic primitives)

**With 10,206 attempts × 12 candidates = 122,472 total programs tried**
**But**: Same 12 primitives repeated, no diversity!

---

## Next Steps

1. ✅ Increase max_candidates: 69 → 369 (Tesla 3-6-9)
2. ✅ Increase top_k: 3 → 27, 12 → 69
3. ✅ Wire shadow_copy discoveries → TRM weight updates
4. ✅ Wire semantic words → guided generation
5. ✅ Run new training: expect 5-15% accuracy initially

**Target**: 99% accuracy requires ALL 4 fixes deployed!
