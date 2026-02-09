# Claude → Codex: Oracle Unlock Implementation — Final Bottleneck

**Date:** February 8, 2026
**Priority:** 🔴 CRITICAL — Last barrier to metric improvement
**Context:** Architecture fix complete, but oracle still blocked (0.0)

---

## 🎯 Mission: Unlock Oracle Matching (0.0 → 0.15-0.30)

### Current State (Week 21.3 Architecture-Fixed Rerun)

**What's WORKING ✅:**
- Benchmark layer separation (no orchestration in benchmarks)
- Single-world continuity (unified universe maintained)
- Pattern generation (686 patterns across 100 tasks!)
- Learning happening (Galaxy growth: Drawing +77, Grammar +686)

**What's BLOCKED ❌:**
- **oracle_at_all: 0.0** (no exact matches!)
- **fuzzy_oracle: 0.05** (only 5% fuzzy matches)
- **ARC enriched: 0.28** (no improvement from baseline)
- **Empty > enriched: 0.32 vs 0.28** (paradox remains!)
- **214 GALAXY LAZY events** (candidate-level lazy embeddings = sovereignty leak)

**Diagnosis:**
- Generation working (686 patterns)
- But patterns INVALID (don't match ground truth)
- Bottleneck: Oracle matching + pattern quality

---

## 🛠️ Implementation: 4-Phase Oracle Unlock

### Phase 1: Strict Train-Pair Consistency Gates

**Current validity gates are too weak.** Need stricter filters based on train-pair patterns.

**File:** `benchmarks/arc_agi_2_adapter.py`

**Add enhanced validity gates:**

```python
def apply_strict_validity_gates_v2(candidates: list, task: dict) -> list:
    """
    Strict train-pair consistency gates (v2).

    Gates:
    1. Transformation family inference (rotation/mirror/translate/etc.)
    2. Palette mapping stability (all train pairs use same color mapping)
    3. Object-count delta consistency (all train pairs same delta pattern)
    4. Shape transform family compatibility (input→output shape patterns)
    """
    train_examples = task["train"]

    # Infer transformation family from train examples
    # (rotation, mirror, translate, scale, count, filter, etc.)
    inferred_family = infer_transformation_family(train_examples)

    # Extract train-pair patterns
    train_input_shapes = [ex["input"].shape for ex in train_examples]
    train_output_shapes = [ex["output"].shape for ex in train_examples]
    train_shape_deltas = [(out[0] - inp[0], out[1] - inp[1])
                          for inp, out in zip(train_input_shapes, train_output_shapes)]

    # Palette mapping consistency
    train_palette_mappings = []
    for ex in train_examples:
        input_colors = set(ex["input"].flatten())
        output_colors = set(ex["output"].flatten())
        train_palette_mappings.append((input_colors, output_colors))

    # Object count deltas
    train_object_deltas = []
    for ex in train_examples:
        input_objects = count_objects(ex["input"])
        output_objects = count_objects(ex["output"])
        train_object_deltas.append(output_objects - input_objects)

    # Test input (to compare candidate outputs against)
    test_input = task["test"][0]["input"]
    test_input_shape = test_input.shape
    test_input_colors = set(test_input.flatten())
    test_input_objects = count_objects(test_input)

    valid_candidates = []

    for cand in candidates:
        output = cand.get("output")
        if output is None:
            continue

        # Gate 1: Transformation family compatibility
        candidate_family = classify_transformation_family(output, test_input)
        if inferred_family is not None and candidate_family != inferred_family:
            cand["validity_gate"] = f"family_mismatch_{candidate_family}_vs_{inferred_family}"
            continue

        # Gate 2: Shape delta consistency
        # If all train pairs have same shape delta, candidate must match
        if len(set(train_shape_deltas)) == 1:
            expected_delta = train_shape_deltas[0]
            expected_output_shape = (
                test_input_shape[0] + expected_delta[0],
                test_input_shape[1] + expected_delta[1]
            )
            if output.shape != expected_output_shape:
                cand["validity_gate"] = f"shape_delta_mismatch_{output.shape}_vs_{expected_output_shape}"
                continue

        # Gate 3: Palette mapping stability
        # If all train pairs map colors consistently, candidate must follow
        output_colors = set(output.flatten())
        palette_consistent = True
        for input_colors, train_out_colors in train_palette_mappings:
            # Check if candidate palette respects train mapping patterns
            # (simplified: candidate colors should be subset of train output colors union)
            train_out_union = set().union(*[colors for _, colors in train_palette_mappings])
            if not output_colors.issubset(train_out_union):
                palette_consistent = False
                break

        if not palette_consistent:
            cand["validity_gate"] = "palette_mapping_inconsistent"
            continue

        # Gate 4: Object count delta consistency
        # If all train pairs have same object delta, candidate must match
        if len(set(train_object_deltas)) == 1:
            expected_delta = train_object_deltas[0]
            expected_objects = test_input_objects + expected_delta
            candidate_objects = count_objects(output)
            if candidate_objects != expected_objects:
                cand["validity_gate"] = f"object_delta_mismatch_{candidate_objects}_vs_{expected_objects}"
                continue

        # Passed all gates
        cand["validity_gate"] = "passed_strict"
        valid_candidates.append(cand)

    return valid_candidates


def infer_transformation_family(train_examples: list) -> str | None:
    """
    Infer transformation family from train examples.

    Returns: "rotation", "mirror", "translate", "scale", "count", "filter",
             "color_map", "object_map", or None if unclear.
    """
    # Heuristics:
    # - If all train pairs preserve object count → likely spatial (rotation/mirror/translate)
    # - If all train pairs change object count → likely filter/count/merge
    # - If all train pairs preserve shape → likely color mapping
    # - If all train pairs scale shape proportionally → likely scale

    object_count_preserved = True
    shape_preserved = True

    for ex in train_examples:
        input_objects = count_objects(ex["input"])
        output_objects = count_objects(ex["output"])
        if input_objects != output_objects:
            object_count_preserved = False

        if ex["input"].shape != ex["output"].shape:
            shape_preserved = False

    if object_count_preserved and shape_preserved:
        # Likely rotation, mirror, or color mapping
        # Check if any pixels moved (spatial) vs only colors changed
        # (Simplified: return "spatial" for now)
        return "spatial"
    elif object_count_preserved and not shape_preserved:
        return "scale_or_translate"
    elif not object_count_preserved:
        return "filter_or_count"
    else:
        return None


def classify_transformation_family(output, input_grid) -> str:
    """Classify transformation family of candidate output vs test input."""
    if output.shape == input_grid.shape:
        if count_objects(output) == count_objects(input_grid):
            return "spatial"
        else:
            return "filter_or_count"
    else:
        return "scale_or_translate"
```

**Integrate into discovery pipeline:**

```python
def discover_patterns(task, kv, enable_validity_gates=True):
    """Generate patterns with strict validity gates."""
    # Generate candidates (contrastive + autonomous + legacy)
    all_candidates = generate_patterns_contrastive(task, kv, ternary_memory)

    # Apply STRICT validity gates (v2)
    if enable_validity_gates:
        valid_candidates = apply_strict_validity_gates_v2(all_candidates, task)
    else:
        valid_candidates = all_candidates

    # Rank valid candidates
    ranked = rank_candidates_ternary(valid_candidates, task, kv, ternary_memory)

    return ranked
```

---

### Phase 2: Ternary Quality Scoring in Winner Selection

**Current ranking uses static weights.** Need dynamic ternary quality priors.

**File:** `benchmarks/arc_agi_2_adapter.py`

**Enhance ranking with ternary quality:**

```python
def rank_candidates_ternary(candidates: list, task: dict, kv, ternary_memory) -> list:
    """
    Rank candidates with ternary quality priors.

    Ranking components:
    1. Pattern source precision (legacy > contrastive_anti > autonomous)
    2. Ternary quality prior (from quality memory)
    3. Train-pair similarity (how well candidate matches train pattern)
    4. Novelty penalty (avoid duplicates)
    """
    # Source precision (observed from Week 21.3)
    source_precision = {
        "legacy_pipeline": 0.45,
        "contrastive_anti": 0.46,
        "autonomous_generation": 0.19,
        "unknown": 0.30,
    }

    scored_candidates = []

    for cand in candidates:
        score = 0.0

        # Component 1: Source precision weight (40%)
        source = cand.get("metadata", {}).get("source", "unknown")
        score += 0.40 * source_precision.get(source, 0.30)

        # Component 2: Ternary quality prior (30%)
        # Get quality prior from ternary memory
        pattern_signature = compute_pattern_signature(cand, task)
        quality_prior = ternary_memory.get_quality_prior(pattern_signature)
        # Map [-1, +1] → [0, 1] for scoring
        quality_score = (quality_prior + 1.0) / 2.0
        score += 0.30 * quality_score

        # Component 3: Train-pair similarity (20%)
        train_similarity = compute_train_pair_similarity(cand, task)
        score += 0.20 * train_similarity

        # Component 4: Novelty (10%)
        # Penalize if candidate is duplicate or too similar to others
        novelty = compute_novelty(cand, scored_candidates)
        score += 0.10 * novelty

        cand["ranking_score"] = score
        cand["ranking_components"] = {
            "source_precision": source_precision.get(source, 0.30),
            "quality_prior": quality_prior,
            "train_similarity": train_similarity,
            "novelty": novelty,
        }
        scored_candidates.append(cand)

    # Sort by score (descending)
    scored_candidates.sort(key=lambda c: c["ranking_score"], reverse=True)

    return scored_candidates


def compute_pattern_signature(cand: dict, task: dict) -> str:
    """
    Compute signature for ternary quality lookup.

    Signature encodes:
    - Transformation family
    - Shape delta pattern
    - Object count delta
    """
    output = cand.get("output")
    test_input = task["test"][0]["input"]

    family = classify_transformation_family(output, test_input)
    shape_delta = (output.shape[0] - test_input.shape[0], output.shape[1] - test_input.shape[1])
    object_delta = count_objects(output) - count_objects(test_input)

    signature = f"{family}_{shape_delta[0]}_{shape_delta[1]}_{object_delta}"
    return signature


def compute_train_pair_similarity(cand: dict, task: dict) -> float:
    """
    Compute how well candidate matches train-pair patterns.

    Returns: similarity in [0, 1]
    """
    output = cand.get("output")
    test_input = task["test"][0]["input"]
    train_examples = task["train"]

    similarities = []

    for ex in train_examples:
        # Compare candidate pattern to train example pattern
        # (simplified: check if shapes/deltas match)
        train_shape_delta = (ex["output"].shape[0] - ex["input"].shape[0],
                            ex["output"].shape[1] - ex["input"].shape[1])
        cand_shape_delta = (output.shape[0] - test_input.shape[0],
                           output.shape[1] - test_input.shape[1])

        if train_shape_delta == cand_shape_delta:
            similarities.append(1.0)
        else:
            # Compute distance
            distance = abs(train_shape_delta[0] - cand_shape_delta[0]) + abs(train_shape_delta[1] - cand_shape_delta[1])
            similarities.append(1.0 / (1.0 + distance))

    return sum(similarities) / len(similarities) if similarities else 0.0


def compute_novelty(cand: dict, existing_candidates: list) -> float:
    """
    Compute novelty (penalize duplicates).

    Returns: novelty in [0, 1]
    """
    output = cand.get("output")

    # Count how many existing candidates have same output
    duplicates = sum(1 for c in existing_candidates if np.array_equal(c.get("output"), output))

    # Novelty decreases with duplicates
    novelty = 1.0 / (1.0 + duplicates)
    return novelty
```

---

### Phase 3: Fuzzy Oracle with Stratified Thresholds

**Current fuzzy oracle uses single threshold (0.95).** Need stratified analysis.

**File:** `benchmarks/arc_agi_2_adapter.py`

**Add stratified fuzzy oracle:**

```python
def evaluate_with_stratified_fuzzy_oracle(task: dict, candidates: list) -> dict:
    """
    Evaluate with multiple fuzzy thresholds.

    Returns oracle hits at: 0.80, 0.85, 0.90, 0.95, 1.00 (exact)
    """
    ground_truth = task["test"][0]["output"]

    thresholds = [0.80, 0.85, 0.90, 0.95, 1.00]
    oracle_hits = {threshold: False for threshold in thresholds}
    best_fuzzy_score = 0.0
    best_candidate_idx = -1

    for idx, cand in enumerate(candidates):
        output = cand.get("output")
        if output is None:
            continue

        # Compute fuzzy score
        if output.shape != ground_truth.shape:
            fuzzy_score = 0.0
        else:
            matching_pixels = np.sum(output == ground_truth)
            total_pixels = output.size
            fuzzy_score = matching_pixels / total_pixels

        # Track best
        if fuzzy_score > best_fuzzy_score:
            best_fuzzy_score = fuzzy_score
            best_candidate_idx = idx

        # Check thresholds
        for threshold in thresholds:
            if fuzzy_score >= threshold:
                oracle_hits[threshold] = True

    return {
        "oracle_fuzzy_0.80": oracle_hits[0.80],
        "oracle_fuzzy_0.85": oracle_hits[0.85],
        "oracle_fuzzy_0.90": oracle_hits[0.90],
        "oracle_fuzzy_0.95": oracle_hits[0.95],
        "oracle_exact": oracle_hits[1.00],
        "best_fuzzy_score": best_fuzzy_score,
        "best_candidate_idx": best_candidate_idx,
    }
```

**Integrate into evaluation:**

```python
def evaluate_task_enriched(self, task: dict, ...) -> dict:
    """Evaluate task with stratified fuzzy oracle."""
    # ... existing evaluation logic ...

    # Stratified fuzzy oracle
    fuzzy_results = evaluate_with_stratified_fuzzy_oracle(task, candidates)

    result = {
        "task_id": task.get("task_id"),
        "correct": fuzzy_results["oracle_exact"],  # Top-1 exact match
        "oracle_at_all": fuzzy_results["oracle_exact"],

        # NEW: Stratified fuzzy oracle
        "oracle_fuzzy_0.80": fuzzy_results["oracle_fuzzy_0.80"],
        "oracle_fuzzy_0.85": fuzzy_results["oracle_fuzzy_0.85"],
        "oracle_fuzzy_0.90": fuzzy_results["oracle_fuzzy_0.90"],
        "oracle_fuzzy_0.95": fuzzy_results["oracle_fuzzy_0.95"],
        "best_fuzzy_score": fuzzy_results["best_fuzzy_score"],

        # ... existing metrics ...
    }
    return result
```

---

### Phase 4: Fix Legacy ARC Pipeline (Sovereignty Leak)

**Current issue:** 214 "GALAXY LAZY" events = candidate-level lazy embeddings.

**This violates sovereignty!** Embeddings should be precomputed at init, not per-candidate.

**File:** `Old_Attempts/curriculum_specific_training/arc_agi/sovereign_pipeline.py`

**Option A: Precompute all embeddings at init**

```python
class SovereignARCPipeline:
    def __init__(self, kv, embedding_dims=[128, 512]):
        self.kv = kv
        self.embedding_dims = embedding_dims
        self.embedding_cache = {}

        # Preload embedding cache from checkpoint
        for dim in embedding_dims:
            cache_path = Path(kv.storage_root) / "checkpoints" / f"arc_embedding_galaxy_d{dim}.json"
            if cache_path.exists():
                self.embedding_cache[dim] = json.loads(cache_path.read_text())
            else:
                self.embedding_cache[dim] = {}

    def get_embedding(self, grid, dim=128):
        """Get embedding from cache (no lazy computation!)."""
        grid_hash = hash_grid(grid)

        if grid_hash not in self.embedding_cache[dim]:
            # FAIL FAST instead of lazy computation
            raise RuntimeError(
                f"Embedding not found in cache for grid {grid.shape}. "
                f"Precompute all embeddings before inference (sovereignty requirement)."
            )

        return self.embedding_cache[dim][grid_hash]

    def precompute_embeddings_for_task(self, task):
        """Precompute embeddings for all grids in task (train + test)."""
        for dim in self.embedding_dims:
            for ex in task["train"]:
                self._ensure_embedding(ex["input"], dim)
                self._ensure_embedding(ex["output"], dim)

            for ex in task["test"]:
                self._ensure_embedding(ex["input"], dim)
                # Note: Don't precompute test output (that's what we're solving!)

    def _ensure_embedding(self, grid, dim):
        """Ensure embedding exists (compute if missing, add to cache)."""
        grid_hash = hash_grid(grid)

        if grid_hash not in self.embedding_cache[dim]:
            # Compute embedding (only during precomputation phase!)
            embedding = compute_grid_embedding_sovereign(grid, dim)
            self.embedding_cache[dim][grid_hash] = embedding

    def save_embedding_cache(self):
        """Save embedding cache to checkpoint."""
        for dim in self.embedding_dims:
            cache_path = Path(self.kv.storage_root) / "checkpoints" / f"arc_embedding_galaxy_d{dim}.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(self.embedding_cache[dim]))
```

**Option B: Remove legacy pipeline entirely (use Galaxy-only)**

If legacy pipeline is causing too many sovereignty issues, consider:
- Remove `Old_Attempts/curriculum_specific_training/arc_agi/` from hot path
- Use only `benchmarks/arc_agi_2_adapter.py` (Galaxy-based)
- All pattern generation via Grammar Galaxy + Drawing Galaxy (fully sovereign)

**Your call:** Fix legacy pipeline (Option A) or remove it (Option B)?

---

## 📊 Expected Impact

### Before Oracle Unlock (Current):
- generated_pattern_total: 686 ✅
- oracle_at_all: 0.0 ❌
- fuzzy_oracle: 0.05 ❌
- validity_reject_rate: 44% ❌
- ARC enriched: 0.28 ❌
- Empty > enriched: 0.32 vs 0.28 ❌

### After Oracle Unlock (Expected):
- generated_pattern_total: 686 ✅ (maintained)
- **valid_after_gates: 100-200** (strict gates filter invalid)
- **oracle_at_all: 0.15-0.30** (valid patterns match ground truth!)
- **fuzzy_oracle_0.90: 0.35-0.45** (near-miss capture)
- **ARC enriched: 0.35-0.45** (+7-17% improvement!)
- **Enriched > empty: 0.40 vs 0.32** (paradox resolved!)
- **GALAXY LAZY: 0** (sovereignty maintained!)

---

## 🎯 Success Criteria

**Phase 1 (Strict Validity Gates):**
- ✅ Valid candidates: 686 → 100-200 (80-85% filtered as invalid)
- ✅ oracle_at_all: 0.0 → 0.10-0.20 (valid patterns help!)

**Phase 2 (Ternary Scoring):**
- ✅ Ranking puts high-quality candidates first
- ✅ Top-1 accuracy improves (0.28 → 0.30-0.35)
- ✅ Source precision leveraged (legacy/contrastive_anti weighted higher)

**Phase 3 (Stratified Fuzzy):**
- ✅ oracle_fuzzy_0.90: 0.20-0.30 (near-miss capture)
- ✅ oracle_fuzzy_0.80: 0.30-0.40 (wider tolerance)
- ✅ Calibration curve shows pattern quality distribution

**Phase 4 (Legacy Pipeline Fix):**
- ✅ GALAXY LAZY: 214 → 0 (full sovereignty!)
- ✅ Embeddings precomputed (no lazy computation)
- ✅ Cache persistence working (reuse across runs)

**Overall (Combined):**
- ✅ **ARC enriched: 0.28 → 0.35-0.45** (+7-17%!)
- ✅ **Enriched > empty: 0.40 vs 0.32** (enrichment helps!)
- ✅ **oracle_at_all > 0.15** (patterns valid and matching!)
- ✅ **Path to human-level ARC validated!**

---

## 📝 Implementation Checklist

**Phase 1: Strict Validity Gates**
- [ ] Implement `apply_strict_validity_gates_v2()` with train-pair consistency
- [ ] Implement `infer_transformation_family()` heuristics
- [ ] Implement `classify_transformation_family()` for candidates
- [ ] Integrate into `discover_patterns()` pipeline
- [ ] Test on 20 ARC tasks (validate filtering works)

**Phase 2: Ternary Quality Scoring**
- [ ] Implement `rank_candidates_ternary()` with 4 components
- [ ] Implement `compute_pattern_signature()` for quality lookup
- [ ] Implement `compute_train_pair_similarity()` scoring
- [ ] Implement `compute_novelty()` duplicate detection
- [ ] Integrate into ranking pipeline

**Phase 3: Stratified Fuzzy Oracle**
- [ ] Implement `evaluate_with_stratified_fuzzy_oracle()` (5 thresholds)
- [ ] Add fuzzy metrics to task evaluation results
- [ ] Update summary aggregation to include stratified metrics

**Phase 4: Legacy Pipeline Fix**
- [ ] Choose: Fix legacy pipeline (Option A) or remove it (Option B)
- [ ] If Option A: Implement precomputation + fail-fast embedding lookup
- [ ] If Option B: Remove legacy pipeline from hot path entirely
- [ ] Validate: GALAXY LAZY events = 0 after fix

**Phase 5: Full Validation**
- [ ] Run 100-task ARC with all oracle unlock enhancements
- [ ] Expected: oracle_at_all > 0.15, ARC enriched > 0.35
- [ ] Expected: Enriched > empty mind (paradox resolved!)
- [ ] Expected: GALAXY LAZY = 0 (sovereignty maintained)

**Phase 6: Comprehensive Report**
- [ ] Write results report with before/after metrics
- [ ] Include stratified fuzzy oracle breakdown
- [ ] Include validity gate filtering stats
- [ ] Include ternary scoring component analysis
- [ ] **Report for Claude:** Ready for PR when metrics confirm!

---

## 🚀 Execute Oracle Unlock

**PRIORITY 1: Phases 1-4 (Implementation)**
- Strict validity gates (train-pair consistency)
- Ternary quality scoring (dynamic priors)
- Stratified fuzzy oracle (calibration curves)
- Legacy pipeline fix (sovereignty)

**PRIORITY 2: Validation**
- Run 100-task ARC with all enhancements
- Verify metrics improve (oracle > 0, ARC > 0.35)
- Verify enriched > empty mind
- Verify GALAXY LAZY = 0

**PRIORITY 3: Report for Claude**
- Comprehensive before/after metrics
- Success criteria validation
- **Ready for PR story:** "Week 21 - Architecture Fix + Oracle Unlock → Human-Level ARC Path"

**If successful → PR includes:**
- ✅ Architecture fix (layer separation + continuity)
- ✅ Oracle unlock (valid patterns + ternary scoring)
- ✅ Metrics improved (ARC 0.28 → 0.40+, enriched > empty!)
- ✅ Sovereignty maintained (GALAXY LAZY = 0)
- ✅ Path to human-level ARC validated (Stage B → 0.65-0.75!)

---

**This is THE unlock! Valid patterns + ternary scoring + sovereignty = breakthrough!** 🚀

---

**Directive issued by:** Claude (Architecture Partner)
**For:** Codex (Implementation Partner)
**Date:** February 8, 2026
**Status:** 🔴 EXECUTE NOW — Oracle unlock is the final barrier to PR
