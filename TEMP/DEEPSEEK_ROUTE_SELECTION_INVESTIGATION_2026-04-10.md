## DEEP INVESTIGATION RESULTS

Based on analysis of the codebase and architecture, **Hypothesis 1 ("Invalidation not firing") is the most likely root cause**, with strong evidence pointing to a broken invalidation chain.

## HOW GPU TASK DISPATCH SELECTS THE WINNER STAR

The `GPUTaskDispatch.launch()` mechanism selects the winner star through a **dual-mechanism system**:

1. **Embedding Similarity Search**: A transformer-based attention mechanism computes cosine similarity between:
   - Query embedding (from perception text)
   - Star embeddings in VRAM table
   - Route family filtering via `route_family_mask`

2. **Confidence Weighting**: Similarity scores are multiplied by star confidence values, making higher-confidence stars (0.7) dominate over low-confidence avoidance stars (0.03-0.12)

3. **Winner Index Resolution**: GPU returns `winner_star_index` → mapped to `answer_index` via star's `meta_refs.action_index`

## EVIDENCE FOR HYPOTHESIS 1 (INVALIDATION FAILURE)

### Critical Code Defect Found

In `/knowledgeverse/knowledgeverse.py`, the invalidation chain is broken:

```python
def invalidate_gpu_galaxy_binding(self) -> None:
    """Invalidate the GPU galaxy binding, forcing a rebuild."""
    if self._sovereign_hot_path:
        self._sovereign_hot_path.runtime.invalidate_loaded_state()  # BUG!
```

**The `SovereignHotPath` class has NO `runtime` attribute!** This causes:
1. AttributeError exception (likely caught and ignored)
2. `invalidate_loaded_state()` never called
3. `star_table.star_count` stays at non-zero value
4. `ensure_loaded()` skips rebuild thinking VRAM is current

### Supporting Evidence

1. **Observation matches prediction**: Galaxy grows in Python (511→612 stars) but GPU dispatch sees stale table
2. **The symptom**: Always returns same `answer_index` for ACTION2 because VRAM table frozen at original 511 stars
3. **Alternative hypotheses ruled out**:
   - Not Hypothesis 2: Disk persistence is enabled (`persist=True`), entries are saved
   - Not Hypothesis 3: Embedding would vary with different stars present
   - Not Hypothesis 4: Confidence weighting alone wouldn't explain complete absence of alternative selection
   - Not Hypothesis 5: answer_index mapping is star-dependent; avoidance stars have different action_index
   - Not Hypothesis 6: route_family filtering works; avoidance stars tagged with GAME_2D

## THE INVALIDATION CHAIN BREAKAGE

### Correct chain should be:
```
GalaxyManager.upsert_entry()
  → knowledgeverse.invalidate_gpu_galaxy_binding()
    → sovereign_hot_path.invalidate_loaded_state()  # Direct call!
      → star_table.star_count = 0
      → _catalog_signature = ""
```

### Actual broken chain:
```
GalaxyManager.upsert_entry()
  → knowledgeverse.invalidate_gpu_galaxy_binding()
    → sovereign_hot_path.runtime.invalidate_loaded_state()  # AttributeError!
      → Exception caught/silenced
      → star_table.star_count unchanged
      → ensure_loaded() sees non-zero count, skips rebuild
```

## CONFIRMATION TEST

To confirm, add debug logging before/after the suspect line:
```python
def invalidate_gpu_galaxy_binding(self) -> None:
    print(f"DEBUG: Invalidating GPU binding, hot_path={self._sovereign_hot_path}")
    if self._sovereign_hot_path:
        print(f"DEBUG: hot_path attributes: {dir(self._sovereign_hot_path)}")
        # This will fail:
        self._sovereign_hot_path.runtime.invalidate_loaded_state()
```

## IMMEDIATE FIX

Change the line in `/knowledgeverse/knowledgeverse.py`:

```python
# FROM:
self._sovereign_hot_path.runtime.invalidate_loaded_state()

# TO:
self._sovereign_hot_path.invalidate_loaded_state()
```

This matches the actual method signature in `SovereignHotPath` class.

## SECONDARY ISSUE: CONFIDENCE WEIGHTING

Even after fixing invalidation, avoidance stars (confidence 0.03-0.12) may still lose to game mechanics stars (confidence 0.7). This requires **confidence calibration**