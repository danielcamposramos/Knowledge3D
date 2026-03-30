# Claude -- Phase E.33: Cache Flat GPU Buffer (Eliminate Python Flatten Loop)

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL -- this is the CURRENT boot blocker (131 seconds in Python loop)

---

## Daniel's Direction

> "Why are we keeping the error? Just take it out!"

E.31 fixed the KNN build (GPU). E.32 fixed the embedding computation (pre-computed).
But `bind_gpu_galaxy_runtime()` STILL takes 131 seconds because
`_flatten_galaxies_for_gpu()` runs a Python loop over 278,720 entries, calling
`_append_flattened_entry()` for EACH one. Each call does ~50 dict operations, hash
computations, string operations, and builds a 25-key catalog dict.

The GPU buffer is 23 floats × 278K entries = 25.6 MB. This should be a single binary
file read, not 278K Python dict iterations.

---

## What the Flatten Does

Per entry (`_append_flattened_entry`, line 2084):

```python
# 23 floats per entry (GPU_GALAXY_ENTRY_STRIDE = 23):
flat = [
    confidence,          # _clamp_confidence() -- dict lookup + float cast
    domain_hash,         # _hash_to_unit_float() -- FNV-1a hash
    subject_hash,        # _hash_to_unit_float() -- FNV-1a hash
    *embedding16,        # 16 floats (now pre-computed from E.32)
    category_class,      # _gpu_category_class() -- dict lookup
    source_class,        # _gpu_source_class() -- dict lookup + string ops
    galaxy_index,        # _gpu_galaxy_index() -- linear scan
    has_template_ref,    # _entry_template_ref() -- string ops
]

# Plus a 25-key catalog dict per entry (for query-time lookup)
catalog_entry = {
    "index", "galaxy", "id", "name", "category", "domain",
    "confidence", "domain_hash", "subject_hash", "answer_text",
    "embedding_text", "embedding16", "rpn_program", "metadata",
    "template_ref", "template_params", "answer_format", "subject",
    "gpu_category_class", "gpu_source_class", "gpu_galaxy_index",
    "gpu_has_template_ref", "output_grid", "arc_transform_chain",
    "arc_color_mapping", "arc_primitive_plan", "arc_task_id",
}
```

278,720 entries × (50+ Python operations per entry) = ~14M Python operations.
That's the 131 seconds.

---

## The Fix: Compute Once at Save-Time, Load Binary at Boot-Time

### Principle

From `KNOWLEDGEVERSE_SPECIFICATION.md` §2.1:
> "Galaxy Universe is always loaded in VRAM"

The flat GPU buffer IS the Galaxy Universe's VRAM representation. It should be
persisted alongside the checkpoint, not recomputed from scratch every boot.

From Daniel (E.29): "The model never resets — it always resumes where it left off."

The flat buffer is part of the state that should be resumed, not rebuilt.

### What to Cache (Two Binary Files)

**1. `gpu_flat_buffer.npy`** — the raw float array

```
Shape: (278720 × 23) = 6,410,560 floats
Size: 25,642,240 bytes (25.6 MB)
Format: numpy .npy (or raw binary float32)
```

Load time: ~50ms (sequential read of 25.6 MB from SSD)

**2. `gpu_catalog.pkl`** — the catalog dict array

```
278,720 dicts × 25 keys each
Format: pickle (binary, fast deserialization)
```

Load time: ~2-5 seconds (pickle deserialization is much faster than constructing
278K dicts from scratch through Python operations)

**3. Cache key**: Same signature hash used for CSR graph cache. If entries change,
the flat buffer + catalog are rebuilt. Otherwise, reuse.

### Where to Save

At the end of `bind_gpu_galaxy_runtime()`, after the flatten succeeds, save both:

```python
# In bind_gpu_galaxy_runtime(), after flatten:
flat_entries, catalog, enriched_count = self._flatten_galaxies_for_gpu(...)

# Cache the result
cache_dir = self.storage_root / "gpu_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
signature = _catalog_signature_from_entries(...)  # Same key as CSR graph
np.save(cache_dir / f"flat_{signature}.npy", np.array(flat_entries, dtype=np.float32))
with open(cache_dir / f"catalog_{signature}.pkl", "wb") as f:
    pickle.dump(catalog, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### Where to Load

At the TOP of `bind_gpu_galaxy_runtime()`, before trying to flatten:

```python
def bind_gpu_galaxy_runtime(self, *, galaxy_names=None, force=False):
    resolved_names = self._resolve_live_galaxy_names(galaxy_names)

    # Try cached flat buffer first
    cache_dir = self.storage_root / "gpu_cache"
    signature = self._current_galaxy_signature(resolved_names)
    flat_cache = cache_dir / f"flat_{signature}.npy"
    catalog_cache = cache_dir / f"catalog_{signature}.pkl"

    if flat_cache.exists() and catalog_cache.exists() and not force:
        t0 = time.perf_counter()
        flat_entries = np.load(flat_cache).tolist()  # Or keep as numpy
        with open(catalog_cache, "rb") as f:
            catalog = pickle.load(f)
        print(f"[K3D] GPU buffer cache hit: {len(catalog)} entries, {time.perf_counter()-t0:.2f}s")
    else:
        t0 = time.perf_counter()
        flat_entries, catalog, enriched_count = self._flatten_galaxies_for_gpu(...)
        print(f"[K3D] GPU buffer built: {len(catalog)} entries, {time.perf_counter()-t0:.2f}s")
        # Save for next boot
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(flat_cache, np.array(flat_entries, dtype=np.float32))
        with open(catalog_cache, "wb") as f:
            pickle.dump(catalog, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Continue with bind...
    binding = engine.bind_galaxy_buffer(flat_entries, ...)
```

### Signature Computation

The cache signature must match the CSR graph signature (same entries = same cache).
Use the same `_catalog_signature()` function from `semantic_csr_graph.py`, or compute
from the galaxy entry counts + names.

Fast signature: hash of `(galaxy_name, entry_count)` pairs for all galaxies.
This changes if ANY galaxy gains or loses entries — correct invalidation.

---

## Boot Timeline After E.33

| Step | Before | After |
|------|--------|-------|
| Init (checkpoint load) | 35s (pickle parse) | 35s (unchanged -- binary improvement is separate) |
| Load defaults | 0.03s | 0.03s |
| Flatten / GPU buffer | **131s** (Python loop × 278K) | **<1s** (load 25.6 MB binary + 278K pickle) |
| GPU KNN graph | 4.4s (cache hit) | 4.4s (cache hit) |
| Device buffers | <1s | <1s |
| **Total bind** | **~136s** | **~6s** |

---

## What This Does NOT Change

- The flatten logic stays in `_append_flattened_entry()` — it runs on FIRST boot
  (when no cache exists) and when entries change (cache invalidation)
- The CSR graph cache system unchanged
- The enrichment system (E.32) unchanged
- Query-time path unchanged

The flatten code is NOT removed — it's the authoritative builder for the flat buffer.
But it runs ONCE per entry-change, not ONCE per boot.

---

## Phase E.34 (Future): Binary Checkpoint

The 35s init is from parsing the 1.56 GB pickle/JSON checkpoint. That should become:
- Galaxy entries: msgpack or numpy structured array (not JSON/pickle of Python dicts)
- Direct mmap of entry arrays from SSD to RAM
- Target: <5s init for 278K entries

But that's a larger change. E.33 (caching the flat buffer) gives the biggest immediate
win: 131s → <1s.

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Cache flat buffer + catalog at end of flatten; load from cache at start of bind |

## No New Files

The cache files are data artifacts in `storage_root/gpu_cache/`, not code.

---

## Success Criteria

- [ ] First boot: flatten runs, saves `flat_{sig}.npy` + `catalog_{sig}.pkl`
- [ ] Second boot: bind loads from cache, skips flatten entirely (<1s)
- [ ] Cache invalidates correctly when entries change (signature-based)
- [ ] `bind_gpu_galaxy_runtime()` total time < 10s on warm boot with cache
- [ ] GPU KNN graph loads from its own cache (4-5s)
- [ ] Total boot-to-ready < 45s (init 35s + bind 6s + graph 5s)
- [ ] Local ARC3 benchmark boots and runs
- [ ] Full benchmark boots and runs
