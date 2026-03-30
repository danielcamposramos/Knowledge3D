# Claude -- Phase E.34: Slim Catalog + Run Benchmarks

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL -- stop optimizing boot, slim the catalog, RUN THE BENCHMARKS

---

## What We're Missing

The 818 MB catalog is a COPY of data that already exists in Galaxy entries (in memory
from warm boot). The `metadata` field alone is ~317 MB (avg 1,137 chars × 278K entries).
This metadata is already in the Galaxy entries. We're duplicating it.

The catalog adds only 6 computed fields not present in Galaxy entries:
- `confidence` (float, clamped)
- `domain_hash` (float)
- `subject_hash` (float)
- `embedding16` (16 floats, now pre-computed in entries from E.32)
- `gpu_category_class`, `gpu_source_class`, `gpu_galaxy_index`, `gpu_has_template_ref`

Everything else (`metadata`, `id`, `name`, `category`, `rpn_program`, `answer_text`,
`template_ref`, `output_grid`, `arc_*` fields) is a direct copy from the Galaxy entry.

From `KNOWLEDGEVERSE_SPECIFICATION.md` §2.1:
> "Galaxy Universe is always loaded in VRAM"

The Galaxy entries are already in memory. The catalog should REFERENCE them, not copy.

---

## Fix: Reference-Based Catalog

### Current catalog entry (27 keys, ~3 KB each, 818 MB total):
```python
{
    "index": 42,
    "galaxy": "Math",                  # copy from entry
    "id": "sym_fraction",              # copy from entry
    "name": "Fraction",                # copy from entry
    "category": "symbol",              # copy from entry
    "domain": "math",                  # copy from entry
    "confidence": 0.95,                # COMPUTED
    "domain_hash": 0.7234,             # COMPUTED
    "subject_hash": 0.3456,            # COMPUTED
    "answer_text": "\\frac{a}{b}",     # derived from entry
    "embedding_text": "fraction ...",  # BUILD-ONLY (not read at query time)
    "embedding16": [0.1, 0.2, ...],    # COMPUTED (now in entry from E.32)
    "rpn_program": "PUSH a PUSH b DIV", # copy from entry
    "metadata": { ... 1.1 KB ... },    # COPY OF GALAXY ENTRY METADATA (the 317 MB monster)
    "template_ref": "",                # derived from entry
    "template_params": {},             # BUILD-ONLY
    "answer_format": "",               # BUILD-ONLY
    "subject": "math",                # derived from metadata
    "gpu_category_class": 3.0,         # COMPUTED
    "gpu_source_class": 0.0,           # COMPUTED
    "gpu_galaxy_index": 5.0,           # COMPUTED
    "gpu_has_template_ref": 0.0,       # COMPUTED
    "output_grid": None,               # copy from entry
    "arc_transform_chain": [],         # copy from entry/metadata
    "arc_color_mapping": {},           # copy from entry/metadata
    "arc_primitive_plan": [],          # copy from entry/metadata
    "arc_task_id": "",                 # BUILD-ONLY
}
```

### Slim catalog entry (~8 computed fields + reference, ~200 bytes each, ~55 MB total):
```python
{
    "index": 42,
    "galaxy": "Math",                  # needed for reference lookup
    "entry_idx": 17,                   # position in galaxy.entries list
    "confidence": 0.95,                # COMPUTED
    "domain_hash": 0.7234,             # COMPUTED
    "subject_hash": 0.3456,            # COMPUTED
    "embedding16": [0.1, 0.2, ...],    # COMPUTED (or read from entry)
    "gpu_category_class": 3.0,         # COMPUTED
    "gpu_source_class": 0.0,           # COMPUTED
    "gpu_galaxy_index": 5.0,           # COMPUTED
    "gpu_has_template_ref": 0.0,       # COMPUTED
}
```

**At query time, when code needs `catalog_entry["metadata"]`:**
```python
# Resolve the full Galaxy entry on demand
galaxy = self.galaxy_manager.get_galaxy(catalog_entry["galaxy"])
full_entry = galaxy.entries[catalog_entry["entry_idx"]]
metadata = full_entry.get("metadata", {})
```

### Access Helper

Add one method that merges the slim catalog entry with the Galaxy entry:

```python
def _resolve_catalog_entry(self, slim_entry: dict) -> dict:
    """Merge slim catalog with Galaxy entry for full access."""
    galaxy = self.galaxy_manager.get_galaxy(slim_entry["galaxy"])
    full_entry = galaxy.entries[slim_entry["entry_idx"]]
    resolved = dict(slim_entry)
    resolved["id"] = full_entry.get("id", full_entry.get("rule_id", ""))
    resolved["name"] = full_entry.get("name", "")
    resolved["category"] = full_entry.get("category", "")
    resolved["domain"] = full_entry.get("domain", slim_entry["galaxy"])
    resolved["metadata"] = full_entry.get("metadata", {})
    resolved["rpn_program"] = full_entry.get("rpn_program", "")
    resolved["answer_text"] = self._entry_answer_text(full_entry)
    resolved["template_ref"] = self._entry_template_ref(
        full_entry, full_entry.get("metadata", {})
    )
    resolved["subject"] = (
        full_entry.get("metadata", {}).get("subject", "")
        if isinstance(full_entry.get("metadata"), dict) else ""
    )
    resolved["output_grid"] = (
        full_entry.get("metadata", {}).get("output_grid", full_entry.get("output_grid"))
    )
    resolved["arc_transform_chain"] = list(
        full_entry.get("metadata", {}).get("arc_transform_chain", [])
    )
    resolved["arc_color_mapping"] = dict(
        full_entry.get("metadata", {}).get("arc_color_mapping", {})
    )
    resolved["arc_primitive_plan"] = list(
        full_entry.get("metadata", {}).get("arc_primitive_plan", [])
    )
    return resolved
```

This method is called ONLY when a query result needs full entry data — not for all
278K entries at bind time.

---

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Catalog pickle size | 818 MB | ~55 MB |
| Catalog deserialize | 36 seconds | ~2 seconds |
| Bind with cache | 58 seconds | ~8 seconds |
| Total boot | 75+ seconds | ~42 seconds |
| Memory duplication | 818 MB of copied Galaxy data | Zero (references only) |

---

## THEN: RUN THE BENCHMARKS

After the slim catalog, the boot chain is:
1. Init (warm boot from binary checkpoint): 34s
2. Flat buffer cache hit: <1s
3. Slim catalog cache hit: ~2s
4. CSR graph cache hit: 4.5s
5. Device buffer upload: <1s
6. **Total: ~42s to ready**

**Immediately after E.34, run:**

### Local ARC3
```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 scripts/run_arc3_local.py \
    --count 20 --grid-size 8 --max-actions 40 \
    --storage-root /K3D/Knowledge3D.local
```

### Full Benchmark
```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 scripts/run_full_benchmark.py \
    --mmlu-count 50 --gsm8k-count 10 --lhe-count 10 --arc2-count 10 \
    --storage-root /K3D/Knowledge3D.local
```

**Report:**
1. Accuracy per suite (compare to March 19 baseline: 3,324/15,601)
2. Stars loaded (should be 248K+)
3. GPU utilization during run
4. Warm boot used (yes/no)
5. Boot time breakdown (init + bind)
6. Any hangs or failures

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Slim catalog in `_append_flattened_entry()`, add `_resolve_catalog_entry()`, update query-time access to resolve on demand |

## No New Files

---

## Success Criteria

- [ ] Catalog pickle < 100 MB (was 818 MB)
- [ ] Bind with cache < 10 seconds (was 58 seconds)
- [ ] Query-time resolution works (all 21 accessed fields available)
- [ ] Local ARC3 runs and produces results
- [ ] Full benchmark runs and produces results
- [ ] Results reported with star counts, GPU util, boot time
