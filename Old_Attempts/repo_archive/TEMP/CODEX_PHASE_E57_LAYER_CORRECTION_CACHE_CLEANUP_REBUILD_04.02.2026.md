# CODEX E.57 — Layer Correction, Cache Cleanup Policy, Warm-Boot Artifact Rebuild

**Date:** April 2, 2026
**Prerequisite:** E.56 implemented and tests passing (32 passed)
**Disk state:** Root SSD freed from 97% → 81% (137GB removed). Rebuild can now complete.
**Sovereignty:** No Python in hot path. No fallbacks. We fail and fix.

---

## Architectural Correction: The Four Layers (Read This First)

Daniel corrected the layer architecture. The spine router stars in E.56 have `layer_id=2` which is wrong. Here is the authoritative understanding:

**Layer 1 — FORM (Drawing Primitives)**
The actual foundation. Pure geometry: LINE, CIRCLE, BEZIER, ARC, RECT, vectors, points. No meaning — just shape. Everything in K3D builds on these. The Drawing Galaxy drawing primitives are Layer 1 stars.

**Layer 2 — SYMBOLOGY**
*Only* letters, tables, characters, and mathematical/scientific symbology — constructed FROM Layer 1 forms. Examples: the glyph "A" built from LINE strokes, the symbol "√" built from BEZIER curves, the character "字" built from ink-stroke primitives. This is the Character Galaxy + Math Galaxy in their symbol-encoding role. NOT general knowledge, NOT routing rules. Just symbols.

**Layer 3 — RULES**
How symbols combine and transform. Grammar rules, mathematical operations (add_op is: A B ADD — a rule about how the + symbol operates), physical laws, action rules (MOVE_UP is: apply +Y displacement). The math operation stars correctly have `layer_id=3`.

**Layer 4 — META-RULES**
Strategies, routing policies, meta-knowledge about how to process things. Routing decisions live here: "when you see a math query, dispatch to math executors" is meta-knowledge, not a math symbol (Layer 2) or a math operation (Layer 3).

**Dynamic loading** — any asset loaded into the House (3D model, network-transferred document, external file) connects to the Galaxy through symlinks that trace back to Layer 1 drawing primitives. The canonical drawing layer is the universal anchor. A photo of a cat loaded from the network: its pixels → Layer 1 drawing primitives (points, curves) → Layer 2 character encoding (how it's labelled) → Layer 3 rules (what cats do) → Layer 4 meta-rules (how to reason about cats). The symlink nature means you don't duplicate; you reference the canonical Layer 1 form.

---

## Task A: Fix layer_id for Router Spine Stars

**File: `knowledge3d/knowledgeverse/foundational_galaxy_builder.py`**

In `_reasoning_spine_stars()`, the two router stars have `layer_id=2` which incorrectly places them in the SYMBOLOGY layer. They are meta-routing strategies → Layer 4.

**Change `math_question_router`:** `"layer_id": 2` → `"layer_id": 4`

**Change `question_router`:** `"layer_id": 2` → `"layer_id": 4`

No other spine stars need changes (executors at layer_id=3 and validators at layer_id=4 are correct).

After this change, rebuild the spine star embedding position: the `semantic_position` for a star is `[domain_hash, subject_hash, float(layer_id) / 4.0]`. Router stars at layer_id=4 → z-position=1.0 (top of the layer stack). This is architecturally correct — meta-rules float above rules and symbology in the 3D House space.

**Verify with:**
```bash
env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table
stars = build_foundational_galaxy_table()
for s in stars:
    if s.get('id') in ('math_question_router', 'question_router'):
        assert s.get('layer_id') == 4, f'{s[\"id\"]} has layer_id={s.get(\"layer_id\")}'
        print(f'{s[\"id\"]}: layer_id=4 OK')
"
```

---

## Task B: Cache Cleanup Policy (Prevent 15GB Accumulation Recurrence)

**Context:** The gpu_cache directory had 77 stale files (15GB) and the checkpoints dir had 76 stale galaxy_consolidated snapshots (137GB), filling root to 97%. This has been manually cleaned; now prevent recurrence.

### B1: GPU Flat Cache Rotation

**File: `knowledge3d/knowledgeverse/knowledgeverse.py`**

In `_save_gpu_flat_cache()` (line ~1631), AFTER saving the new files, delete all OTHER flat_*.npy and catalog_*.pkl files in `gpu_cache/` that don't match the current signature:

```python
def _save_gpu_flat_cache(self, *, signature: str, flat_entries, catalog):
    flat_path, catalog_path = self._gpu_flat_cache_paths(signature)
    cache_dir = self._gpu_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # ... existing save logic ...
    # AFTER saving: delete stale entries (keep only current signature)
    for stale in cache_dir.glob("flat_*.npy"):
        if stale.stem != f"flat_{signature}":
            try:
                stale.unlink(missing_ok=True)
                catalog_stale = stale.with_name(stale.name.replace("flat_", "catalog_").replace(".npy", ".pkl"))
                catalog_stale.unlink(missing_ok=True)
            except Exception:
                pass
```

This keeps exactly one flat cache (current signature). Cache misses trigger a rebuild that saves the new one and deletes the old.

### B2: Galaxy Consolidated Snapshot Rotation

**File: `knowledge3d/knowledgeverse/knowledgeverse.py`** (or wherever galaxy_consolidated_*.json is written)

Find where `galaxy_consolidated_TIMESTAMP.json` files are written. After saving a new snapshot and updating the `galaxy_consolidated_latest.json` symlink, delete all timestamped snapshots except the 2 most recent:

```python
# After saving new galaxy_consolidated_TIMESTAMP.json:
consolidated_dir = self.storage_root / "checkpoints"
snapshots = sorted(consolidated_dir.glob("galaxy_consolidated_2*.json"), key=lambda p: p.name)
for stale in snapshots[:-2]:  # keep 2 most recent
    try:
        stale.unlink(missing_ok=True)
    except Exception:
        pass
```

This caps snapshots at 2 × ~1.8GB = ~3.6GB maximum.

**Verify after implementing:**
```bash
ls /K3D/Knowledge3D.local/checkpoints/galaxy_consolidated_*.json | wc -l
# Expected: ≤ 2 timestamped files + 1 symlink = 3 total
```

---

## Task C: Warm-Boot Artifact Rebuild

**Context:** `sovereign_runtime_bundle.pkl` does not exist. Every boot hits the slow rebuild path. The disk now has 167GB free. The rebuild needs to run to completion ONCE to create the artifact.

The rebuild path in `sovereign_hot_path.ensure_loaded()`:
1. `_load_runtime_artifacts()` → fails (no bundle) → falls through
2. `build_gpu_catalog_only()` → checks `_load_gpu_flat_cache()` → may hit or miss
3. `_build_stars_from_catalog(catalog)` → builds 259k star dicts with resolved refs
4. `star_table.load_stars(stars)` → uploads to GPU
5. `save_runtime_artifacts()` → saves bundle pkl + manifest json

**Steps for Codex:**

### C1: Profile the rebuild to find bottleneck

Add timing logs around the catalog build in `sovereign_hot_path.ensure_loaded()` (lines ~370-400):

```python
catalog_build_t0 = time.perf_counter()
catalog = list(self.knowledgeverse.build_gpu_catalog_only(...))
catalog_build_s = time.perf_counter() - catalog_build_t0
# log: f"catalog_build: {len(catalog)} entries in {catalog_build_s:.1f}s"

star_build_t0 = time.perf_counter()
stars = self._build_stars_from_catalog(catalog)
star_build_s = time.perf_counter() - star_build_t0
# log: f"star_build: {len(stars)} stars in {star_build_s:.1f}s"
```

This timing already exists in some form — make sure it's logged to stdout so we see where time is spent.

### C2: Trigger the rebuild once with GPU access

After implementing C1 and Task A (layer fix), run this maintenance command:

```bash
export CUDA_VISIBLE_DEVICES=0
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python - <<'EOF'
import time
t0 = time.perf_counter()
print("Starting sovereign runtime rebuild...")
from knowledge3d.knowledgeverse.knowledgeverse import KnowledgeVerse
kv = KnowledgeVerse()
kv.startup()
shp = kv.sovereign_hot_path
shp.invalidate_loaded_state()
result = shp.ensure_loaded()
print(f"Rebuild complete in {time.perf_counter()-t0:.1f}s")
print(f"Stars loaded: {result.get('star_count')}")
print(f"Mode: {result.get('mode')}")
manifest = shp.save_runtime_artifacts()
print(f"Artifact saved: {manifest.get('star_count')} stars")
print(f"  default_knowledge_signature: {manifest.get('default_knowledge_signature')}")
kv.shutdown()
EOF
```

This must run to completion (let it run — expect 5-15 minutes for first rebuild on 259k stars). After it finishes, subsequent boots will use the artifact and be fast (< 30 seconds).

### C3: Accept the rebuild time as a one-time sovereign cost

Document in `docs/ENV_POLICY.md` or `docs/MAINTENANCE.md`:

```
## Warm-Boot Artifact Rebuild

When the knowledge corpus changes (new stars, fixed refs, schema updates), the
sovereign runtime artifact (sovereign_runtime_bundle.pkl) must be rebuilt once:

  export CUDA_VISIBLE_DEVICES=0
  python scripts/rebuild_sovereign_artifact.py

Rebuild time: ~5-15 minutes for 259k stars (one-time cost).
Subsequent boots: < 30 seconds (artifact loaded from disk).

Do NOT interrupt the rebuild — it must complete to save the artifact.
```

Create `scripts/rebuild_sovereign_artifact.py` as a proper maintenance script with progress output.

---

## Task D: Write `scripts/rebuild_sovereign_artifact.py`

This is the canonical maintenance tool for forcing an artifact rebuild:

```python
#!/usr/bin/env python3
"""Force rebuild of the sovereign runtime artifact (sovereign_runtime_bundle.pkl).

Run after:
- Adding new knowledge to the Galaxy
- Changing foundational star structure (refs, roles, embeddings)
- Schema version bumps

Usage:
    export CUDA_VISIBLE_DEVICES=0
    python scripts/rebuild_sovereign_artifact.py [--verbose]
"""
import argparse
import os
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("K3D Sovereign Runtime Rebuild")
    print("=" * 40)

    from knowledge3d.knowledgeverse.knowledgeverse import KnowledgeVerse
    kv = KnowledgeVerse()
    print("Starting up...")
    kv.startup()

    shp = kv.sovereign_hot_path
    print("Invalidating cached state...")
    shp.invalidate_loaded_state()

    print("Building sovereign star table (this takes 5-15 minutes)...")
    result = shp.ensure_loaded()

    elapsed = time.perf_counter() - t0
    star_count = result.get("star_count", 0)
    mode = result.get("mode", "unknown")
    print(f"  Mode: {mode}")
    print(f"  Stars loaded: {star_count:,}")
    print(f"  Elapsed: {elapsed:.1f}s")

    if args.verbose:
        for key, value in sorted(result.items()):
            print(f"  {key}: {value}")

    print("Saving artifact...")
    manifest = shp.save_runtime_artifacts()
    print(f"  Saved: {manifest.get('star_count'):,} stars")
    print(f"  Signature: {manifest.get('default_knowledge_signature')}")
    print(f"  Path: {shp._artifact_bundle_path()}")

    print("Shutting down...")
    kv.shutdown()
    print(f"Done in {time.perf_counter()-t0:.1f}s")

if __name__ == "__main__":
    main()
```

---

## Validation

```bash
# 1. Layer fix
env PYTHONPATH=. .../python -c "
from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table
stars = build_foundational_galaxy_table()
for s in stars:
    if s.get('id') in ('math_question_router', 'question_router'):
        assert s.get('layer_id') == 4, f'FAIL: {s[\"id\"]} layer={s.get(\"layer_id\")}'
print('Layer fix: PASS')
"

# 2. Existing tests still pass
env PYTHONPATH=. .../python -m pytest -q tests/test_house_state.py tests/test_trm_game_loop.py tests/test_galaxy_vram_table.py tests/test_gpu_task_dispatch.py tests/test_spine_routing.py
# Expected: 32 passed

# 3. Rebuild script exists and parses
env PYTHONPATH=. .../python scripts/rebuild_sovereign_artifact.py --help

# 4. (GPU required) Run the actual rebuild — let it complete
export CUDA_VISIBLE_DEVICES=0
env PYTHONPATH=. .../python scripts/rebuild_sovereign_artifact.py --verbose
# Expected output: "Saved: N stars" where N ≥ 259,943
```

---

## Files Changed

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/foundational_galaxy_builder.py` | `math_question_router` + `question_router`: layer_id 2 → 4 |
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Cache rotation in `_save_gpu_flat_cache()` + galaxy_consolidated rotation |
| `scripts/rebuild_sovereign_artifact.py` | New maintenance script |
| `docs/MAINTENANCE.md` (create if absent) | Document rebuild procedure |

No hot-path changes. No sovereignty impact. Tests must stay at 32 passed.
