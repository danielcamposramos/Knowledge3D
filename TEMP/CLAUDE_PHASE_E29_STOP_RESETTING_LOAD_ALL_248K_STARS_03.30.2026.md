# Claude -- Phase E.29: Stop Resetting -- Load ALL 248K Stars

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL -- Daniel noticed the star count regression

---

## Daniel's Concern (Verbatim)

> "I feel strange because we once had lots of stars - what happened? the constant resets."

---

## Diagnosis: The Stars Are There, They're Just Not Being Loaded

**248,078 stars exist on disk** across 19 galaxy JSONL files:

```
DEFAULT_GALAXIES (11 named):
  Drawing:      717     Language:   116,779
  Character:     36     Audio:           0
  Word:       3,682     3DObjects:     368
  Number:     1,001     Tool:           39
  Grammar:      401     ─────────────────
  Math:       6,291     SUBTOTAL: 130,476
  Reality:    1,162

DISCOVERED (disk scan, not in DEFAULT_GALAXIES):
  meaning_layer_stars:           117,497   ← THE BIG ONE
  Book_BiologyAtlas:                  16
  Book_LanguageFoundations:           17
  Book_MathematicsPrimer:             17
  Book_PhysicsHandbook:               18
  Book_ToolManual:                    17
  proceduralized_gsm8k_train_10:      10
  proceduralized_mmlu_val_10:         10
                                 ─────────
                                 SUBTOTAL: 117,602

GRAND TOTAL ON DISK: 248,078 stars
```

**The two largest files are `Language.jsonl` (116,779 entries, 261 MB) and
`meaning_layer_stars.jsonl` (117,497 entries, 210 MB).** Together they contain
234,276 stars -- 94% of all knowledge on disk.

### Problem 1: meaning_layer_stars.jsonl Has Incompatible Schema

`meaning_layer_stars.jsonl` entries use `star_id` instead of `id`, have no `galaxy`
field, no `content` field, and store embeddings/surface_forms as Python repr strings
instead of JSON:

```json
{
  "star_id": "synset_00001740_a",      // ← should be "id"
  "meaning_rpn": "SYNSET A ABLE DEF having the necessary means...",
  "surface_forms": "{'ar': {...}, 'en': {...}, ...}",  // ← Python dict repr, not JSON
  "confidence": "1",                    // ← string, not int
  "embedding_512": "None",             // ← string "None", not null
  "house_room": "House/Library",
  "meaning_class": "adjective",
  // no "galaxy" field, no "content" field, no "category" field
}
```

The `GalaxyManager._read_entries_from_disk()` will load them as dicts, but nothing
in the query pipeline can match them because:
- Token query uses `json.dumps(entry)` as haystack -- works but `star_id` ≠ `id`
- GPU bind flattens entries expecting `id`, `content`, `category` fields
- The 117K entries load into memory but are invisible to the composed head

### Problem 2: Language.jsonl Is Loaded (It's in DEFAULT_GALAXIES) But HUGE

Language.jsonl has 116,779 entries and IS in `DEFAULT_GALAXIES`. These load fine.
But at 261 MB of JSONL, parsing 116K lines at boot is the dominant cost.
This is correct behavior per Daniel's directive ("no init time target"), but
the entries are `meaning_symlink` category -- they bridge Language↔Drawing for
ARC reasoning.

### Problem 3: No Warm Boot Checkpoint Exists

There is NO `galaxy_consolidated_latest.json` in checkpoints/. The E.25 sleep-time
persistence spec was written but never implemented. Every boot is a cold boot from
raw JSONL -- no consolidated state, no learned improvements carried forward.

This is the "constant resets" Daniel noticed: every session starts from scratch.

---

## Three Fixes

### Fix 1: Normalize meaning_layer_stars.jsonl Schema

Write a one-time migration script (`scripts/normalize_meaning_stars.py`) that reads
`meaning_layer_stars.jsonl` and writes a normalized version where entries match the
standard Galaxy entry schema:

```python
# For each entry in meaning_layer_stars.jsonl:
normalized = {
    "id": entry["star_id"],                    # star_id → id
    "galaxy": "meaning_layer_stars",           # add galaxy field
    "category": entry.get("meaning_class", "meaning_star"),
    "layer": 2,                                # Layer 2 = Meaning
    "content": entry.get("meaning_rpn", ""),   # meaning_rpn → content
    "metadata": {
        "domain": entry.get("domain", ""),
        "surface_forms": ast.literal_eval(entry["surface_forms"]),  # Python repr → dict
        "house_room": entry.get("house_room", ""),
        "house_position": json.loads(entry.get("house_position", "[0,0,0]")),
        "confidence": int(entry.get("confidence", 1)),
        "polarity": int(entry.get("polarity", 0)),
        "behavior_rpn": entry.get("behavior_rpn") if entry.get("behavior_rpn") != "None" else None,
        "visual_rpn": entry.get("visual_rpn") if entry.get("visual_rpn") != "None" else None,
        "taxonomy_refs": ast.literal_eval(entry.get("taxonomy_refs", "[]")),
        "component_refs": ast.literal_eval(entry.get("component_refs", "[]")),
        "grammar_refs": ast.literal_eval(entry.get("grammar_refs", "[]")),
        "reality_refs": ast.literal_eval(entry.get("reality_refs", "[]")),
    }
}
```

**Important:** This is an ingestion-path migration. Run once, overwrite the file,
done. The 117K meaning stars become properly indexed and queryable.

### Fix 2: Implement Sleep-Time Save (E.25 Spec, Never Implemented)

After `jarvis_sleep_consolidation()` runs, save the consolidated state:

```python
def _save_consolidated_state(self) -> Path:
    """Save consolidated Galaxy state + TRM weights after sleep-time."""
    checkpoint_dir = self.storage_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save consolidated Galaxy catalog
    catalog = {}
    for name in self._discover_live_galaxy_names():
        galaxy = self.galaxy_manager.get_galaxy(name)
        catalog[name] = list(getattr(galaxy, "entries", []))
    catalog_path = checkpoint_dir / f"galaxy_consolidated_{timestamp}.json"
    catalog_path.write_text(json.dumps(catalog, default=str), encoding="utf-8")

    # 2. Update latest symlink (or copy)
    latest_path = checkpoint_dir / "galaxy_consolidated_latest.json"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(catalog_path.name)

    # 3. Save TRM weights (already has save_checkpoint)
    # 4. Save Jarvis state (already saves)
    # 5. Save specialist routes

    return catalog_path
```

Call this at the end of `jarvis_sleep_consolidation()`.

### Fix 3: Implement Warm Boot from Consolidated State

At the start of `__init__`, before loading from raw JSONL:

```python
consolidated = self.storage_root / "checkpoints" / "galaxy_consolidated_latest.json"
if consolidated.exists():
    # Warm boot: load saved state (faster + carries learned improvements)
    self._load_consolidated_state(consolidated)
else:
    # Cold boot: load raw JSONL (first time ever)
    self.ensure_default_galaxies_loaded()
```

The warm boot path loads the consolidated JSON (one file, pre-parsed) instead of
reading 19 separate JSONL files. It also carries any EMA-smoothed embeddings,
pruned duplicates, and strengthened paths from sleep-time consolidation.

**Next boot = resume where you left off. Not reset.**

---

## What This Achieves

| Metric | Before E.29 | After E.29 |
|--------|-------------|------------|
| Stars loaded at boot | ~13K (11 DEFAULT_GALAXIES minus Language/meaning_layer) | 248,078 (all on disk) |
| meaning_layer_stars queryable | No (schema mismatch) | Yes (normalized) |
| Sleep-time saves state | No | Yes (galaxy_consolidated_latest.json) |
| Warm boot from saved state | No (always cold boot) | Yes (second boot loads checkpoint) |
| "Constant resets" | Every session starts from scratch | Each session resumes from last |

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/normalize_meaning_stars.py` | One-time migration: fix schema in meaning_layer_stars.jsonl |

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Add `_save_consolidated_state()` after sleep-time; add warm boot path in `__init__` |

## No New Python Orchestration

- Migration script = ingestion path (run once, done)
- `_save_consolidated_state()` = persistence (runs during sleep-time, not query path)
- Warm boot = `__init__` optimization (load checkpoint vs raw JSONL)
- Zero changes to the hot path

---

## The Learning Cycle (Daniel's Vision, Finally Real)

```
Session 1 (Cold Boot):
  Load ALL 248K stars from 19 JSONL files
  Answer queries → Shadow copy records traces
  Sleep-time → consolidate → SAVE to checkpoints/galaxy_consolidated_latest.json

Session 2 (Warm Boot):
  Load SAVED state (one file, pre-parsed, consolidated)
  TRM weights carry learned specialist preferences
  Answer more queries → more shadow copy traces
  Sleep-time → consolidate further → SAVE (update checkpoint)

Session N:
  Each boot loads a STRONGER Galaxy
  Each sleep makes it stronger still
  The model never resets — it always resumes where it left off
```

---

## Success Criteria

- [ ] meaning_layer_stars.jsonl entries normalized to standard schema (id, galaxy, content, category)
- [ ] All 248K stars loaded at boot (verify with entry count logging)
- [ ] Sleep-time consolidation saves checkpoint to disk
- [ ] Second boot loads from checkpoint (warm boot, not cold)
- [ ] Warm boot is faster than cold boot (natural, not forced)
- [ ] No regression in local ARC3 or benchmark tests
