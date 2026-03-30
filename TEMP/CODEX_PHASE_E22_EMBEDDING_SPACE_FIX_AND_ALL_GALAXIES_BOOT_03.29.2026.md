# Codex — Phase E.22: Embedding Space Fix + All-Galaxies-Always-Loaded Boot

**Date:** 2026-03-29
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — ARC3 nav rules never selected; embedding space mismatch confirmed

---

## Two Problems, Two Fixes

---

## Problem 1: ARC3 Nav Rules Never Win — Embedding Space Mismatch

### Root Cause (Traced to Exact Code)

The composed head pipeline computes cosine similarity between the **query embedding** and each **catalog entry embedding**. These two embeddings are built by **different algorithms**:

**Query embedding** → `_embed_query_gpu()` → `RPNEmbeddingEngine.embed_sentence_gpu()`:
- Tokenizes by whitespace
- For each token: extracts character trigrams → hashes each trigram → looks up in embedding table → mean-pools
- This is a **trigram-based** embedding

**Catalog entry embedding** → `_entry_embedding16()`:
- Checks `entry.get("embedding16")` → not present for ARC3 entries
- Checks `entry.get("embedding")` → **FOUND for ARC3 entries** (32-dim FNV-1a from `embed_text_sovereign()`)
- Truncates first 16 dims and returns
- This is an **FNV-1a hash bucket** embedding

**These are incompatible spaces.** Cosine similarity between a trigram embedding and an FNV-1a embedding is essentially noise. ARC3 nav rules always score near zero — which is why `reasoning_arc_grid_transform_top1` (an ARC-2 entry with no stored embedding → falls through to trigram path → same space as query) always wins.

### Why Other Galaxy Entries Work

Most Galaxy entries (Drawing, Reality, Grammar rules from older population) do NOT have a stored `embedding` field in their JSONL. So `_entry_embedding16()` falls through to the text fallback:
```python
text = self._entry_embedding_text(entry)
if text:
    return self._normalize_embedding(list(self.get_text_embedding_engine().embed_sentence(text)))
```
This calls `RPNEmbeddingEngine.embed_sentence()` (CPU path, same trigram algorithm as query GPU path). Same space → meaningful cosine similarity.

The ARC3 knowledge builder explicitly stored FNV-1a embeddings, breaking this alignment.

---

### Fix 1A: Remove stored `embedding` from ARC3 entries

In `knowledge3d/knowledgeverse/arc3_knowledge_builder.py`, **delete the embedding generation entirely** from `_make_entry()`. The function should return the entry as-is, without computing or storing an `embedding` field:

```python
def _make_entry(entry_def: dict[str, Any]) -> dict[str, Any]:
    return dict(entry_def)
```

Remove `_embed_nav_rule()` and `_normalize_embedding()` — they are no longer needed.
Remove the import of `embed_text_sovereign` — it is no longer called.

When entries have no `embedding` field, `_entry_embedding16()` falls through to:
```python
text = self._entry_embedding_text(entry)
return self._normalize_embedding(list(self.get_text_embedding_engine().embed_sentence(text)))
```

`_entry_embedding_text()` checks `metadata.query_anchor` FIRST (before `name`, `content`, `description`). This is where we put our optimized embedding text.

---

### Fix 1B: Add `metadata.query_anchor` to each directional nav rule

`_entry_embedding_text()` checks `metadata.query_anchor` before all other fields. This string becomes the embedding text — the one that must score high against the query.

The query for a frame where the object is above center looks like:
```
"arc3 interactive game frame grid 64x64 object above center top north goal absent available actions move down move right perform levels navigation visual"
```

The `query_anchor` for `arc3_nav_move_up` should contain: "object above center top north", "move up", "navigation", "arc3" — tokens that share trigrams with the query.

Update each directional rule's `metadata` dict in `ARC3_GRAMMAR_RULES`:

```python
# arc3_nav_move_up — add to metadata:
"query_anchor": "object above center top north navigate arc3 game frame move up"

# arc3_nav_move_down — add to metadata:
"query_anchor": "object below center bottom south navigate arc3 game frame move down"

# arc3_nav_move_left — add to metadata:
"query_anchor": "object left west navigate arc3 game frame move left"

# arc3_nav_move_right — add to metadata:
"query_anchor": "object right east navigate arc3 game frame move right"

# arc3_nav_perform — add to metadata:
"query_anchor": "perform action arc3 game frame execute interact"

# arc3_nav_click — add to metadata:
"query_anchor": "click coordinates arc3 game frame x y interact"

# arc3_nav_undo — add to metadata:
"query_anchor": "undo arc3 game frame recovery stagnant action loop"

# arc3_rule_keyboard_game — add to metadata:
"query_anchor": "keyboard navigation arc3 game frame move directional"

# arc3_rule_keyboard_click_game — add to metadata:
"query_anchor": "keyboard click navigation arc3 game frame move click interact"
```

**Why this works:** When the query contains "object above center top north", the trigram engine processes token "north" → trigrams ["nor", "ort", "rth"] → same buckets as when it processes "north" in the `query_anchor`. Cosine similarity increases proportional to shared trigrams. The spread between "move up" vs "move down" increases from ~0.05 to ~0.3+ because "north/above/top" trigrams differ from "south/below/bottom" trigrams.

---

### Fix 1C: Re-run the builder

After the code changes:
```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  python3 knowledge3d/knowledgeverse/arc3_knowledge_builder.py
```

This uses the idempotent upsert logic — entries with matching `id` are replaced. The rebuilt entries will have NO `embedding` field + `metadata.query_anchor`. After the next `Knowledgeverse()` init, `_entry_embedding16()` will compute trigram embeddings at runtime.

---

## Problem 2: Language Galaxy Not in DEFAULT_GALAXIES (Violates ALL-Always-Loaded)

### Root Cause

Daniel's mandate: **ALL knowledge is ALWAYS loaded. No per-task unload/reload.**

Current `DEFAULT_GALAXIES` in `knowledgeverse.py`:
```python
DEFAULT_GALAXIES: tuple[str, ...] = (
    "Drawing", "Character", "Word", "Number",
    "Grammar", "Math", "Reality", "Audio", "3DObjects", "Tool",
)
```

`"Language"` is missing. But `GPU_ARC_TARGET_GALAXIES = ("Language", "Drawing", "Grammar", "Tool")` includes it. When an ARC-2 benchmark call uses this route, `bind_gpu_galaxy_runtime(galaxy_names=["Language", "Drawing", "Grammar", "Tool"])` is called. The `_pinned_all_default_binding` guard checks `set(resolved_names).issubset(set(bound_names))` — Language is NOT in `bound_names` (DEFAULT_GALAXIES) → **triggers a catalog rebuild** that swaps Language in. This is exactly the per-task reload that must not happen.

### Fix 2: Add "Language" to DEFAULT_GALAXIES

In `knowledge3d/knowledgeverse/knowledgeverse.py`, add "Language" to `DEFAULT_GALAXIES`:

```python
DEFAULT_GALAXIES: tuple[str, ...] = (
    "Drawing",
    "Character",
    "Word",
    "Number",
    "Grammar",
    "Math",
    "Reality",
    "Audio",
    "3DObjects",
    "Tool",
    "Language",  # ADD THIS — ALL galaxies always loaded
)
```

**After this change:**
- `_pin_all_default_gpu_binding()` at init loads ALL 11 default galaxies including Language
- `set(["Language", "Drawing", "Grammar", "Tool"]).issubset(set(bound_names))` → True (Language now in bound)
- No per-task rebuild for ARC-2 routes
- No per-task rebuild for any route (all existing `GPU_*_TARGET_GALAXIES` tuples are now subsets)

**Note on TRM galaxy weights:** `_is_trm_galaxy_weights_valid()` checks `weights.shape != (len(self.DEFAULT_GALAXIES), ...)`. Adding Language changes `len(DEFAULT_GALAXIES)` from 10 to 11. Existing serialized TRM weight tensors (if any) will fail this check and be skipped gracefully (they're reloaded from defaults). The system is boot-tolerant: TRM weights are rebuilt from uniform priors when the shape doesn't match.

---

## Execution Sequence

1. **Fix `arc3_knowledge_builder.py`:**
   - Delete `_embed_nav_rule()`, `_normalize_embedding()`, and `embed_text_sovereign` import
   - Update `_make_entry()` to return `dict(entry_def)` with no embedding
   - Add `metadata.query_anchor` to each nav rule in `ARC3_GRAMMAR_RULES`
   - Add `query_anchor` to `arc3_rule_keyboard_game` and `arc3_rule_keyboard_click_game` in `ARC3_GRAMMAR_RULES`

2. **Fix `knowledgeverse.py`:**
   - Add `"Language"` to `DEFAULT_GALAXIES`

3. **Compile check:**
   ```bash
   conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
     python3 -m py_compile knowledge3d/knowledgeverse/arc3_knowledge_builder.py \
     knowledge3d/knowledgeverse/knowledgeverse.py
   ```

4. **Re-run builder:**
   ```bash
   conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
     python3 knowledge3d/knowledgeverse/arc3_knowledge_builder.py
   ```
   Expected output: `replaced=7` (all 7 Grammar rules replaced), `replaced=3` (Tool), `added=N` (Reality).

5. **Live probe (10 steps):**
   ```bash
   conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
     env CUDA_VISIBLE_DEVICES=0 K3D_DEVICE_PIPELINE=1 K3D_TRM_SHADOW=1 K3D_TRM_NAVIGATE=1 \
     python scripts/run_arc3_agent.py --game-id re86-4e57566e --max-actions 10
   ```
   **Expected behavior after fix:**
   - `program_type: "gpu_arc3_navigation_rule"` (not `gpu_arc_no_output_grid`)
   - `result_answer_index: 0/1/2/3` based on object position
   - Actions VARY across steps (not all Move Up)
   - `confidence > 0.1` for directional matches

6. **Full benchmark:**
   ```bash
   conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
     env CUDA_VISIBLE_DEVICES=0 \
     python scripts/run_full_benchmark.py \
       --storage-root /K3D/Knowledge3D.local \
       --synthetic-count 10 --mmlu-count 50 --gsm8k-count 10 --lhe-count 10 --arc3-count 5
   ```

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/arc3_knowledge_builder.py` | Remove embedding generation; add `metadata.query_anchor` to nav rules |
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Add "Language" to `DEFAULT_GALAXIES` |

## Files NOT to Touch

| File | Why |
|------|-----|
| `benchmarks/arc_agi_3.py` | Already correct (E.20 spatial tokens in query) |
| `scripts/run_arc3_agent.py` | Already correct (GPU env defaults set) |
| `knowledge3d/knowledgeverse/galaxy_loader.py` | Already correct |
| All other knowledgeverse files | Already correct |
| All test files | No behavioral change in test-visible outputs |

---

## Why This Architecture Is Correct

The Knowledgeverse has one embedding engine for queries (`RPNEmbeddingEngine`, trigram-based). All catalog entries that DON'T store a pre-computed embedding get their embedding text via `_entry_embedding_text()` → `embed_sentence()` — the same algorithm. This is the **unified trigram space**: query and catalog are comparable.

When `arc3_knowledge_builder.py` stored FNV-1a embeddings, it created a **foreign body** in this space. Removing those embeddings restores alignment.

The `metadata.query_anchor` is the right mechanism for controlling embedding text: it's checked FIRST by `_entry_embedding_text()`, it persists in the catalog, and it doesn't require changes to the hot path. It is pure I/O encoding — describing what the nav rule IS in the language the trigram engine can compare.

**The loop after this fix:**
1. Frame → `_frame_to_query_text()` → "object above center top north ... move down move right ..."
2. `_embed_query_gpu()` → trigram embedding with high components for "north", "above", "top"
3. Catalog scan: `arc3_nav_move_up.metadata.query_anchor` = "object above center top north..." → trigram embedding similar to query → HIGH similarity (~0.5+)
4. `arc3_nav_move_down.metadata.query_anchor` = "object below center bottom south..." → LOW similarity (~0.1)
5. `_answer_arc_query()`: match = `arc3_nav_move_up`, `metadata.action_index = 0`
6. Returns `{"answer_index": 0, "program_type": "gpu_arc3_navigation_rule"}`
7. Adapter: ACTION1 (Move Up)
8. Object moves up → toward center → `levels_completed` increases → `outcome=1`
9. Sleep-time: `jarvis_sleep_consolidation()` strengthens "object above → move_up" path
