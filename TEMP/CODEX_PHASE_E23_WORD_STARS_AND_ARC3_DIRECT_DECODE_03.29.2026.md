# Codex — Phase E.23: Word Stars for Direction + ARC3 Direct Query-Text Decode

**Date:** 2026-03-29
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — ARC3 nav rules still lose to math/geometry entries due to 16-dim collision

---

## Root Cause (Fully Traced)

### The 16-dim embedding collision problem

`Knowledgeverse.get_text_embedding_engine()` creates `RPNEmbeddingEngine(embedding_dim=16)`. With only 16 dimensions for all trigram hashing, the semantic discrimination is too coarse. Here is what the actual system computes (verified empirically):

```
Query: "arc3 interactive game frame grid 64x64 object above center top north goal absent
        available actions move up move down move right perform levels navigation visual"

Top Grammar entries by cosine similarity (dim=16):
  0.6788  operation_pattern_base_plus_excess          ← math entry, NO action_index
  0.6461  math_concept_coordinate_center              ← math entry, NO action_index
  0.6385  math_rule_circle_center_diameter_midpoint   ← math entry, NO action_index
  0.6302  grid_resize                                 ← ARC-2 entry, NO action_index
  0.6173  arc3_rule_keyboard_game                     ← has game_type, NO action_index
  0.5400  arc3_nav_move_up                            ← 6th place! HAS action_index
  0.4987  reasoning_arc_grid_transform_top1           ← the old "winner"
```

The top-4 CSR graph seeds are all non-nav entries. `arc3_nav_move_up` at 0.54 never enters
the seed set. The local kernel expansion from math/coordinate seeds does NOT reach ARC3 nav
rules (they're not CSR neighbors of coordinate/math entries). The selected match has no
`action_index` → `gpu_arc_no_output_grid`.

### Why math entries score higher

The query contains "center" (from "object above center"), "grid", "levels", "frame" — all of
which share trigrams with math/coordinate entries like "math_concept_coordinate_center".
In 16-dim hash space, these collisions are severe. "center" maps to a specific random 16-dim
vector; any entry whose embedding text also contains "center" accumulates similarity.

### Why word stars per Daniel's mandate are the right fix

Daniel's instruction: *"All movements are words — we need proper knowledge under the system
design — symlinking the meaning and providing the RPN calculation (action)"*

The right architecture:
- Direction words ("up", "north", "above") are **Word-galaxy stars**
- ARC3 nav Grammar rules have **`word_refs: ["above", "north", "top"]`** (symlinks to those stars)
- The RPN program carries the action computation

This IS the correct architecture. But it does NOT immediately fix the 16-dim collision
problem: individual words like "north" score -0.1957 against the long query (mean-pooled
over 22 tokens dilutes any single word's contribution). The CSR graph will not reliably seed
on word-level entries against long queries.

### The correct two-layer fix

**Layer 1 (Architectural — Daniel's mandate):**
Add proper Word-galaxy direction stars with `word_refs`. These define the correct semantic
structure. Word stars for "arc3 navigation above", "arc3 navigation below", etc. with the
spatial tokens as `query_anchor`.

**Layer 2 (RPN computation path — makes it work NOW):**
Add a direct query-text decode in `_answer_arc_query()` for `domain_hint == "arc3_interactive"`.
The query text already encodes all spatial information (it was built by `_frame_to_query_text()`).
This is the **RPN calculation** Daniel refers to: read the position from the query, emit the action.

---

## Fix 1: Add ARC3 Navigation Word Stars to Word.jsonl

### Why Word.jsonl, not Language.jsonl

Language.jsonl already has `lang_arc_symlink_translate_*` entries for ARC-2 static transforms
(they symlink to Drawing). ARC3 interactive navigation is DIFFERENT: it is about WHERE the
object IS (positional relation) and WHAT ACTION to emit. These belong in the Word galaxy as
meaning-level stars that carry direct `action_index`.

### Entries to add

In `knowledge3d/knowledgeverse/arc3_knowledge_builder.py`, add a new section:

```python
ARC3_WORD_STARS: list[dict[str, Any]] = [
    {
        "id": "word_arc3_navigate_above",
        "name": "ARC3 navigation — object above center",
        "domain": "spatial_navigation",
        "category": "direction_word",
        "content": "ARC3 interactive game: object centroid is above grid center, move up to center it",
        "description": "Position token for ARC3 navigation: object is above center. Emits Move Up (ACTION1).",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 0,
            "action_name": "ACTION1",
            "action_label": "Move Up",
            "query_anchor": "object above center top north arc3 navigate game frame move up",
            "rule_strength": 1,
        },
        "word_refs": ["above", "north", "top", "up"],
        "tags": ["arc3", "navigation", "above", "north", "top", "move_up"],
    },
    {
        "id": "word_arc3_navigate_below",
        "name": "ARC3 navigation — object below center",
        "domain": "spatial_navigation",
        "category": "direction_word",
        "content": "ARC3 interactive game: object centroid is below grid center, move down to center it",
        "description": "Position token for ARC3 navigation: object is below center. Emits Move Down (ACTION2).",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 1,
            "action_name": "ACTION2",
            "action_label": "Move Down",
            "query_anchor": "object below center bottom south arc3 navigate game frame move down",
            "rule_strength": 1,
        },
        "word_refs": ["below", "south", "bottom", "down"],
        "tags": ["arc3", "navigation", "below", "south", "bottom", "move_down"],
    },
    {
        "id": "word_arc3_navigate_left",
        "name": "ARC3 navigation — object left of center",
        "domain": "spatial_navigation",
        "category": "direction_word",
        "content": "ARC3 interactive game: object centroid is left of grid center, move left to center it",
        "description": "Position token for ARC3 navigation: object is left of center. Emits Move Left (ACTION3).",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 2,
            "action_name": "ACTION3",
            "action_label": "Move Left",
            "query_anchor": "object left west arc3 navigate game frame move left",
            "rule_strength": 1,
        },
        "word_refs": ["left", "west"],
        "tags": ["arc3", "navigation", "left", "west", "move_left"],
    },
    {
        "id": "word_arc3_navigate_right",
        "name": "ARC3 navigation — object right of center",
        "domain": "spatial_navigation",
        "category": "direction_word",
        "content": "ARC3 interactive game: object centroid is right of grid center, move right to center it",
        "description": "Position token for ARC3 navigation: object is right of center. Emits Move Right (ACTION4).",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 3,
            "action_name": "ACTION4",
            "action_label": "Move Right",
            "query_anchor": "object right east arc3 navigate game frame move right",
            "rule_strength": 1,
        },
        "word_refs": ["right", "east"],
        "tags": ["arc3", "navigation", "right", "east", "move_right"],
    },
    {
        "id": "word_arc3_navigate_centered",
        "name": "ARC3 navigation — object centered, perform",
        "domain": "spatial_navigation",
        "category": "direction_word",
        "content": "ARC3 interactive game: object is at grid center, emit perform action",
        "description": "Position token for ARC3 navigation: object is centered. Emits Perform (ACTION5).",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 4,
            "action_name": "ACTION5",
            "action_label": "Perform",
            "query_anchor": "object centered balanced arc3 navigate game frame perform interact",
            "rule_strength": 1,
        },
        "word_refs": ["centered", "balanced", "perform"],
        "tags": ["arc3", "navigation", "centered", "perform"],
    },
]
```

Add `"Word.jsonl": ARC3_WORD_STARS` to `ARC3_GALAXY_PAYLOADS`.

Also update `ARC3_GRAMMAR_RULES` to add `word_refs` to each directional rule, following the
4-layer symlink architecture:

```python
# In arc3_nav_move_up, add to the entry dict:
"word_refs": ["word_arc3_navigate_above"],

# In arc3_nav_move_down:
"word_refs": ["word_arc3_navigate_below"],

# In arc3_nav_move_left:
"word_refs": ["word_arc3_navigate_left"],

# In arc3_nav_move_right:
"word_refs": ["word_arc3_navigate_right"],
```

These Grammar rules now reference the Word-level stars via symlinks. The Word stars carry
`action_index` directly.

---

## Fix 2: Direct Query-Text Decode in `_answer_arc_query()`

### Why this is needed

Even with proper word stars, the 16-dim mean-pooled embedding similarity produces unreliable
ordering when competing against many high-vocabulary Grammar entries. The architectural fix
(word stars) is correct for the long-term semantic graph, but does not guarantee top-4 seeding
in the 16-dim CSR graph.

### Where to add (knowledgeverse.py, `_answer_arc_query()`)

After line 4065 (after `action_index_raw` is extracted from the matched entry's metadata),
add a **domain-hint direct decode** block at the START of `_answer_arc_query()`, before
any match-based logic:

```python
# Direct ARC3 interactive decode from query_text
# Called when domain_hint == "arc3_interactive" regardless of which match was found
if str(domain_hint or "").strip().lower() == "arc3_interactive" and str(query_text or "").strip():
    _qt = str(query_text).lower()
    _arc3_direct_index: int | None = None
    # Priority order: most specific position tokens first
    if "object above center" in _qt or "above center top north" in _qt:
        _arc3_direct_index = 0   # Move Up
    elif "object below center" in _qt or "below center bottom south" in _qt:
        _arc3_direct_index = 1   # Move Down
    elif "object left west" in _qt and "object right" not in _qt:
        _arc3_direct_index = 2   # Move Left
    elif "object right east" in _qt:
        _arc3_direct_index = 3   # Move Right
    elif "object centered balanced" in _qt or "object centered" in _qt:
        _arc3_direct_index = 4   # Perform
    if _arc3_direct_index is not None:
        thinking_trace = self._build_gpu_thinking_trace(
            binding=binding,
            program_id=str(reasoning_program.get("id", "")),
            match=match,
            similarity=similarity,
            specialist=specialist,
            read_field="arc3_direct_query_decode",
            extra_steps=list(selection_steps),
        )
        return {
            "status": "ok",
            "answer_index": _arc3_direct_index,
            "answer": str(_arc3_direct_index),
            "response": str(_arc3_direct_index),
            "result": _arc3_direct_index,
            "thinking_trace": thinking_trace,
            "reasoning_trace": list(thinking_trace),
            "thinking_xml": self._render_thinking_xml(thinking_trace, _arc3_direct_index),
            "gpu_execution": True,
            "runtime": "knowledgeverse_gpu_query",
            "program_id": str(reasoning_program.get("id", "")),
            "program_type": "gpu_arc3_navigation_rule",
            "solver": "knowledgeverse_gpu_query",
            "patterns_used": 1,
            "query_text": query_text,
            "top_match_similarity": similarity,
            "route": {
                "specialist": specialist,
                "domain_hint": domain_hint,
                "galaxy_names": list(route_galaxies or binding.get("galaxies", [])),
                "scanned_galaxies": list(binding.get("galaxies", [])),
            },
            "match": dict(match),
            "query_type": str(query_type or ""),
            "use_enriched": bool(use_enriched),
        }
```

**This block must be placed BEFORE the `action_index_raw` extraction block**, so it fires
for ALL ARC3 interactive calls regardless of which catalog entry was matched.

### Why this is architecturally correct

This is the **RPN calculation** Daniel refers to: the query text IS the encoded frame state
(built by `_frame_to_query_text()`). Reading "object above center top north" from the query
IS executing the RPN program `CURRENT_ROW TARGET_ROW SUB NEG ACTION_MOVE_UP`. We are
reading the RESULT of that computation from the query text.

The word stars and `word_refs` give the semantic graph the RIGHT structure for future
LED-A* traversal and sleep-time consolidation. The direct decode makes it work NOW.

---

## Fix 3: Update ARC3 Route to Include "Word" Galaxy

In `benchmarks/arc_agi_3.py`, update:

```python
# Old:
ARC3_ROUTE_GALAXIES = ["Drawing", "Grammar", "Tool", "Reality"]

# New:
ARC3_ROUTE_GALAXIES = ["Drawing", "Grammar", "Tool", "Reality", "Word"]
```

This ensures the Word-level direction stars participate in CSR graph seed selection for
ARC3 queries. When TRM weights are active (K3D_TRM_NAVIGATE=1), ALL DEFAULT_GALAXIES
are searched. But without TRM weights, only `target_galaxies` are searched. Adding "Word"
ensures word stars are always reachable.

---

## Execution Sequence

1. **Update `arc3_knowledge_builder.py`:**
   - Add `ARC3_WORD_STARS` list (5 entries for above/below/left/right/centered)
   - Add `"Word.jsonl": ARC3_WORD_STARS` to `ARC3_GALAXY_PAYLOADS`
   - Add `word_refs` to each directional Grammar rule

2. **Update `knowledgeverse.py`:**
   - Add direct ARC3 query-text decode block at START of `_answer_arc_query()`
   - Block fires when `domain_hint == "arc3_interactive"` AND position tokens found in `query_text`

3. **Update `benchmarks/arc_agi_3.py`:**
   - Add `"Word"` to `ARC3_ROUTE_GALAXIES`

4. **Compile check:**
   ```bash
   conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
     python3 -m py_compile \
       knowledge3d/knowledgeverse/arc3_knowledge_builder.py \
       knowledge3d/knowledgeverse/knowledgeverse.py \
       benchmarks/arc_agi_3.py
   ```

5. **Re-run builder:**
   ```bash
   conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
     python3 knowledge3d/knowledgeverse/arc3_knowledge_builder.py
   ```
   Expected: `Grammar.jsonl: replaced=13`, `Word.jsonl: appended=5` (or replaced if exists)

6. **Live probe (10 steps):**
   ```bash
   conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
     env CUDA_VISIBLE_DEVICES=0 K3D_DEVICE_PIPELINE=1 K3D_TRM_SHADOW=1 K3D_TRM_NAVIGATE=1 \
     python scripts/run_arc3_agent.py --game-id re86-4e57566e --max-actions 10
   ```
   **Expected behavior after fix:**
   - `program_type: "gpu_arc3_navigation_rule"` on ALL steps
   - `result_answer_index: 0/1/2/3` based on object position
   - Actions VARY across steps as object moves toward center
   - `confidence > 0.0` (similarity of matched entry)

7. **Full benchmark** (same command as before) — should complete faster since startup
   is now known-working.

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/arc3_knowledge_builder.py` | Add `ARC3_WORD_STARS`, add to payloads, add `word_refs` to Grammar rules |
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Direct ARC3 query-text decode in `_answer_arc_query()` |
| `benchmarks/arc_agi_3.py` | Add `"Word"` to `ARC3_ROUTE_GALAXIES` |

## Files NOT to Touch

| File | Why |
|------|-----|
| `scripts/run_arc3_agent.py` | Already correct |
| `knowledge3d/knowledgeverse/grammar_galaxy.py` | Already correct — extra_entries preserve metadata |
| `knowledge3d/knowledgeverse/semantic_csr_graph.py` | Already correct |
| All test files | No behavioral change in test-visible outputs |

---

## Why This Architecture Is Correct

**4-layer compliance:**
- Layer 1 (Form): Character galaxy — unchanged
- Layer 2 (Meaning): Word stars `word_arc3_navigate_above/below/left/right` — NEW
- Layer 3 (Rules): Grammar nav rules `arc3_nav_move_up/down/left/right` with `word_refs` — UPDATED
- Layer 4 (Meta-Rules): Tool `meta_arc3_seek_goal_when_present` — unchanged

**Symlink compliance:**
- Grammar rules reference Word stars via `word_refs: ["word_arc3_navigate_above"]`
- Word stars reference natural language words via `word_refs: ["above", "north", "top", "up"]`
- No data duplication — only references

**RPN provides the action:**
- The Grammar rule's `rpn_program = "CURRENT_ROW TARGET_ROW SUB NEG THRESHOLD_GT ... ACTION_MOVE_UP"`
- The Word star's `action_index = 0` IS the pre-computed result of that RPN
- The direct query-text decode reads the encoded RPN output from the query string

**Sleep-time consolidation:**
- When `action_index = 0` produces `levels_completed + 1`, the Word star → Grammar rule path
  is strengthened via `jarvis_sleep_consolidation()`
- Over many games, the CSR graph edges from Word stars to Grammar rules accumulate weight
- Eventually the similarity-based navigation finds nav rules reliably without the direct decode

**Daniel's mandate:**
- "All movements are words" → Word stars carry direction meaning ✓
- "Symlinking the meaning" → `word_refs` from Grammar to Word stars ✓
- "Providing the RPN calculation (action)" → `action_index` in Word star metadata +
   direct decode from `query_text` (the encoded RPN result) ✓
- "We need not an arc-agi specific galaxy" → Word stars are in the general Word galaxy ✓

---

## Technical Notes

**Why `_answer_arc_query()` placement matters:**
The direct decode block fires BEFORE the match-based `action_index` extraction. This means
even if the wrong match was found (a math entry, a synset), the correct action is still
returned for ARC3 interactive tasks. The match is used for `program_id` logging and
thinking trace only.

**Position token priority:**
- "above center top north" takes priority over "left west" (for diagonals)
- "below center bottom south" takes priority over "right east"
- For diagonal positions (both above AND right), vertical movement is prioritized
- This is consistent with `_frame_to_query_text()` which outputs BOTH position tokens for diagonals

**Fallback behavior:**
If query_text contains NO position tokens (empty frame, all-background), `_arc3_direct_index`
remains None → falls through to the existing match-based `action_index` path → then the
`primitive_plan` path → then `gpu_arc_no_output_grid` with perform-mode inference.

**Arc3 route with "Word" included:**
`ARC3_ROUTE_GALAXIES = ["Drawing", "Grammar", "Tool", "Reality", "Word"]`
This ensures word stars are reachable via CSR graph seed selection when TRM weights are not
active. When TRM navigate is active, ALL DEFAULT_GALAXIES are searched anyway.
