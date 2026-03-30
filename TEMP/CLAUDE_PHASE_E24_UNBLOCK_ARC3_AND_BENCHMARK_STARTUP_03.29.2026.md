# Claude — Phase E.24: Universal Movement Knowledge + Fix Startup Hang

**Date:** 2026-03-29
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — movement is universal knowledge, not benchmark-specific

---

## Daniel's Correction (Verbatim Principle)

> "ARC3 actions are words with meanings — they CAN be translated into RPN actions
> that actually encode true movement. No bootstrapping, default stars loaded from
> the House, that contains true RPN opcodes for the TRM single brain with internal
> specialist swarm to reason inside the Galaxy memory. All knowledge must be
> available from the start. Why are we crafting a new galaxy that's benchmark-tied
> when the knowledge of movement is procedural and universal?"

---

## What's Wrong (Three Violations)

### 1. Bootstrap at init = knowledge computed at runtime instead of persisted

```python
# WRONG — in knowledgeverse.py __init__ (unstaged diff lines 361-362)
self.reality_galaxy = RealityGalaxy(galaxy_path=...)
self.reality_action_node_ids = bootstrap_spatial_actions(self.reality_galaxy, encode_embedding=True)
```

Knowledge doesn't get bootstrapped. Knowledge lives in the House, loads into Galaxy at boot.
This is also what hangs the benchmark startup — CPU-bound embedding computation.

### 2. Benchmark-named stars = confusion

```
word_arc3_navigate_above    ← WRONG: "above" is not an ARC3 concept
arc3_nav_move_up            ← WRONG: "move up" is universal
arc3_rule_keyboard_game     ← WRONG: keyboard navigation is universal
```

"Above", "up", "north" are WORDS with MEANING. The same star answers:
- ARC3: "object is above center → move up"
- House navigation: "the library is above → go upstairs"
- Physics: "apply upward force"
- Robotics: "move arm up"

### 3. Action atoms in a separate `RealityGalaxy` object, not in the unified VRAM Galaxy

`action_primitives_bootstrap.py` has the RIGHT content:
- `behavior_rpn: "y RECALL dy RECALL - y STORE"` (true movement computation)
- `law_rpn: "y RECALL y_min RECALL GTE"` (boundary constraint)
- `visual_rpn: "0 0 DRAW_MOVE 0 -1 DRAW_LINE DRAW_STROKE"` (visual form)
- `surface_forms: {"en": "move up", "pt": "mover para cima"}` (multilingual)
- `reusable_contexts: ["arc3", "house_navigation", "grid_world", "physics_sim"]`

But it's stored in a SEPARATE `RealityGalaxy` object that nothing in the query path reads.
It should be Reality Galaxy STARS in Reality.jsonl — loaded into VRAM with everything else.

---

## The Correct 4-Layer Architecture for Movement Knowledge

### Layer 1 — Form (Drawing/Character Galaxy)

Already exists. Glyphs for "u","p" → "up", arrow symbols, etc.

### Layer 2 — Meaning (Word Galaxy) — UNIVERSAL direction stars

Each direction is a meaning-centric star. One concept = all languages.

| Star ID | Concept | Surface Forms | Displacement |
|---------|---------|---------------|-------------|
| `word_direction_above` | upward / above / north | en: above, up, north, upward; pt: acima, norte, para cima | `[0, -1]` |
| `word_direction_below` | downward / below / south | en: below, down, south, downward; pt: abaixo, sul, para baixo | `[0, +1]` |
| `word_direction_left` | leftward / west | en: left, west, leftward; pt: esquerda, oeste | `[-1, 0]` |
| `word_direction_right` | rightward / east | en: right, east, rightward; pt: direita, leste | `[+1, 0]` |
| `word_direction_centered` | centered / aligned / at target | en: centered, aligned, at target; pt: centralizado, alinhado | `[0, 0]` |

These stars carry `metadata.displacement` (the procedural meaning of the direction).

### Layer 3 — Rules (Grammar Galaxy) — Movement rules with RPN programs

These are universal spatial reasoning rules, NOT benchmark rules:

| Rule ID | Rule | RPN Program |
|---------|------|-------------|
| `grammar_spatial_move_toward_above` | "When target is above, move up" | `CURRENT_ROW TARGET_ROW SUB NEG THRESHOLD_GT ACTION_MOVE_UP` |
| `grammar_spatial_move_toward_below` | "When target is below, move down" | `TARGET_ROW CURRENT_ROW SUB POS THRESHOLD_GT ACTION_MOVE_DOWN` |
| `grammar_spatial_move_toward_left` | "When target is left, move left" | `CURRENT_COL TARGET_COL SUB NEG THRESHOLD_GT ACTION_MOVE_LEFT` |
| `grammar_spatial_move_toward_right` | "When target is right, move right" | `TARGET_COL CURRENT_COL SUB POS THRESHOLD_GT ACTION_MOVE_RIGHT` |
| `grammar_spatial_interact_centered` | "When at target, interact" | `TARGET_REACHED ACTION_PERFORM` |

Each rule has `word_refs` pointing to the Layer 2 meaning stars.
Each rule has `reality_refs` pointing to the Layer 3 Reality Galaxy action atoms.

### Layer 4 — Meta-Rules (Tool Galaxy) — Navigation strategies

| Meta-Rule ID | Strategy |
|-------------|----------|
| `meta_spatial_seek_target` | "Given a grid with object and goal, detect offset, apply matching direction rule" |
| `meta_spatial_recovery_undo` | "When action loop detected (same action N times, no progress), undo" |

These are already approximately right in the current tool entries — just rename from `meta_arc3_*`.

### Reality Galaxy — Action atoms with full RPN triple

The atoms from `action_primitives_bootstrap.py` become Reality.jsonl entries:

```json
{
  "id": "reality_action_move_up",
  "name": "Move Up — spatial translation",
  "domain": "spatial_action",
  "category": "translation",
  "content": "Move one unit in negative-Y direction",
  "visual_rpn": "0 0 DRAW_MOVE 0 -1 DRAW_LINE DRAW_STROKE",
  "behavior_rpn": "y RECALL dy RECALL - y STORE",
  "law_rpn": "y RECALL y_min RECALL GTE",
  "metadata": {
    "displacement": [0, -1],
    "action_type": "spatial_translation",
    "inverse": "reality_action_move_down",
    "surface_forms": {"en": "move up", "pt": "mover para cima"},
    "query_anchor": "move up upward above north spatial translation direction"
  }
}
```

(Same pattern for move_down, move_left, move_right, perform, click, undo, diagonal,
reach, grab, hold, release, walk_to, teleport, look_at — ALL 15 atoms from the bootstrap file.)

---

## What Codex Must Do

### Step 1: Remove bootstrap from Knowledgeverse init

In `knowledgeverse.py`, remove the two lines added in the unstaged diff:

```python
# DELETE these lines from __init__:
self.reality_galaxy = RealityGalaxy(...)
self.reality_action_node_ids = bootstrap_spatial_actions(...)
```

And remove the corresponding imports:
```python
# DELETE:
from knowledge3d.cranium.action_primitives_bootstrap import bootstrap_spatial_actions
from knowledge3d.cranium.reality_galaxy import RealityGalaxy
```

This fixes the startup hang. The bootstrap content moves to Galaxy JSONL files below.

### Step 2: Convert action atoms to Reality.jsonl entries

Create a one-time population script (or extend `arc3_knowledge_builder.py` → rename to
`spatial_knowledge_builder.py`) that writes the 15 action atoms from
`action_primitives_bootstrap.py` as proper Reality.jsonl entries with:
- Universal IDs: `reality_action_move_up` (not `atom:action:move_up`)
- All three RPN programs: `visual_rpn`, `behavior_rpn`, `law_rpn`
- Surface forms + displacement in metadata
- `query_anchor` for trigram embedding alignment

These are PERSISTENT. Written once to Reality.jsonl. Loaded at boot with all other Galaxy entries.

### Step 3: Rename Word stars from benchmark-specific to universal

Replace the 5 entries in Word.jsonl:

| Old ID (DELETE) | New ID |
|----------------|--------|
| `word_arc3_navigate_above` | `word_direction_above` |
| `word_arc3_navigate_below` | `word_direction_below` |
| `word_arc3_navigate_left` | `word_direction_left` |
| `word_arc3_navigate_right` | `word_direction_right` |
| `word_arc3_navigate_centered` | `word_direction_centered` |

Each new entry:
- `domain: "spatial_direction"` (not `"spatial_navigation"`)
- `category: "direction_meaning"` (not `"direction_word"`)
- Carries `metadata.displacement: [0, -1]` (the procedural meaning)
- Carries `metadata.behavior_rpn` from the action atom
- `query_anchor`: "above upward north up direction spatial movement" (universal, no "arc3")
- `word_refs`: ["above", "north", "top", "up", "upward"] (natural language symlinks)
- `reality_refs`: ["reality_action_move_up"] (symlink to the action atom)

### Step 4: Rename Grammar rules from benchmark-specific to universal

In `arc3_knowledge_builder.py` (rename file to `spatial_knowledge_builder.py`):

| Old ID | New ID |
|--------|--------|
| `arc3_nav_move_up` | `grammar_spatial_move_toward_above` |
| `arc3_nav_move_down` | `grammar_spatial_move_toward_below` |
| `arc3_nav_move_left` | `grammar_spatial_move_toward_left` |
| `arc3_nav_move_right` | `grammar_spatial_move_toward_right` |
| `arc3_nav_perform` | `grammar_spatial_interact_centered` |
| `arc3_nav_click` | `grammar_spatial_click_coordinates` |
| `arc3_nav_undo` | `grammar_spatial_undo_recovery` |

Each rule:
- `domain: "spatial_reasoning"` (not `"arc3_interactive"`)
- `word_refs: ["word_direction_above"]` (universal symlink)
- `reality_refs: ["reality_action_move_up"]` (action atom symlink)
- RPN programs stay the same — they're already correct
- `query_anchor` drops "arc3": "object above center top north navigate game frame move up direction spatial"

### Step 5: Update domain_hint routing

The ARC3 benchmark already passes `domain_hint="arc3_interactive"`. But the Grammar rules
are now `domain: "spatial_reasoning"`. The routing should match on SPATIAL CONTENT in the
query, not on a domain name. The `query_anchor` tokens ("above center", "direction",
"spatial") will naturally score high against ARC3 frame queries that contain position tokens.

In `benchmarks/arc_agi_3.py`, the `ARC3_ROUTE_GALAXIES` should be:
```python
ARC3_ROUTE_GALAXIES = ["Drawing", "Grammar", "Tool", "Reality", "Word"]
```
(Add "Word" and "Reality" — the direction stars and action atoms live there.)

### Step 6: Direct decode — label as transitional I/O decode

The direct decode at `_answer_arc_query()` line 4047 reads position from the encoded
query text. Keep it for now — it's I/O decoding, not reasoning. But rename the comment:

```python
# Transitional I/O decode: reads position result from _frame_to_query_text() encoding.
# Target replacement: TRM navigates Galaxy → direction Word star → RPN execution → action.
```

When the TRM game loop is wired (E.25+), this block is deleted. The TRM will navigate
to `word_direction_above` directly via Galaxy similarity, execute `behavior_rpn`, emit action.

### Step 7: Verify

```bash
# 1. Compile
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  python3 -m py_compile knowledge3d/knowledgeverse/knowledgeverse.py

# 2. Init timing (must be < 10s, no bootstrap hang)
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  python3 -c "
import time, os; os.environ['CUDA_VISIBLE_DEVICES']='0'
t=time.time()
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
kv=Knowledgeverse(); print(f'Init: {time.time()-t:.1f}s')
"

# 3. Re-run builder (now spatial_knowledge_builder.py or updated arc3_knowledge_builder.py)
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  python3 knowledge3d/knowledgeverse/arc3_knowledge_builder.py

# 4. ARC3 probe (10 steps)
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 K3D_DEVICE_PIPELINE=1 K3D_TRM_SHADOW=1 K3D_TRM_NAVIGATE=1 \
  python scripts/run_arc3_agent.py --game-id re86-4e57566e --max-actions 10

# 5. Full benchmark
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python scripts/run_full_benchmark.py \
    --storage-root /K3D/Knowledge3D.local \
    --synthetic-count 10 --mmlu-count 50 --gsm8k-count 10 --lhe-count 10 --arc3-count 5
```

---

## Why This Architecture Is Correct

**Meaning-centric:** "Above" is a concept. One star. All languages point to it. All tasks
that need "above" find the same star. ARC3 doesn't own it.

**Procedural:** The meaning of "move up" IS the RPN program `y RECALL dy RECALL - y STORE`.
Form + Meaning unified. The TRM executes meaning, not string labels.

**Persistent:** Stars are in the House (JSONL on disk). Loaded into Galaxy at boot.
Available from the start. No bootstrap, no lazy init, no runtime computation of knowledge.

**4-layer compliant:**
- L1 Form: glyphs, arrow visual (Drawing/Character galaxy)
- L2 Meaning: direction concept with displacement + surface forms (Word galaxy)
- L3 Rules: spatial reasoning RPN programs (Grammar galaxy) + action RPN triple (Reality galaxy)
- L4 Meta-Rules: navigation strategies (Tool galaxy)

**Symlink compliant:** Grammar rules → Word stars → Reality atoms. No duplication.

**Sovereignty compliant:** RPN programs execute on GPU. TRM navigates Galaxy to find them.
The direct decode is transitional I/O — clearly labeled, deleted when TRM is wired.

**Universal:** The same direction stars serve ARC3, House navigation, physics sims,
robotics, any spatial task. We never name knowledge by its consumer.

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Remove `RealityGalaxy` + `bootstrap_spatial_actions` from init and imports |
| `knowledge3d/knowledgeverse/arc3_knowledge_builder.py` | Rename IDs to universal; add Reality.jsonl entries from action atoms; rename Word stars |
| `benchmarks/arc_agi_3.py` | Add "Word" and "Reality" to `ARC3_ROUTE_GALAXIES` |

## Files NOT to Touch

| File | Why |
|------|-----|
| `knowledge3d/cranium/action_primitives_bootstrap.py` | Keep as reference — its CONTENT is correct, just wrong delivery mechanism |
| `scripts/run_arc3_agent.py` | Already correct |
| Direct decode in knowledgeverse.py | Already present, label as transitional |

---

## Success Criteria

- [ ] `Knowledgeverse()` init completes in < 10 seconds (no bootstrap hang)
- [ ] No `RealityGalaxy` or `bootstrap_spatial_actions` in `Knowledgeverse.__init__`
- [ ] Word.jsonl: direction stars named `word_direction_*` (no `word_arc3_*`)
- [ ] Grammar.jsonl: rules named `grammar_spatial_*` (no `arc3_nav_*`)
- [ ] Reality.jsonl: 15 action atoms with `visual_rpn`, `behavior_rpn`, `law_rpn`
- [ ] ARC3 probe: `program_type == "gpu_arc3_navigation_rule"` on ≥ 8/10 steps
- [ ] Full benchmark: all 5 suites emit result files within 60 seconds of start
- [ ] Existing tests pass after ID renames
