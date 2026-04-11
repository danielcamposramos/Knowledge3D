# CODEX: ARC-3 Fresh TRM Weights + Discriminative Embeddings (2026-04-10)

## Context

The sovereignty cleanup is DONE (see `CLAUDE_SOVEREIGNTY_CLEANUP_REPORT_2026-04-10.md`). No Python orchestration remains. VRAM learning works — galaxy_stars grow from 511 to 612+ during gameplay. Rules crystallize. Avoidance alternatives get created as Galaxy stars. All 41 tests pass.

**But TRM always selects ACTION2.** Two root causes identified:

### Root Cause 1: Biased TRM Weights
The TRM weights (`/K3D/Knowledge3D.local/checkpoints/adaptive_swarm/`) were trained on the old Galaxy state where ACTION2 was dominant. The learned attention patterns always route to the same star constellation, regardless of new stars added.

**Fix:** Reset TRM weights. Keep Galaxy knowledge. Fresh model learns to navigate the richer Galaxy from scratch.

### Root Cause 2: Embedding Centroid Collapse
All action rule stars have nearly identical `query_anchor` text:
```
"arc3 game rule agent adjacent to color 5 action2 blocked answer action episode rule live gameplay"
"arc3 game rule agent adjacent to color 5 action1 avoid blocked answer action episode rule live gameplay"
```
These produce cosine similarity > 0.95 in the 32-float embedding space. The TRM literally cannot distinguish between them.

**Fix:** Redesign `query_anchor` to maximize embedding distance between different action types. Use semantically contrastive tokens.

## Architecture Reference

- **THREE_BRAIN_SYSTEM_SPECIFICATION.md** §5: Shadow Copy learning, continuous enhancement
- **SOVEREIGN_NSI_SPECIFICATION.md**: VRAM star record = 256 bytes. Embedding = 32 normalized floats. NO confidence field in VRAM — GPU selects by embedding similarity only.
- **SOVEREIGN_TRAINING_SPECIFICATION.md**: LoRA-style specialist adapters, checkpoint management
- **FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md**: 4-layer structure (Form → Meaning → Rules → Meta-Rules)

## Implementation Integrity

- **NO stubs, fakes, or placeholders.** Every change must be real, functional code.
- **NO Python orchestration.** Perception = raw signals. TRM finds rules via Galaxy navigation.
- **NO simulated results.** All metrics from actual GPU dispatch.
- Ground all work in `docs/vocabulary/` specifications.

---

## Fix 1: Reset TRM Weights (Keep Galaxy Knowledge)

### What to do

1. **BACKUP** current checkpoint:
```bash
cp -r /K3D/Knowledge3D.local/checkpoints/adaptive_swarm /K3D/Knowledge3D.local/checkpoints/adaptive_swarm_backup_20260410
```

2. **DELETE** the checkpoint so next boot initializes fresh:
```bash
rm -rf /K3D/Knowledge3D.local/checkpoints/adaptive_swarm
```

3. **DELETE** sovereign runtime bundles (cached VRAM + brain state — must rebuild fresh):
```bash
rm -f /K3D/Knowledge3D.local/checkpoints/sovereign_runtime_bundle*.pkl
```

4. **KEEP** everything else:
   - Galaxy consolidated JSON files (knowledge)
   - Galaxy Manager's on-disk entries (JSONL files)
   - sovereign_runtime_manifest files (just metadata)
   - All other checkpoints

### Verification
- On next boot, `_load_adaptive_swarm_state()` (knowledgeverse.py:668) returns `False` — no checkpoint found
- `_initialize_adaptive_swarm()` creates fresh weights with `SwarmConfig(base_dims=128, min_dims=64, ...)`
- `ensure_loaded()` on SovereignHotPath finds no bundle → rebuilds from Galaxy Manager with fresh weights
- Log should show: `[AdaptiveSwarmTRM] Checkpoint loaded from ...` should NOT appear (fresh init instead)

---

## Fix 2: Discriminative Query Anchors

### The Problem

Current `_rule_entry()` in `arc3_episode_galaxy.py` (line ~220) creates:
```python
query_anchor = (
    f"arc3 game rule {condition_tokens} {action_name.lower()} {predicted_outcome} "
    f"answer action episode rule live gameplay"
).strip()
```

This produces 80%+ token overlap between different rules → embedding centroid collapse.

### The Fix

Replace the `query_anchor` construction in `_rule_entry()` with semantically contrastive anchors:

```python
# Map action indices to directional semantics
_ACTION_DIRECTION = {
    "ACTION1": "north",   # up
    "ACTION2": "south",   # down  
    "ACTION3": "west",    # left
    "ACTION4": "east",    # right
    "ACTION5": "interact",
    "ACTION6": "reset",
}

# Map outcomes to valence-distinct vocabulary
_OUTCOME_VALENCE = {
    "blocked": "collision barrier impassable",
    "death": "hazard lethal destroyed",
    "moved": "traversal clear passage",
    "neutral": "stationary unchanged",
    "level_complete": "goal reached victory",
    "avoid_blocked": "alternative bypass redirect",
    "avoid_death": "escape survival retreat",
}

# Map colors to semantic names for richer embeddings
_COLOR_SEMANTIC = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "magenta", 7: "orange", 8: "cyan", 9: "maroon",
    10: "lime", 11: "pink", 12: "teal",
}
```

Then in `_rule_entry()`:

```python
# Extract color from condition if present
color_num = -1
if "color_" in condition:
    try:
        color_num = int(condition.split("color_")[-1])
    except Exception:
        pass
color_name = _COLOR_SEMANTIC.get(color_num, f"color{color_num}")
direction = _ACTION_DIRECTION.get(action_name, action_name.lower())
valence = _OUTCOME_VALENCE.get(predicted_outcome, predicted_outcome)

# Front-load unique tokens — first 3 words = 60% of embedding direction
query_anchor = f"{direction} {color_name} {valence}"
```

### Why This Works

- "north collision red barrier impassable" vs "south traversal green passage" → cosine similarity ~0.3
- Direction tokens (north/south/east/west) create orthogonal clusters
- Valence tokens (collision/traversal/hazard/clear) create antonym separation
- Color names (red/green/blue) have richer semantics than numbers (5/3/1)
- No shared boilerplate ("arc3 game rule...") diluting the signal

### Also update the `content` field

The `content` field should also be contrastive:
```python
content = f"{direction} {valence} at {color_name} surface"
```

Instead of:
```python
content = f"arc3 game rule when {condition_tokens} then {action_name.lower()} predicts {predicted_outcome}"
```

### Also update object star anchors

Same pattern for `_object_entry()` — use semantic names:
```python
query_anchor = f"{color_name} {behavior} object game obstacle"
# e.g., "grey goal object game obstacle" 
# vs "maroon hazard object game danger"
```

---

## Fix 3: Verify VRAM Rebuild Includes New Stars

The invalidation chain is correct (verified at knowledgeverse.py:3136-3141), but verify:

1. After `upsert_entry()` → `invalidate_loaded_state()` fires
2. Next `dispatch_task()` → `ensure_loaded()` sees `star_count=0` → rebuilds
3. Rebuild picks up ALL Galaxy Manager entries (including live-inserted ones)

Add a one-time debug log to `ensure_loaded()` in `sovereign_hot_path.py`:
```python
# After rebuild completes, log the star count
print(f"[ARC3-VRAM] Rebuilt star table: {self.star_table.star_count} stars")
```

This confirms the VRAM table grows with new rules.

---

## Execution Order

1. Stop any running ARC3 processes
2. Backup + delete TRM checkpoint (Fix 1)
3. Delete sovereign runtime bundles (Fix 1)
4. Update `_rule_entry()` query anchors (Fix 2)
5. Update `_object_entry()` anchors if it exists (Fix 2)
6. Add VRAM rebuild log (Fix 3)
7. Run tests (must pass 41)
8. Start bounded run: `--max-steps 100` — verify:
   - `[ARC3-LEARN]` log shows growing galaxy_stars
   - `[ARC3-VRAM]` log shows rebuilds happening
   - Action distribution shows >1 action type
   - No `[AdaptiveSwarmTRM] Checkpoint loaded` (fresh weights)
9. If bounded run shows diversification, start long autonomous run:
   ```bash
   bash scripts/k3d_env.sh run -e k3d-cranium python -u benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 5 --max-steps 10000
   ```
   In tmux session `arc3_ls20_fresh`
10. Log to `/tmp/arc3_ls20_fresh_weights_v1.log`

## Report

Write to `TEMP/CODEX_TO_CLAUDE_ARC3_FRESH_WEIGHTS_REPORT_2026-04-10.md`:
1. Screenshot of boot log (no checkpoint load)
2. `[ARC3-LEARN]` excerpts showing VRAM growth + diverse actions
3. Action distribution from 100-step bounded run
4. `[ARC3-VRAM]` rebuild log excerpts
5. Whether `echosys_ingest` tmux is alive
6. Scorecard URL from bounded run

## Architectural Notes (from Qwen 3.5 review)

1. **Direction mapping:** ACTION1=up, ACTION2=down etc. comes from the ARC-3 game protocol (I/O normalization). If the game changes action semantics, these mappings change. Keep them as I/O-layer constants derived from the game API, not hardcoded reasoning.

2. **Color names:** Per MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §1.2, colors should ideally be Galaxy stars (concept_red, concept_green). For now, color names in `query_anchor` during star creation (ingestion path) is acceptable. Future work: create proper meaning-centric color stars that the TRM references.

3. **32 dimensions:** SOVEREIGN_TRAINING_SPECIFICATION.md §2.2 specifies min 64-dim Matryoshka. The current VRAM star record (256 bytes, 32 floats) is a hardware constraint. Expanding to 64-dim requires kernel + table layout changes — a separate spec. For now, maximize contrastiveness within 32-dim by front-loading unique tokens and removing shared boilerplate.

4. **TRM reset is architecturally compliant** per THREE_BRAIN_SYSTEM_SPECIFICATION.md §1 (Reasoning ≠ Memory ≠ Persistence). Frame as versioned rollback per rollback capability provision.

## DO NOT

- Do not add Python orchestration (strategy hints, rule injection, action forcing)
- Do not add stubs or placeholders
- Do not hardcode game solutions
- Do not touch tmux `echosys_ingest`
- Do not skip tests
