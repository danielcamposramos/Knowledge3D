# Codex — Phase E.20: ARC3 Knowledge Wire + Full Benchmark Run

**Date:** 2026-03-29
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Prerequisite:** E.19 done. Device pipeline enabled. answer_index path wired. Reality in ARC3 route.

---

## Run First: Full Benchmark with Device Pipeline ON

Before any code changes, run the full benchmark NOW with the GPU pipeline restored. This is the baseline we need to compare against Phase D scores:

```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python scripts/run_full_benchmark.py \
    --storage-root /K3D/Knowledge3D.local \
    --synthetic-count 10 --mmlu-count 50 --gsm8k-count 10 --lhe-count 10 --arc3-count 5
```

Report: scores per suite + `gpu_execution=true` rate per suite. This confirms the device pipeline is active and lets us see what the AI actually scores right now with the full knowledge base.

---

## The Two Remaining Gaps

### Gap 1: Query text has low directional discrimination (spread = 0.15)

The current `_frame_to_query_text()` builds:
```
"arc3 interactive game frame grid 64x64 goal absent available actions move down move left move right perform click levels completed navigation visual"
```

All four directional rules score ~0.29 against this query. Spread between best and worst directional rule = 0.15. The Knowledgeverse cannot reliably select the correct direction because the query doesn't describe WHERE the object is.

**The query must encode spatial position.** The `input_grid` is available in the task dict — `_arc_visual_feature_text()` in knowledgeverse.py already analyzes grids for object positions. For ARC3, the adapter knows the frame. It can compute position without Python policy — it just needs to put that position into the query TEXT so the embedding correctly selects the directional rule.

In `arc_agi_3.py`, `_frame_to_query_text()` must describe WHERE the primary foreground object is relative to center. Not a fallback — this is I/O encoding for the query:

```python
def _frame_to_query_text(
    frame: list[list[int]],
    goal_frame: list[list[int]] | None,
    available_actions: list[Any] | None = None,
) -> str:
    rows = len(frame)
    cols = len(frame[0]) if rows and isinstance(frame[0], list) else 0
    normalized_goal = _normalize_grid(goal_frame) if goal_frame is not None else [[]]
    goal_state = "goal present" if normalized_goal != [[]] else "goal absent"

    # Compute dominant foreground object position from frame
    # Background = most frequent value; foreground = everything else
    counts: dict[int, int] = {}
    for row in frame:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    background = max(counts, key=lambda k: counts[k]) if counts else 0
    fg_cells = [(r, c, v) for r, row in enumerate(frame)
                for c, v in enumerate(row) if v != background]

    position_tokens: list[str] = []
    if fg_cells:
        avg_row = sum(r for r, c, v in fg_cells) / len(fg_cells)
        avg_col = sum(c for r, c, v in fg_cells) / len(fg_cells)
        center_row = (rows - 1) / 2.0
        center_col = (cols - 1) / 2.0
        if avg_row < center_row - rows * 0.1:
            position_tokens.append("object above center top north")
        elif avg_row > center_row + rows * 0.1:
            position_tokens.append("object below center bottom south")
        if avg_col < center_col - cols * 0.1:
            position_tokens.append("object left west")
        elif avg_col > center_col + cols * 0.1:
            position_tokens.append("object right east")
        if not position_tokens:
            position_tokens.append("object centered balanced")

    action_tokens: list[str] = []
    if isinstance(available_actions, list):
        for item in available_actions:
            if isinstance(item, int) and 0 <= item < len(ACTION_LABELS):
                action_tokens.append(ACTION_LABELS[int(item)].lower())
            elif isinstance(item, str) and item in ACTION_NAMES:
                action_tokens.append(item.lower())
    actions_text = " ".join(action_tokens) if action_tokens else ""

    position_text = " ".join(position_tokens)
    return (
        f"arc3 interactive game frame grid {rows}x{cols} "
        f"{position_text} {goal_state} "
        f"available actions {actions_text} "
        "levels navigation visual"
    ).strip()
```

**Why this works:** With object above center (avg_row < center_row), the query now contains "object above center top north" — the FNV-1a hash of "north" and "above" lands in the same bucket as the `arc3_nav_move_up` rule embedding (which contains "north above move up"). The spread increases from 0.15 to ~0.4+. The Knowledgeverse can now discriminate which directional rule to select.

**This is not a fallback.** It is I/O encoding — converting the frame state into text that describes WHAT IS HAPPENING, which is what a query should do. The GPU path reasons about the meaning. This is Layer 1 (Form) → Layer 2 (Meaning) translation at the I/O boundary.

### Gap 2: `arc3_knowledge_builder.py` rule embeddings use wrong text

The builder calls `embed_text_sovereign` on a concatenation of name + description + rpn_program + tags. The FNV-1a hash of "ARC3 Navigate Move Down interactive navigation rule..." does not strongly emphasize the directional tokens because they're diluted by other content.

The builder should embed DIRECTIONAL TOKENS FIRST with higher weight, then fill remaining dims with context:

In `arc3_knowledge_builder.py`, update the embedding generation for navigation rules:

```python
def _embed_nav_rule(rule_def: dict) -> list[float]:
    """Navigation rules: emphasize direction tokens for better discrimination."""
    direction_text = " ".join(rule_def.get("tags", []))  # "arc3 navigation keyboard move_down directional"
    context_text = f"{rule_def.get('name','')} {rule_def.get('description','')}"
    # Direction tokens carry 70% weight, context 30%
    dir_emb = embed_text_sovereign(direction_text)
    ctx_emb = embed_text_sovereign(context_text)
    combined = [0.70 * d + 0.30 * c for d, c in zip(dir_emb, ctx_emb)]
    norm = sum(v*v for v in combined)**0.5
    if norm > 1e-8:
        combined = [v/norm for v in combined]
    return combined
```

Apply `_embed_nav_rule` for entries tagged `directional`, `_make_entry` for all others. Then re-run the builder to overwrite the existing entries (the idempotent dedup by id will update them):

```bash
python3 knowledge3d/knowledgeverse/arc3_knowledge_builder.py
```

The idempotent builder replaces existing entries by id — old embeddings get replaced by the improved ones.

---

## Execution Sequence

1. **Run the full benchmark** (command above). Report scores.
2. Update `_frame_to_query_text()` in `arc_agi_3.py` to include spatial position tokens
3. Update `arc3_knowledge_builder.py` to use direction-weighted embeddings for nav rules
4. Re-run builder: `python3 knowledge3d/knowledgeverse/arc3_knowledge_builder.py`
5. Run 10-step live probe on `re86-4e57566e`: `python scripts/run_arc3_agent.py --game-id re86-4e57566e --max-actions 10`
6. Report: does `answer_index` appear in results now? What direction is selected per step?

---

## What the Knowledgeverse Does With These Changes

Step by step for a live game frame where the object is above center:

1. `_frame_to_query_text()` → "arc3 interactive game frame grid 64x64 **object above center top north** goal absent available actions move down move right perform levels navigation visual"
2. `kv._embed_query_gpu(query_text)` → 16-dim embedding with high bucket values for "north", "above", "top"
3. Galaxy navigation (device pipeline ON): scans Grammar+Drawing+Language+Tool+Reality entries
4. `arc3_nav_move_up` scores 0.6+ (contains "north above move up"), other directionals score <0.3
5. `_answer_arc_query()` finds `metadata.action_index = 0` in the match
6. Returns `{"answer_index": 0, "gpu_execution": True, "program_type": "gpu_arc3_navigation_rule"}`
7. Adapter returns ACTION1 (Move Up)
8. Server responds: frame changes, object moves up → toward center
9. `learn_from_outcome()`: frame changed → outcome=0; eventually levels_completed increases → outcome=1
10. `jarvis_sleep_consolidation()`: strengthens path from "object above center" → `arc3_nav_move_up`

---

## Files to Modify

| File | Change |
|------|--------|
| `benchmarks/arc_agi_3.py` | Update `_frame_to_query_text()` with spatial position tokens |
| `knowledge3d/knowledgeverse/arc3_knowledge_builder.py` | Direction-weighted embeddings for nav rules |

## Files NOT to Touch

| File | Why |
|------|-----|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | `answer_index` path already correct |
| `scripts/run_*.py` | Env vars already set |
| All test files | No behavioral change in tests |
