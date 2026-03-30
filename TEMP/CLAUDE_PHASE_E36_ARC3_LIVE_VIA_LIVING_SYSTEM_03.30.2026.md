# Claude — Phase E.36: ARC3 Live Server via Living System

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — validates local 20/20 against real server

---

## Context

ARC3 local benchmark scores 20/20 = 100% (first move 20/20). The next step is
validating this against the live ARC3 server at `https://three.arcprize.org`.

**CRITICAL**: This MUST run through the living Knowledgeverse, NOT as an isolated
one-off script that loads a fraction of knowledge. K3D is a living system — the
full Galaxy must be loaded, warm-boot caches active, all 278K entries in VRAM.

---

## Available Infrastructure

**API Key:** `/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt` (36 chars, confirmed)

**Available Games** (from `GET /api/games`):

| game_id | title | tags | baseline_actions (per level) |
|---------|-------|------|------------------------------|
| r11l-aa269680 | R11L | click | 7, 28, 30, 20, 37, 45 |
| ft09-0d8bbf25 | FT09 | — | 17, 19, 15, 21, 65, 26 |
| cd82-fb555c5d | CD82 | keyboard_click | 41, 8, 30, 21, 19, 17 |
| sb26-7fbdac44 | SB26 | keyboard_click | 18, 16, 15, 15, 31, 24, 17, 17 |
| tn36-ab4f63cc | TN36 | click | 23, 22, 26, 37, 25, 56, 61 |
| (20 more available) | | | |

**Runners:**
- `scripts/run_arc3_agent.py` — single game via `run_live_arc3()`
- `scripts/run_arc3_session.py` — multi-game session with inter-game consolidation

**Agent:** `benchmarks/arc_agi_3.py::K3DARC3Agent` wrapping `kv.execute_task()`

---

## What Codex Must Do

### Step 1: Ensure Full Living System is Active

The runner at `scripts/run_arc3_agent.py` line 163 creates a fresh
`Knowledgeverse()` with no arguments. This MUST use the full warm-boot path:
- All cached GPU buffers loaded
- All 278K Galaxy entries in VRAM
- CSR graph loaded from cache
- Warm-boot checkpoint loaded

Verify that the Knowledgeverse init in the runner uses `storage_root="/K3D/Knowledge3D.local"`
so it picks up all cached state and full knowledge.

### Step 2: Run Session Against Multiple Games

Use `scripts/run_arc3_session.py` with 3-5 games covering different interaction types:

```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 scripts/run_arc3_session.py \
    --game-id r11l-aa269680 \
    --game-id ft09-0d8bbf25 \
    --game-id cd82-fb555c5d \
    --game-id sb26-7fbdac44 \
    --game-id tn36-ab4f63cc \
    --max-actions-per-game 80 \
    --api-url https://three.arcprize.org
```

### Step 3: Report Results

Log everything:
- Per-action JSONL (already implemented)
- Scorecard URLs (already returned by API)
- Final summary with state, levels_completed, actions per game
- Post-session consolidation results

---

## Success Criteria

- [ ] Full Knowledgeverse loaded (278K+ entries, not partial)
- [ ] At least 3 games played against live server
- [ ] All JSONL logs saved to `/K3D/Knowledge3D.local/logs/`
- [ ] Scorecard URLs captured for each game
- [ ] Post-session sleep consolidation executed
- [ ] Report: levels_completed per game, total actions, win/loss state

---

## Note on WINE Direction

The current `K3DARC3Agent` class is transitional (E.35 WINE spec covers its removal).
For THIS run, use the existing agent — it works (20/20 local). The WINE migration
happens after we have live server results to compare against.
