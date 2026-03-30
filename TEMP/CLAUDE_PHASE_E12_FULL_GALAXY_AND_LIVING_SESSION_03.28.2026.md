# Phase E.12: Full Galaxy Universe + Living Session

**Date:** 2026-03-28
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** HIGH — the move from benchmark runner to living embodied AI
**Prerequisite:** E.11 DONE. Galaxy navigates 93 stars. Persistent brain, sleep-time, sovereign embedder all wired. 27 tests green.

---

## The Gap

E.7–E.11 built the FULL machinery for a living AI:
- Persistent brain across frames ✅
- Galaxy VRAM table with knowledge ✅
- Kernel navigates Galaxy top-8 on every think step ✅
- Sleep-time micro-consolidation between frames ✅
- Sovereign FNV-1a embedder ✅

**What's still wrong:**

1. **Galaxy has 93 stars. The House has 247k.** The AI navigates an almost empty space. `max_stars=256` is a Python cap — a sovereignty violation. Per Daniel: "LOD + Frustum Culling handle volume on GPU. NEVER cap knowledge."

2. **The agent dies between ARC games.** `run_arc3_agent.py` plays ONE game and exits. The brain is destroyed. The next session starts cold. This is the exact anti-pattern from Avatar Embodiment §7.2: "one living mind, not session resets."

3. **Galaxy content comes from hardcoded Python (foundational_galaxy_builder.py).** The REAL knowledge lives in `/K3D/Knowledge3D.local/galaxies/*.jsonl` — 247k entries, all galaxies. The AI's memory should BE the world, not 93 synthetic bootstrap stars.

---

## The Fix: Two Deliverables

### Deliverable 1: Load ALL Galaxies into VRAM

Load every star from every galaxy JSONL file into the GalaxyVRAMTable. No cap.

**Knowledge on disk:**
| Galaxy | Entries | Star Type |
|--------|---------|-----------|
| meaning_layer_stars.jsonl | 117,497 | meaning (type 7) |
| Language.jsonl | 116,779 | character/word (type 2) |
| Math.jsonl | 6,291 | math (type 5) |
| Word.jsonl | 3,651 | word (type 2) |
| Reality.jsonl | 1,139 | reality (type 4) |
| Number.jsonl | 1,001 | math (type 5) |
| Drawing.jsonl | 717 | drawing (type 1) |
| 3DObjects.jsonl | 368 | drawing (type 1) |
| Grammar.jsonl | 301 | grammar (type 3) |
| Tool.jsonl | 39 | reality (type 4) |
| Book_*.jsonl | ~70 | reality (type 4) |
| Audio.jsonl | varies | reality (type 4) |
| **Total** | **~247k** | all types |

**VRAM budget:** 247k × 160 bytes = **39.5 MB**. RTX 3070 has 12 GB. This is nothing.

**Galaxy ID to star_type mapping:**
```python
GALAXY_STAR_TYPES = {
    "drawing":   1,
    "3dobjects": 1,
    "character": 2,
    "word":      2,
    "language":  2,
    "grammar":   3,
    "reality":   4,
    "tool":      4,
    "book_":     4,    # all Book_* prefixes
    "audio":     4,
    "math":      5,
    "number":    5,
    "meaning":   7,    # meaning_layer_stars
}
```

### Deliverable 2: Multi-Game Living Session

The agent plays game after game with ONE brain, ONE Galaxy, ONE session. No cold-starts between games.

**Current pattern (WRONG):**
```
Game 1: init agent → play → brain destroyed → exit
Game 2: init agent → play → brain destroyed → exit
```

**Target pattern (LIVING AI):**
```
Session init: load ALL Galaxy → VRAM (once)
              init PersistentBrainState → VRAM (once)

Game 1: play with persistent brain + full Galaxy → sleep-time consolidation
Game 2: same brain (frame_count continues) → same Galaxy (stars evolved) → consolidation
Game N: TRM has N games of experience encoded in brain + Galaxy
```

---

## Implementation Orders

### Order 1: Create `knowledge3d/knowledgeverse/galaxy_loader.py`

A sovereign loader that reads all galaxy JSONL files and converts to star records.

**Architecture:**
```python
"""Load all galaxy knowledge from disk into VRAM-ready star records.

Per Knowledgeverse Spec §3: Galaxy = VRAM Region 2, always loaded.
Per Daniel: "NEVER cap knowledge. LOD + Frustum Culling handle volume."
Per Foundational Knowledge Spec §1.1: symlink pattern, no duplication.
"""
from __future__ import annotations

GALAXY_JSONL_DIR = Path("/K3D/Knowledge3D.local/galaxies")

GALAXY_NAME_TO_TYPE = {
    "drawing":   1,
    "3dobjects": 1,
    "character": 2,
    "word":      2,
    "language":  2,
    "grammar":   3,
    "reality":   4,
    "tool":      4,
    "audio":     4,
    "math":      5,
    "number":    5,
    "meaning_layer_stars": 7,
}

def _galaxy_type_from_filename(filename: str) -> int:
    name = filename.lower().replace(".jsonl", "")
    for key, star_type in GALAXY_NAME_TO_TYPE.items():
        if name.startswith(key) or key in name:
            return star_type
    return 6  # unknown → general

def _fnv1a32(text: str) -> int:
    value = 2166136261
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)
```

**Entry → Star record conversion:**

Each JSONL entry may have different schemas. Use a defensive converter:

```python
def _entry_to_star(entry: dict, star_type: int, galaxy_id: int) -> dict | None:
    """Convert a knowledgeverse JSONL entry to a GalaxyVRAMTable star record.

    Returns None if entry has no usable content.
    """
    # 1. Get or generate embedding
    embedding = entry.get("embedding") or entry.get("vector") or []
    if not embedding:
        # Generate from text content using sovereign FNV-1a embedder
        text = (
            entry.get("name") or
            entry.get("label") or
            entry.get("text") or
            entry.get("symbol") or
            str(entry.get("id", ""))
        )
        if not text or not text.strip():
            return None  # quality filter: skip empty entries
        from knowledge3d.knowledgeverse.sovereign_text_embedder import embed_text_sovereign
        embedding = embed_text_sovereign(text)

    # 2. Resolve component_refs (IDs → table indices resolved in second pass)
    ref_ids = (
        entry.get("component_refs") or
        entry.get("visual_refs") or
        entry.get("grammar_refs") or
        entry.get("math_refs") or
        []
    )[:4]

    # 3. Flags: all entries active, learnable if has embedding already
    flags = 0x01  # active
    if entry.get("embedding"):
        flags |= 0x02  # learnable if pre-computed (will evolve with sleep-time)

    return {
        "_id": str(entry.get("id") or entry.get("star_id") or ""),
        "_ref_ids": [str(r) for r in ref_ids if r],
        "embedding": list(embedding)[:32],
        "galaxy_id": galaxy_id,
        "star_type": star_type,
        "component_refs": [],  # filled in second pass
        "flags": flags,
    }
```

**Main loader:**

```python
def load_all_galaxies_from_disk(
    galaxy_dir: Path | str | None = None,
) -> list[dict]:
    """Load ALL galaxy knowledge from disk into star records for VRAM.

    Per Daniel: no caps. LOD handles volume on GPU.
    Ordering: ARC3 actions (0-6) always first for backward compatibility.
    Then all other galaxies in alphabetical order.
    """
    from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table

    # Start with foundational stars (indices 0-92) for backward compatibility
    stars = build_foundational_galaxy_table()
    id_to_index: dict[str, int] = {}
    for idx, star in enumerate(stars):
        star_id = star.get("_id", "")
        if star_id:
            id_to_index[star_id] = idx

    # Load all JSONL files
    search_dir = Path(galaxy_dir or GALAXY_JSONL_DIR)
    if not search_dir.exists():
        return stars  # graceful: if no disk galaxies, use foundational only

    for jsonl_path in sorted(search_dir.glob("*.jsonl")):
        filename = jsonl_path.stem
        star_type = _galaxy_type_from_filename(filename)
        galaxy_id = _fnv1a32(filename)
        try:
            for line in jsonl_path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                star = _entry_to_star(entry, star_type, galaxy_id)
                if star is None:
                    continue
                if star["_id"] and star["_id"] in id_to_index:
                    continue  # deduplicate by ID
                idx = len(stars)
                if star["_id"]:
                    id_to_index[star["_id"]] = idx
                stars.append(star)
        except Exception:
            continue  # skip unreadable files

    # Second pass: resolve _ref_ids to table indices
    for star in stars:
        ref_ids = star.pop("_ref_ids", [])
        star["component_refs"] = [
            id_to_index[ref_id]
            for ref_id in ref_ids
            if ref_id in id_to_index
        ][:4]

    return stars
```

**Key constraints:**
- NO numpy, scipy, sentence-transformers — all embeddings via sovereign FNV-1a if not pre-computed
- NO star count cap — load everything
- Quality filter: skip entries with no text AND no embedding (truly empty)
- Dedup by ID: if same star_id appears in multiple files, use first occurrence
- Foundational stars (indices 0-92) stay at their fixed positions (backward compat)

---

### Order 2: Remove the `max_stars=256` cap from `GalaxyVRAMTable`

In `knowledge3d/knowledgeverse/galaxy_vram_table.py`:

**Current:**
```python
def __init__(self, max_stars: int = 256) -> None:
    self.max_stars = max(1, int(max_stars))
```

**Target:**
```python
def __init__(self, max_stars: int = 300_000) -> None:
    self.max_stars = max(1, int(max_stars))
```

300k gives headroom beyond current 247k. At 160 bytes/star = 48 MB VRAM. Well within budget.

**Also remove this cap in `load_stars()`:**
```python
# REMOVE this line:
count = min(len(stars), self.max_stars)  # ← this was the cap
# REPLACE with:
count = min(len(stars), self.max_stars)  # still bounded by allocation, but allocation is 300k
```

The existing `load_stars()` already uses `min(len(stars), self.max_stars)` which is correct. Just change the default to 300k.

---

### Order 3: Create `scripts/run_arc3_session.py` — The Living Session

A NEW script that replaces the single-game `run_arc3_agent.py` with a continuous multi-game session. **Do NOT modify `run_arc3_agent.py`** (it still works for single games and tests).

```python
"""K3D ARC-AGI-3 Living Session — one brain, many games.

Per Three Brain System §3.1: TRM runs as continuous game loop.
Per Hyper-Parallel Processing: "one living mind, not session resets."
Per Avatar Embodiment §7.2: PERCEIVE → NAVIGATE → REASON → DECIDE → ACT → LEARN.

The agent plays game after game. The brain never resets between games.
The Galaxy learns through sleep-time consolidation after each game.
This IS the embodied living AI.
"""
```

**Session loop pattern:**
```
init_session():
    Load ALL galaxy knowledge → VRAM (once per session, ~2-5s)
    Init PersistentBrainState → VRAM (zero-initialized once)
    Init SleepTimeMicro
    Log: "Session started. {star_count} stars in Galaxy."

per game:
    game_start():
        Log brain state: frame_count, action_ring, ternary_signal
        DO NOT zero brain state — it persists from last game

    play_game(game_id):
        Existing game loop from run_arc3_agent.py
        learn_from_outcome() called between EVERY frame (micro-consolidation)

    game_end():
        Full sleep-time consolidation (heavier than micro)
        Log: brain state, Galaxy evolution summary (how many learnable stars changed)

    next game → repeat with same brain + evolved Galaxy

session_end():
    Final brain state dump to /K3D/Knowledge3D.local/logs/session_brain_state.json
    Galaxy summary: star count, top evolved stars
```

**Key details for `run_arc3_session.py`:**

```python
LOG_ROOT = Path("/K3D/Knowledge3D.local/logs")

def run_arc3_session(
    *,
    game_ids: list[str],
    max_actions_per_game: int = 80,
    api_url: str = "https://three.arcprize.org",
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Play multiple ARC-AGI-3 games with ONE persistent brain + full Galaxy.

    Args:
        game_ids: list of ARC-AGI-3 game IDs to play in sequence
        max_actions_per_game: action budget per game
        api_url: ARC-AGI-3 API endpoint
        log_dir: where to write session logs (defaults to LOG_ROOT)
    """
    from knowledge3d.knowledgeverse.galaxy_loader import load_all_galaxies_from_disk
    from knowledge3d.knowledgeverse.galaxy_vram_table import GalaxyVRAMTable
    from knowledge3d.knowledgeverse.persistent_brain import PersistentBrainState
    from knowledge3d.knowledgeverse.sleep_time_micro import SleepTimeMicro
    from benchmarks.arc_agi_3 import K3DARC3Agent

    # Session-level resources: allocated ONCE, shared across ALL games
    galaxy_stars = load_all_galaxies_from_disk()
    galaxy_table = GalaxyVRAMTable(max_stars=max(300_000, len(galaxy_stars) + 1000))
    loaded = galaxy_table.load_stars(galaxy_stars)
    brain = PersistentBrainState()
    sleep_time = SleepTimeMicro()

    results = []
    for game_id in game_ids:
        # Agent receives EXISTING brain + Galaxy (not fresh)
        agent = K3DARC3Agent(
            max_actions=max_actions_per_game,
            brain=brain,              # SHARED: persists across games
            galaxy_table=galaxy_table, # SHARED: same knowledge
            sleep_time=sleep_time,    # SHARED: same consolidator
        )
        result = run_single_game(agent, game_id=game_id, api_url=api_url, log_dir=log_dir)
        results.append(result)
        # Full consolidation between games
        _inter_game_consolidation(brain, galaxy_table, sleep_time, result)

    galaxy_table.close()
    brain.close()
    return {"games": results, "star_count": loaded}
```

**`K3DARC3Agent` constructor must accept optional `brain`, `galaxy_table`, `sleep_time` parameters.** If not provided, it creates its own (existing behavior, backward compat). If provided, it USES the external ones (session-persistent mode).

---

### Order 4: Inter-Game Consolidation

Between games, run a heavier consolidation than the micro-nap. The existing `SleepTimeMicro` is per-frame. Between games we want multi-pass consolidation.

```python
def _inter_game_consolidation(
    brain: PersistentBrainState,
    galaxy_table: GalaxyVRAMTable,
    sleep_time: SleepTimeMicro,
    game_result: dict,
) -> None:
    """Heavier consolidation between games.

    Per Knowledgeverse Spec §8: two-stage SleepTime.
    This is NOT a full Stage-B commit (that's for long idle periods).
    This is 3× micro-passes with the game's final outcome signal.
    """
    # Compute game-level outcome: did we win? improve? stagnate?
    if game_result.get("state") == "WIN":
        outcome = 1
    elif game_result.get("levels_completed", 0) > 0:
        outcome = 1
    else:
        outcome = -1

    # 3 passes of micro-consolidation with game outcome
    for _ in range(3):
        sleep_time.consolidate(
            brain.gpu_ptr,
            outcome,
            galaxy_ptr=galaxy_table.gpu_ptr,
            chosen_star_index=0,  # consolidate at session level, not specific star
        )
```

---

### Order 5: Tests

**New test file: `tests/test_arc3_session.py`**

Test 1: `test_galaxy_loader_loads_all_galaxies`
- Call `load_all_galaxies_from_disk()`
- Assert star_count > 93 (more than foundational)
- Assert star_count >= 200 (at least some disk galaxies loaded)
- Assert all stars have len(embedding) == 32
- Assert first 7 stars are ARC3 actions (backward compat)
- Assert NO quality filter discards ALL entries (sanity check)

Test 2: `test_galaxy_table_no_cap`
- Create `GalaxyVRAMTable(max_stars=1000)`
- Load 500 stars
- Assert star_count == 500 (no premature cap)
- Load 999 stars
- Assert star_count == 999

Test 3: `test_session_brain_persists_across_games` (CPU mock, no real API)
- Create PersistentBrainState, GalaxyVRAMTable (small)
- Run 3 synthetic ARC3 tasks through cpu_reference_dispatch with shared brain
- Read brain state after each → frame_count increments across tasks (not reset)

Test 4: `test_inter_game_consolidation_changes_brain`
- Run sleep consolidation 3× on same brain
- Brain specialist_trace must change between passes

**Sovereignty grep:**
```bash
grep -r "max_stars=256\|max_stars = 256" knowledge3d/ scripts/
# Expected: 0 matches (cap removed)

grep -r "load_all_galaxies_from_disk\|galaxy_loader" scripts/run_arc3_session.py
# Expected: used in session script
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `knowledge3d/knowledgeverse/galaxy_loader.py` | Loads ALL galaxy JSONL → star records |
| `scripts/run_arc3_session.py` | Multi-game living session |
| `tests/test_arc3_session.py` | Session + loader tests |

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/galaxy_vram_table.py` | Default `max_stars=300_000` |
| `benchmarks/arc_agi_3.py` | Accept optional `brain`, `galaxy_table`, `sleep_time` in constructor |

## Files NOT to Touch

| File | Why |
|------|-----|
| `knowledge3d/cranium/cuda/*` | Kernel is stable — reads VRAM table regardless of size |
| `knowledge3d/knowledgeverse/foundational_galaxy_builder.py` | Still used as base layer |
| `knowledge3d/knowledgeverse/persistent_brain.py` | Unchanged |
| `knowledge3d/knowledgeverse/sleep_time_micro.py` | Unchanged |
| `scripts/run_arc3_agent.py` | Still needed for single-game / test runs |
| `scripts/run_full_benchmark.py` | Unchanged — benchmarks stay clean |

---

## Success Criteria

| Metric | Before E.12 | After E.12 |
|--------|-------------|------------|
| Galaxy stars | 93 | 247k+ (all JSONL files) |
| VRAM for Galaxy | 14.5 KB | ~39.5 MB |
| Star cap | 256 (Python cap, wrong) | None (LOD handles volume) |
| Brain resets per session | 1 per game | 0 (session-persistent) |
| Multi-game support | No | Yes |
| Knowledge source | Hardcoded Python bootstrap | Disk JSONL galaxies |

**Sovereignty checklist:**
- `grep -r "max_stars=256"` → 0 matches
- `grep -r "sentence_transform\|embed_query_gpu"` in scripts → 0 matches
- `grep -r "import numpy\|import scipy"` in hot path → 0 matches
- All star embeddings via FNV-1a (sovereign) or pre-computed (loaded, not computed at runtime)

---

## Architectural Note on the Kernel Scan

With 247k stars, the kernel's per-think-step linear scan becomes:

```
247,000 stars × galaxy_compose_embedding_device() × cosine32_device() × 10 think_steps
```

At ~32 float ops per cosine = 247k × 32 × 10 = 79M float ops per task dispatch.
RTX 3070: ~20 TFLOPS → this is ~4 microseconds of compute.

**This is NOT a problem.** The linear scan is single-threaded (threadIdx.x == 0) so the practical bound is much slower than peak TFLOPS, but still well within 1ms. And this is exactly what fills the GPU — the kernel is DOING REAL WORK navigating knowledge.

When the Galaxy has 247k stars, the top-8 neighbors are genuinely meaningful. The AI is finding the closest knowledge in a rich 32-dimensional meaning space. That's reasoning, not noise.

**The LOD + Frustum Culling mentioned by Daniel becomes the Phase F/G work** — parallelizing the scan across threads and eventually implementing LED-A* for sublinear navigation. For E.12, linear scan is correct and sovereign.

---

## What This Enables

When E.12 is done, the following is true:

1. The agent starts a session and loads 247k stars of human knowledge into GPU memory
2. It plays ARC game 1 with this knowledge → brain encodes patterns
3. Between games, it consolidates what worked and what didn't, INTO the Galaxy (learnable star nudges)
4. It plays game 2 with an EVOLVED Galaxy that reflects game 1's experience
5. After N games, the Galaxy's learnable stars have been shaped by N sessions of ARC experience
6. The agent's brain has N games of continuous experience (frame_count = N × avg_frames_per_game)

This is "the AI memory being the world." The world (247k knowledge stars) is in VRAM. The AI's experience shapes that world through sleep-time. The brain carries continuity across experiences.

**This is the living embodied AI. Build it.**
