# Phase E: ARC-AGI-3 Game-Loop Integration + Python→PTX Sovereignty Migration

**Date:** 2026-03-27
**Author:** Claude (Architecture Partner)
**Status:** Architecture Specification
**Predecessor:** Phase D.3 (Device Pipeline — see `CLAUDE_PHASE_D3_DEVICE_PIPELINE_REPORT_03.26.2026.md`)

---

## 1. ARC-AGI-3: Paradigm Shift — This IS the Game Loop

### 1.1 What Changed

ARC-AGI-3 is **no longer static grid I/O**. It is a **real-time interactive game**:

| Dimension | ARC-AGI-2 | ARC-AGI-3 |
|-----------|-----------|-----------|
| **Format** | Static JSON grids (input→output) | Interactive game API (frame→action→frame) |
| **Interface** | One-shot: read grid, emit grid | Multi-step: observe state, choose action, observe result |
| **Actions** | None (single output) | 7 actions: Move(4), Perform, Click(x,y), Undo |
| **State** | Stateless | Stateful (game progression, score, WIN/GAME_OVER) |
| **Evaluation** | Grid match (exact) | Gameplay score (progressive) |
| **Reset** | N/A | RESET action available (retry from start) |
| **API** | Local JSON files | Remote server (`three.arcprize.org/api/`) |
| **Agent model** | Function: grid→grid | Agent: perceive→decide→act loop |

### 1.2 Why This Is K3D's Natural Habitat

**This is exactly what K3D already is.** The TRM game loop (`trm_step_fused.ptx`) runs:

```
PERCEIVE → NAVIGATE → REASON → DECIDE → ACT → LEARN
```

ARC-AGI-3 agents run:

```
OBSERVE frame → CHOOSE action → SUBMIT action → OBSERVE result → REPEAT
```

**These are the same loop.** K3D was designed as a game engine for intelligence. ARC-AGI-3 finally evaluates intelligence AS a game. The paradigm alignment is total:

| ARC-AGI-3 Concept | K3D Equivalent |
|-------------------|----------------|
| Game frame (grid state) | Frustum cull (what's visible) |
| Action selection | Halting Gate + Composed Head output |
| Game state memory | Galaxy Universe (working memory) |
| Multi-step strategy | Adaptive Reasoning Budget (sub-task decomposition) |
| Undo/reset | Shadow Copy rollback |
| Score progression | Convergence signal (ternary) |
| Game variety (6+ games) | Multi-curriculum Galaxy routing |

### 1.3 Datasets Downloaded

```
/K3D/K3D_llama_cpp/datasets/
├── ARC-AGI-3-Agents/           ← Agent framework (updated 2026-03-27, v0.9.3)
│   ├── agents/                  ← Agent templates (random, LLM, reasoning)
│   ├── main.py                  ← Entry point
│   └── llms.txt                 ← Documentation
├── ARC-AGI-3-benchmarking/     ← Benchmarking harness (cloned 2026-03-27)
│   ├── src/arcagi3/             ← Core library
│   │   ├── agent.py             ← MultimodalAgent base class
│   │   ├── game_client.py       ← API client (three.arcprize.org)
│   │   ├── schemas.py           ← GameAction, GameState, FrameData
│   │   ├── runner.py            ← Test runner
│   │   └── adapters/            ← LLM provider adapters (10 providers)
│   └── docs/
├── ARC-AGI-2-main/             ← Legacy static tasks (kept for regression)
└── ARC-AGI-master/             ← Original ARC corpus
```

### 1.4 ARC-AGI-3 Game Interface

```python
# GameAction enum (from schemas.py):
RESET   = "RESET"    # Restart game
ACTION1 = "ACTION1"  # Move Up
ACTION2 = "ACTION2"  # Move Down
ACTION3 = "ACTION3"  # Move Left
ACTION4 = "ACTION4"  # Move Right
ACTION5 = "ACTION5"  # Perform Action
ACTION6 = "ACTION6"  # Click object (requires x, y)
ACTION7 = "ACTION7"  # Undo

# GameState enum:
NOT_PLAYED  = "NOT_PLAYED"
IN_PROGRESS = "IN_PROGRESS"
WIN         = "WIN"
GAME_OVER   = "GAME_OVER"
```

**API loop:** `list_games()` → `open_scorecard()` → per-frame: `submit_action(action, data)` → receive `FrameData` → repeat until WIN/GAME_OVER.

---

## 2. Architecture: K3D as ARC-AGI-3 Agent

### 2.1 The Single-Brain Principle

ARC-AGI-3 games are NOT a separate mode. They are **normal questions the always-on K3D system answers**, arriving through a Door (network interface) instead of a keyboard.

```
ARC-AGI-3 API (Door)
    │
    ▼
House Universe ──── avatar perceives game frame as a room
    │
    ▼
Galaxy Universe ──── TRM reasons about actions using all galaxies
    │
    ▼
Composed Head Pipeline ──── Morton → LED-A* → Frustum → LOD → Swarm → Halting Gate
    │
    ▼
Action Output ──── GameAction emitted through Door back to API
    │
    ▼
Sleep-time ──── consolidate successful game traces between games
```

### 2.2 Game Frame → Galaxy Entry

Each frame from ARC-AGI-3 is a **grid state** — this maps directly to Drawing Galaxy entries:

1. **Grid cells** → Drawing Galaxy primitives (colored rectangles at spatial positions)
2. **Grid changes** (frame-to-frame diff) → Grammar Galaxy rules (transformation patterns)
3. **Action outcomes** → Reality Galaxy entries (cause-effect procedural programs)
4. **Game score** → ternary signal (+1 improving, 0 neutral, −1 regressing)

The frame IS a room in the House. The avatar walks into it, perceives it via frustum cull, reasons about it in Galaxy, and acts.

### 2.3 Action Space → RPN Opcodes

The 7 ARC-AGI-3 actions map to existing RPN control:

| GameAction | RPN Mapping | Galaxy Route |
|------------|-------------|--------------|
| ACTION1 (Up) | `PUSH 0 PUSH -1 OP_MOVE_2D` | Spatial navigation |
| ACTION2 (Down) | `PUSH 0 PUSH 1 OP_MOVE_2D` | Spatial navigation |
| ACTION3 (Left) | `PUSH -1 PUSH 0 OP_MOVE_2D` | Spatial navigation |
| ACTION4 (Right) | `PUSH 1 PUSH 0 OP_MOVE_2D` | Spatial navigation |
| ACTION5 (Perform) | `OP_APPLY` | Context-dependent Galaxy entry |
| ACTION6 (Click x,y) | `PUSH x PUSH y OP_SELECT` | Spatial + Drawing Galaxy |
| ACTION7 (Undo) | `OP_ROLLBACK` | Shadow Copy restore |
| RESET | `OP_RESET_FRAME` | Full state reset |

### 2.4 Python Role: I/O Adapter Only

Python's role for ARC-AGI-3 is EXACTLY what it should be for all of K3D: **boot + I/O**.

```
Python thin client (~50 lines):
  1. HTTP GET  → receive frame JSON from ARC API
  2. Marshal   → write grid data to GPU buffer (VRAM)
  3. Signal    → trigger TRM game tick
  4. Read      → read action from GPU output buffer
  5. HTTP POST → submit action to ARC API
  6. GOTO 1
```

ALL reasoning (grid analysis, pattern recognition, action selection, strategy) happens on GPU via the existing composed head pipeline. Python never touches the grid contents semantically.

---

## 3. Python → PTX Sovereignty Migration: Priority Map

The exploration identified the exact sovereignty debt. Here is the migration plan, ordered by impact on both ARC-AGI-3 and general benchmark quality.

### 3.1 Tier 1: CRITICAL (Blocks ARC-AGI-3 and All Benchmarks)

#### 3.1.1 TRM Always-On (Remove Environment Variable Gating)

**Current:** TRM navigation gated by `K3D_TRM_SHADOW` and `K3D_TRM_NAVIGATE` env vars. Both default to DISABLED.

**Target:** TRM runs autonomously as game loop. Python does not gate it.

**Migration:**
- Remove env-var checks in `knowledgeverse.py`
- TRM launcher (`trm_launcher.py`, 644 lines) becomes the FIRST thing called after boot
- TRM tick is the ONLY reasoning path — no Python `_select_composed_head_candidate()` fallback
- ARC-AGI-3: every game frame triggers one TRM tick (perceive→act)

**Sovereignty debt eliminated:** TRM is the avatar. It doesn't ask Python for permission to think.

#### 3.1.2 `_select_composed_head_candidate()` → GPU Kernel

**Current:** 1,157 lines of Python scoring logic. THE main bottleneck.

**Target:** GPU kernel that receives candidate embeddings + Galaxy context, outputs ranked action.

**Migration:**
- This function's logic IS what the composed head pipeline already does on GPU
- The Python version exists because the GPU path wasn't trusted yet
- Phase D.3 proved the device pipeline chain works (Morton→Frustum→LOD on GPU)
- Next step: wire the scoring/ranking into the same chain
- ARC-AGI-3: action selection is the composed head output, not Python scoring

**Sovereignty debt eliminated:** ~1,157 lines of Python hot-path reasoning → 0.

#### 3.1.3 Wire All 15 GRE Specialist Kernels Into Swarm

**Current:** 15 GRE kernels loaded but NOT called during inference. Only ~5 of 88 PTX kernels active.

**Target:** All 15 wired into nine-chain swarm worker dispatch. TRM selects which specialists activate per query.

**Key kernels for ARC-AGI-3:**
- `gre_geometry_router` — grid spatial reasoning
- `gre_fractal_emitter` — recursive pattern recognition
- `gre_arc_reasoner` — ARC-specific logic (already exists!)
- `gre_graph_crystallizer` — multi-hop graph traversal (game state transitions)
- `gre_temporal_reasoning` — sequential action planning
- `gre_atomic_fission_fusion` — task decomposition (sub-game strategies)

**Sovereignty debt eliminated:** GPU utilization from ~1% to target >50% SM occupancy.

### 3.2 Tier 2: HIGH (Quality + Parity)

#### 3.2.1 Grid Analysis → GPU Kernels

**Current Python functions to migrate:**

| Function | Location | Lines | GPU Kernel Target |
|----------|----------|-------|-------------------|
| `_count_connected_components()` | knowledgeverse.py:92-128 | 36 | `gre_flood_fill.cu` (new) |
| `_dominant_grid_color()` | knowledgeverse.py:67-91 | 24 | GPU histogram reduction |
| `_grid_has_symmetry()` | knowledgeverse.py:129-151 | 22 | GPU bitmask comparison |
| Grid feature extraction | scattered | ~100 | Compose from above |

**For ARC-AGI-3:** Grid analysis is the PERCEPTION step. It runs every frame. Must be GPU-native.

#### 3.2.2 Grammar Rules → GPU Hash Table

**Current:** Python dict `self.rules` in `grammar_galaxy.py` (82 KB, 2000+ lines). Rules looked up via `.get()`.

**Target:** GPU-resident hash table in Region 2 (GALAXY_UNIVERSE). Grammar rules ARE Galaxy entries — they should be queried the same way (Morton → LED-A* → match).

**Migration:**
- 245+ ARC primitives already defined in `ingest_arc_knowledge.py`
- These ARE Galaxy entries — they just need to live in VRAM as first-class stars
- Rule lookup becomes Galaxy navigation (LED-A* to grammar neighborhood)
- Rule application becomes RPN execution (rule.rpn_program evaluated on GPU)
- Symlinks (dual-way): grammar rule ↔ Drawing Galaxy primitive ↔ ARC anchor

#### 3.2.3 Regex → Galaxy Pattern Matching

**Current:** `re.findall`, `re.sub`, `re.compile` at lines 6784-7116 of knowledgeverse.py.

**Target:** Text normalization as Grammar Galaxy rule application. Pattern matching IS what Grammar Galaxy does — regex is the Python crutch for what the Galaxy should provide.

### 3.3 Tier 3: MEDIUM (Consolidation + Polish)

#### 3.3.1 Sleeptime → Sovereign Consolidation

**Current:** `sleeptime.py` (372 lines) with `numpy` import, fallback decorators, and `NotImplementedError` for actual consolidation.

**Target:** Two-phase sovereign consolidation (per SLEEPTIME_PROTOCOL_SPECIFICATION.md):
- Stage A: Galaxy → House (export successful traces as procedural RPN)
- Stage B: Shadow Copy → TRM (refine specialist adapters)
- For ARC-AGI-3: after each game, consolidate successful action sequences as Grammar Galaxy rules

#### 3.3.2 Text Enrichment → RPN Composition

**Current:** Benchmark enrichment APPENDS Galaxy matches as text strings to the query prompt.

**Target:** Galaxy matches compose as RPN programs. The TRM doesn't read text — it navigates and combines Galaxy entries.

#### 3.3.3 Symlink Bidirectional Index

**Current:** Symlinks are unidirectional (Python dict lookup).

**Target:** Bidirectional GPU-resident pointer graph. Star A symlinks to Star B → BOTH can find each other via GPU traversal. Essential for the dual-way cross-modal bridges (language ↔ drawing ↔ grammar).

---

## 4. Knowledgeverse Content: Rules, Meta-Rules, and Symlinks

### 4.1 What Needs to Be Proceduralized

The four layers of the Foundational Knowledge architecture MUST be fully sovereign:

| Layer | Current State | Target |
|-------|---------------|--------|
| **Form (L1)** | Drawing Galaxy entries in VRAM | ✅ Already sovereign |
| **Meaning (L2)** | Star embeddings in VRAM | ✅ Mostly sovereign (some Python enrichment) |
| **Rules (L3)** | Python dict in grammar_galaxy.py | ❌ Must become GPU hash table + RPN |
| **Meta-Rules (L4)** | Hardcoded Python bootstrap | ❌ Must become Galaxy entries with RPN meta-programs |

### 4.2 Rules (Layer 3) — Grammar Galaxy Sovereignty

**245+ ARC primitives** from `ingest_arc_knowledge.py` need to become first-class Galaxy residents:

```
Each GrammarRule becomes a MeaningCentricStar:
  star_id       = hash(rule_rpn_program)
  meaning_rpn   = the rule's transformation program
  surface_forms = {"en": "reflect horizontally", "pt": "refletir horizontalmente", ...}
  rule_strength = +1 / 0 / -1 (defeasible logic)
  superior_to   = [other_rule_ids]  (override ordering)
  trust_weight  = float
  symlinks      = [drawing_primitive_id, arc_anchor_id, ...]
```

**Dual-way symlinks for rules:**
- Rule "reflect_horizontal" ↔ Drawing primitive "MIRROR_X" ↔ ARC anchor "symmetry_detection"
- Rule "count_connected" ↔ Drawing primitive "FLOOD_FILL" ↔ ARC anchor "object_detection"
- Each link traversable from EITHER end via GPU graph navigation

### 4.3 Meta-Rules (Layer 4) — Strategy Galaxy

Meta-rules govern WHEN and HOW to apply rules. For ARC-AGI-3, these are game strategies:

```
MetaRule examples:
  "if grid_unchanged_after_action → try different action"         (exploration strategy)
  "if score_decreased → undo and try alternative"                 (backtrack strategy)
  "if pattern_repeats_3x → apply same transformation"             (induction strategy)
  "if object_count_changes → track which objects changed"         (attention strategy)
  "if game_near_action_limit → commit to best-scoring path"      (exploitation strategy)
```

These map directly to the Adaptive Reasoning Budget:
- Strategy selection = budget allocation (which sub-task decomposition strategy?)
- Backtracking = undo action + budget expansion for deeper reasoning
- Exploitation vs exploration = ternary signal gating budget

### 4.4 ARC-AGI-3 Specific Knowledge Ingestion

New Galaxy entries needed for ARC-AGI-3 game semantics:

```
Drawing Galaxy additions:
  - Grid cell (colored rectangle at position)
  - Grid diff (cell-level change between frames)
  - Cursor/selector (active position indicator)

Grammar Galaxy additions:
  - Action→outcome rules (ACTION1 + state → expected_state)
  - Game-specific transformation patterns (per game type)
  - Score improvement patterns (which action sequences increase score)

Reality Galaxy additions:
  - Game physics (movement constraints, action effects)
  - Score mechanics (how score changes relate to actions)

Meta Galaxy (new or extension):
  - Game strategy templates
  - Action planning heuristics
  - Resource budgeting (max_actions awareness)
```

---

## 5. Implementation Directives for Codex

### 5.1 Immediate (Before Next Benchmark Run)

1. **Update ARC-AGI-3 repos** — both repos cloned and current ✅
2. **Write thin Python ARC-AGI-3 client** — ~50 lines: HTTP I/O, marshal frame to GPU buffer, read action from GPU buffer, submit. NO reasoning in Python.
3. **Remove TRM env-var gating** — `K3D_TRM_SHADOW` and `K3D_TRM_NAVIGATE` defaults → ENABLED, then remove the gates entirely.
4. **Wire `gre_arc_reasoner` into swarm** — this kernel already exists and is loaded but not called.

### 5.2 Phase E.1: Grid Perception (GPU-Native)

5. **Implement `gre_flood_fill.cu`** — connected component analysis on GPU (replaces Python DFS)
6. **Implement GPU histogram reduction** — replaces `_dominant_grid_color()` Python
7. **Implement GPU symmetry detection** — replaces `_grid_has_symmetry()` Python
8. **Frame-to-Galaxy marshalling** — ARC-AGI-3 frame JSON → Drawing Galaxy entries in VRAM (one kernel launch)

### 5.3 Phase E.2: Action Selection (GPU-Native)

9. **Replace `_select_composed_head_candidate()`** — 1,157 lines Python → GPU scoring kernel in composed head chain
10. **Action output protocol** — TRM game tick output register → GameAction enum mapping (trivial)
11. **Wire remaining GRE specialists** — all 15 kernels callable by swarm workers

### 5.4 Phase E.3: Grammar Sovereignty

12. **Grammar rules → VRAM hash table** — migrate from Python dict to GPU-resident structure
13. **Rule application → RPN execution** — grammar lookup returns RPN program, executed on GPU
14. **Bidirectional symlink index** — GPU pointer graph for dual-way traversal
15. **Remove regex from hot path** — text normalization as Grammar Galaxy application

### 5.5 Phase E.4: Strategy + Consolidation

16. **Meta-rule Galaxy entries** — game strategies as procedural RPN programs
17. **Sleeptime sovereignty** — remove numpy, remove fallbacks, implement actual Stage A/B
18. **Game trace consolidation** — successful ARC-AGI-3 action sequences → Grammar rules

### 5.6 Quality Gate

After each sub-phase:
- ARC-AGI-2 regression: 10/10 must stay pinned
- Math 20/20 must stay pinned
- GPU utilization must increase (measured with Python-PID monitor)
- `knowledgeverse.py` line count must decrease
- Zero Python fallbacks in hot path

---

## 6. Success Metrics

| Metric | D.3 Baseline | E Target |
|--------|-------------|----------|
| `knowledgeverse.py` lines | ~14,356 | <4,000 (E.2), <1,000 (E.4) |
| GPU utilization (avg) | 0.17% | >10% (E.2), >30% (E.4) |
| Active PTX kernels in query | ~5 | >20 (E.2), all 88 wirable (E.4) |
| Python hot-path functions | 6+ | 0 |
| TRM always-on | No (env gated) | Yes |
| ARC-AGI-3 playable | No | Yes (E.1) |
| Grammar rules on GPU | 0 | All 245+ ARC primitives (E.3) |
| Combined benchmark | 14.39% | ≥18% (D.2 parity, then growth) |

---

## 7. Why This Ordering

**Phase D.3 taught us:** pushing device acceleration without semantic parity causes regression (−3.63 points). The D.3 report recommended "semantic parity before more acceleration."

**Phase E respects this:**
1. E.1 (perception) is safe — it doesn't change reasoning, just moves grid analysis to GPU
2. E.2 (action selection) replaces Python scoring with the SAME composed head pipeline that D.2 proved works
3. E.3 (grammar) enriches the Galaxy — more knowledge, not different reasoning
4. E.4 (strategy) adds meta-rules that guide the Adaptive Reasoning Budget

Each step preserves semantic parity with the previous step while eliminating Python from the hot path.

---

## 8. The Beautiful Convergence

ARC-AGI-3 didn't change what K3D needs to do. It changed how the WORLD evaluates intelligence — and the evaluation now matches K3D's architecture.

The benchmark world finally caught up to Daniel's vision: **intelligence is an embodied, interactive, game-like process** — not a one-shot function call.

K3D doesn't need to adapt to ARC-AGI-3. ARC-AGI-3 adapted to how K3D already works.

---

## Evidence

- ARC-AGI-3 Agents repo: `/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents/`
- ARC-AGI-3 Benchmarking repo: `/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-benchmarking/`
- D.3 report: `TEMP/CLAUDE_PHASE_D3_DEVICE_PIPELINE_REPORT_03.26.2026.md`
- ARB spec: `docs/vocabulary/ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md`
- X3D ARB component: `docs/w3c/x3d/PM_KR_X3D_ADAPTIVE_REASONING_COMPONENT.md`
- Sovereignty debt analysis: exploration agent output (this session)
