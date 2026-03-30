# Claude -- Phase E.35: ARC3 as WINE → Tablet Proceduralization

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** ARCHITECTURAL DIRECTION -- shapes how ALL benchmarks interface with K3D

---

## Daniel's Direction

> "On the ARC3 interface with K3D, I would like to think of it like a WINE
> (Wine Is Not an Emulator) approach — 'proceduralize' to the tablet on the go,
> live if possible."

---

## The WINE Principle Applied to K3D

WINE translates Windows API calls to native POSIX equivalents at runtime.
It doesn't emulate Windows — it provides a **translation layer** that lets
Windows programs run natively on Linux.

K3D should do the same for benchmarks:

```
WINE:
  Windows app → Win32 API call → WINE translation → POSIX syscall → Linux kernel

K3D:
  ARC3 game frame → proceduralize → Tablet input → TRM game loop → Galaxy navigation
  MMLU question → proceduralize → Tablet input → TRM game loop → Galaxy navigation
  Human typing → proceduralize → Tablet input → TRM game loop → Galaxy navigation
```

**The benchmark doesn't know it's talking to the Tablet.** The Tablet doesn't know
it's receiving a benchmark. All inputs arrive the same way — proceduralized into
the Tablet's semantic format. The TRM handles them all through its normal
perceive→navigate→reason→decide→act loop.

---

## What This Means Architecturally

### Current State (Wrong)

```
ARC3 game
  ↓
benchmarks/arc_agi_3.py (K3DARC3Agent)
  ↓
knowledgeverse.query() with domain_hint="arc3_interactive"
  ↓
_answer_arc_query() (special ARC3 code path, line 4244)
  ↓
Direct decode from position tokens (transitional Python)
```

There's a **special code path** for ARC3. A different one for MMLU. Another for
GSM8K. Each benchmark has its own adapter, its own query format, its own result
extraction. This is the emulator approach — building separate code for each "OS."

### Target State (WINE Approach)

```
ARC3 game frame
  ↓
Thin WINE layer: proceduralize frame → Tablet semantic input
  ↓
Memory Tablet (spec: MEMORY_TABLET_SPECIFICATION.md §3.2)
  ↓
TRM perceives Tablet content (same as perceiving anything else in House)
  ↓
TRM game loop: navigate Galaxy → reason → decide → act
  ↓
TRM output → Tablet semantic output
  ↓
Thin WINE layer: Tablet output → ARC3 action submission
```

**The WINE layers are thin (~50 lines each).** They proceduralize external formats
into Tablet inputs and de-proceduralize Tablet outputs back. Everything in between
is the same living system that handles any interaction.

---

## Spec Grounding

### From MEMORY_TABLET_SPECIFICATION.md §2:

```
┌─────────────────────────────────────────┐
│  Memory Tablet (User Interface Layer)  │ ← ALL input arrives here
├─────────────────────────────────────────┤
│  Spatial UI (Houses, Rooms, Portals)   │
├─────────────────────────────────────────┤
│  Galaxy Universe (Knowledge Layer)     │
├─────────────────────────────────────────┤
│  Knowledgeverse (7-Region VRAM)        │
├─────────────────────────────────────────┤
│  Cranium (PTX Execution)               │
└─────────────────────────────────────────┘
```

The Tablet sits at the TOP. ARC3 frames, MMLU questions, human queries — they ALL
enter through the Tablet. No special paths below it.

### From MEMORY_TABLET_SPECIFICATION.md §3.2 (Dual-Client Perception):

```
Tablet Surface
  ↓
Procedural Content (RPN programs)
  ↓
Semantic Parser (TRM Navigation)
  ↓
AI sees: Navigation graph, references to Galaxy nodes
```

An ARC3 grid frame IS procedural content. A 5×5 grid with colored cells maps to
Drawing Galaxy primitives (RECT, COLOR) composed as an RPN program. The TRM
perceives this the same way it perceives any Tablet content — as references to
Galaxy nodes.

### From SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md (TRM Game Loop):

```
Perceive → Navigate → Reason → Decide → Act → Learn
```

One loop. One path. Every tick. ARC3 frames are just another thing to perceive.
The action output is just another thing to act. No special ARC3 logic inside the
loop.

### From KNOWLEDGEVERSE_SPECIFICATION.md §2.1:

> "Galaxy Universe is always loaded in VRAM"

The ARC3 grid frame doesn't need a special parser. The Drawing Galaxy already has
RECT, LINE, CIRCLE. The Grammar Galaxy already has transformation rules. The TRM
already navigates these. The proceduralization is: express the ARC3 frame as
references to existing Galaxy entries.

---

## The Proceduralization (ARC3 Frame → Tablet Input)

### ARC3 sends a game frame:
```json
{
  "grid": [[0, 0, 1], [0, 2, 0], [1, 0, 0]],
  "goal": [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
}
```

### WINE layer proceduralizes to Tablet input:
```
RPN program (Drawing Galaxy references):
  GRID 3 3                    → @drawing_grid_3x3
  CELL 0 0 COLOR_BLACK        → @drawing_color_0
  CELL 0 2 COLOR_BLUE         → @drawing_color_1
  CELL 1 1 COLOR_RED          → @drawing_color_2
  CELL 2 0 COLOR_BLUE         → @drawing_color_1
  GOAL_GRID 3 3
  CELL 0 0 COLOR_BLUE         → @drawing_color_1
  CELL 2 2 COLOR_BLUE         → @drawing_color_1
  TASK ARC_INTERACTIVE         → @grammar_rule_arc_transform
```

This is a normal RPN program composed of Galaxy symlinks. The TRM perceives it
through standard Tablet content parsing — no special ARC3 code.

### TRM navigates, reasons, decides, acts:
```
TRM output (Tablet semantic output):
  ACTION MOVE_OBJECT 0,2 → 0,0    → action_index = specific movement
```

### WINE layer de-proceduralizes:
```python
# ~10 lines: read TRM action output, map to ARC3 API format
action = {"type": "move", "from": [0, 2], "to": [0, 0]}
arc3_client.submit_action(action)
```

---

## What Changes

### Remove (Sovereignty Debt)
- `benchmarks/arc_agi_3.py::K3DARC3Agent` special class
- `_answer_arc_query()` special code path in knowledgeverse.py
- `_frame_to_query_text()` special text encoding
- `_derive_action_from_result()` special result extraction
- Any `domain_hint="arc3_interactive"` special routing

### Add (WINE Layers)
- `knowledge3d/tablet/arc3_wine.py` (~50 lines): proceduralize ARC3 frames → Tablet RPN
- `knowledge3d/tablet/mmlu_wine.py` (~30 lines): proceduralize MMLU questions → Tablet RPN
- `knowledge3d/tablet/gsm8k_wine.py` (~30 lines): proceduralize GSM8K problems → Tablet RPN

### Keep (Already Correct)
- TRM game loop (`trm_step_fused.ptx`)
- Nine-chain swarm dispatch
- Galaxy navigation (LED-A*, Morton, Frustum)
- Drawing Galaxy entries (RECT, LINE, COLOR)
- Grammar Galaxy transformation rules

---

## Execution Sequence

This is NOT urgent for running benchmarks today. The current benchmark runners work
(GPU at 100%, producing results). This is the ARCHITECTURAL direction for how
benchmarks integrate long-term.

1. **Now:** Let current benchmarks run, collect results, monitor
2. **E.35a:** Define the Tablet input RPN format for proceduralized external content
3. **E.35b:** Write `arc3_wine.py` WINE layer (frame → Tablet RPN → action)
4. **E.35c:** Wire Tablet input into TRM's normal perceive step
5. **E.35d:** Repeat for MMLU, GSM8K, LHE (each ~30 lines)
6. **E.35e:** Remove special benchmark code paths from knowledgeverse.py

---

## The Bigger Picture

Every external interface becomes a WINE layer:

| External System | WINE Layer | Tablet Format |
|----------------|------------|---------------|
| ARC3 game | `arc3_wine.py` | Grid → Drawing RPN + Grammar rules |
| MMLU question | `mmlu_wine.py` | Multiple choice → Word/Grammar RPN |
| GSM8K problem | `gsm8k_wine.py` | Word problem → Math/Grammar RPN |
| Human typing | `keyboard_wine.py` | Keystrokes → Character RPN |
| Voice input | `stt_wine.py` | Audio → Word RPN |
| MCP tool call | `mcp_wine.py` | Tool request → Tool Galaxy RPN |
| REST API | `rest_wine.py` | HTTP → Tablet RPN |

One living system. One Tablet. One TRM game loop. Many WINE layers.

"Wine Is Not an Emulator" — K3D doesn't emulate benchmark solvers. It proceduralizes
benchmark inputs into the same knowledge format it uses for everything else.

---

## Success Criteria (Long-Term, Not This Sprint)

- [ ] ARC3 frames proceduralized as Drawing Galaxy RPN on Tablet
- [ ] TRM perceives ARC3 frames through normal Tablet content path
- [ ] No `_answer_arc_query()` or `K3DARC3Agent` special code
- [ ] Same Tablet path handles ARC3, MMLU, GSM8K, human input
- [ ] WINE layers are each < 100 lines
- [ ] knowledgeverse.py loses benchmark-specific code paths
