# Christoph Viewer Task List

**Date:** 2026-03-22
**Focus:** Viewer / User Client — Visualize TRM Thinking, Galaxy View, Query Interface
**Division of Labor:** Christoph = Viewer/Visualizer. Claude+Codex = TRM/Brain Enhancement.

---

## 1. Setup & Build

### Build Environment

The viewer is a TypeScript/Vite/Three.js application. All source lives in `viewer/`.

```bash
# Option A: Use the pre-configured build env (recommended)
bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh

# Option B: Manual
cd viewer
npm install
npm run dev      # dev server with HMR (http://localhost:5173)
npm run build    # production build (tsc + vite build)
npm run test     # jest tests (22 test files)
```

### Dev Server

`npm run dev` starts Vite on `http://localhost:5173`. The viewer auto-connects to the K3D daemon via WebSocket on ports `8765`, `8787`, `8788`, `8789` (fallback cascade). Override with `?ws=ws://host:port` URL param or `VITE_K3D_WS_URL` env var.

### Developer UI

Append `?dev=1` to the URL to show the full developer panel (top-left). This reveals:
- Expert House selector
- Agent navigation controls
- Chat log with command buttons (Pause, Resume, Status, **Ask Thoughts**, Who am I?)
- Explain log (reasoning trace output)
- Dataset info, cache controls
- Coloring modes (position/cluster), LOD HUD, **Spatial CoT toggle**
- Tablet mode (AI/Human), visibility toggle

### House Scene

The viewer loads a GLB scene from `runtime_boot.json` (polled every 500ms during boot). The scene contains:
- **6 rooms:** Foyer, Library, Study, Workshop, Garden, Living Room
- **60+ nodes:** doors, shelves, books, displays, tools, instruments, tablet, holodesk, galaxy pod
- Navigation graph for keyboard movement (WASD + room transitions via doors)

---

## 2. Current Viewer Architecture

### Key Source Files

| File | Purpose |
|------|---------|
| `viewer/src/main.ts` | Entry point — scene setup, render loop, event wiring, WS connection |
| `viewer/src/chat.ts` | `ChatClient` — WebSocket client for daemon communication |
| `viewer/src/agent.ts` | `K3DAgent` — yellow sphere that navigates Galaxy graph, pathfinding, trail rendering |
| `viewer/src/tablet.ts` | `Tablet3D` — 3D tablet object with 19 apps (Console, Chat, Notes, RPN, Galaxy, Stats, Exams...) |
| `viewer/src/apps.ts` | All tablet app implementations (`TabletApp` interface) |
| `viewer/src/projection/galaxyPodProjector.ts` | Galaxy Pod — stellarium-mode point cloud of Galaxy entries (domain-colored) |
| `viewer/src/projection/holodeskProjector.ts` | HoloDesk — holographic wireframe projector for 3D objects and RPN programs |
| `viewer/src/projection/surface.ts` | `ProjectionSurface` — shared base for stellarium/holographic projection modes |
| `viewer/src/behavior/interpreter.ts` | Behavior dispatch: ROOM_ENTER, DOOR_TRAVERSE, OPEN_BOOK, DISPLAY, TABLET, etc. |
| `viewer/src/behavior/activator.ts` | `HouseActivator` — click-to-activate House nodes |
| `viewer/src/behavior/roomContext.ts` | `RoomContext` — tracks current room, emits room-change events |
| `viewer/src/behavior/domProjection.ts` | DOM projection — renders K3D content as HTML elements in 3D space |
| `viewer/src/behavior/contentRenderer.ts` | Renders content entries as overlay panels |
| `viewer/src/navigation/keyboardNav.ts` | WASD navigation, room transitions, door traversal |
| `viewer/src/navigation/minimap.ts` | Top-down room minimap |
| `viewer/src/navigation/roomLabels.ts` | Floating room name labels |
| `viewer/src/navigation/welcome.ts` | Welcome overlay on first visit |
| `viewer/src/roomCamera.ts` | `RoomCamera` — camera presets per room |
| `viewer/src/rpn/` | Browser-side RPN engine: `engine.ts`, `meshOps.ts`, `pathOps.ts`, `mat4Ops.ts`, `domOps.ts`, `audioOps.ts` |
| `viewer/src/extensions/smartGraph.ts` | `AISuggestionManager`, `DynamicLayerManager`, `LODRenderer`, `GridCulledPoints` |
| `viewer/src/shapes.ts` | Instanced star/branch rendering for Galaxy visualization |
| `viewer/src/cluster.ts` | k-means clustering for coloring modes |
| `viewer/src/loadK3D.ts` | Load Galaxy data (K3DRecord format) |
| `viewer/src/loadHouseScene.ts` | Parse GLB into HouseNode tree |
| `viewer/src/contentLoader.ts` | Load House content (entries, refs, surface forms) |
| `viewer/src/materials/` | House materials, lighting, atmosphere |

### WebSocket Protocol

The viewer communicates with the K3D daemon via JSON messages over WebSocket:

**Outgoing (viewer → daemon):**
```json
{ "type": "chat", "from": "user", "text": "/ask-thoughts" }
{ "type": "chat", "from": "user", "text": "/status" }
{ "type": "chat", "from": "user", "text": "What is photosynthesis?" }
```

**Incoming (daemon → viewer):**
```json
{ "type": "chat", "from": "system", "text": "..." }
{ "type": "command", "command": "goto", "target": "library" }
{ "type": "command", "command": "cot", "target": "{\"steps\":[...], \"waypoints\":[...]}" }
```

### Existing Thinking Infrastructure

Already partially built:
- **Spatial CoT Overlay** (`main.ts:1839`): `drawReasoningOverlay(payload)` renders colored spheres + polylines tracing TRM reasoning steps through Galaxy space. Color-coded: cyan=retrieve, yellow=compare, pink=synthesize, red/green=verify.
- **CoT Toggle** (`index.html:92`): checkbox to show/hide the spatial reasoning overlay.
- **Ask Thoughts Button** (`index.html:45`): sends `/ask-thoughts` via WS.
- **Explain Log** (`index.html:51-53`): div for displaying reasoning text.
- **`cotOverlay` variable** (`main.ts:44`): THREE.Group that holds the current reasoning visualization.

### Backend Thinking Trace

The Knowledgeverse already generates `thinking_trace` for every task (`knowledgeverse.py:2942`):
```
Specialist route: visual
Scanning Galaxy: Drawing, Language, Grammar (117497 entries)
Top match: entry[4521] "circle_primitive" (similarity=0.87, confidence=0.92)
RPN program: rpn_arc_transform_01
Galaxy read: answer_text from entry[4521]
```

This is wrapped in `<thinking>...</thinking>` XML and returned alongside every answer. The daemon already passes `thinking_trace` and `reasoning_trace` arrays in task responses.

---

## 3. Task List

### Task V1: Live Thinking Panel (Priority: HIGH)

**Goal:** Show TRM's `thinking_trace` in real-time as it processes any query (benchmark question, user question, or idle consolidation).

**What exists:** The `Explain` log div and `ConsoleApp` tablet app already display text. The `/ask-thoughts` command exists. But there's no **live streaming** of thinking during processing.

**What to build:**
1. **New WS message type: `thinking`** — The daemon should emit incremental thinking lines as TRM processes:
   ```json
   { "type": "thinking", "step": "Scanning Galaxy: Math, Grammar (117497 entries)", "phase": "retrieve", "specialist": "math", "task_id": "MMLU_Q1234" }
   ```
   *(Claude+Codex will add the daemon-side emission — Christoph handles the viewer-side reception.)*

2. **Thinking Panel UI** — A collapsible panel (bottom-right or side) that shows:
   - Current specialist being used (e.g., "math", "visual", "chat")
   - Each thinking step as it arrives (scrolling log)
   - Phase indicator: retrieve → compare → synthesize → verify → answer
   - Confidence bar for current match
   - Task type badge (ARC_TASK, MATH_TASK, MMLU_TASK, etc.)

3. **Wire into existing `ChatClient`** — Add `onThinking` handler alongside existing `onChat`/`onCommand`.

**Key files to modify:** `chat.ts` (add message type), `main.ts` (add panel + handler), new `thinkingPanel.ts` component.

---

### Task V2: Galaxy Live View During Benchmarks (Priority: HIGH)

**Goal:** When TRM is processing a benchmark, visually show WHICH Galaxy entries it's visiting in the Galaxy Pod stellarium.

**What exists:** `GalaxyPodProjector` renders a static point cloud of Galaxy entries (domain-colored, fibonacci-sphere layout). The `K3DAgent` (yellow sphere) can navigate to records.

**What to build:**
1. **Highlight active entries** — When a `thinking` message arrives naming an entry, pulse/highlight that star in the Galaxy Pod. Use a bright glow or size pulse to show "TRM is looking at THIS entry right now."

2. **Trace path visualization** — As TRM moves through Galaxy entries during a solve, draw a glowing trail connecting visited entries (similar to existing `drawReasoningOverlay` but integrated into the Galaxy Pod, not scene-space).

3. **Benchmark progress indicator** — Small HUD showing: "MMLU: 1106/4915 | Current: Q1234 | Specialist: chat | Galaxy: Math, Grammar"

4. **Real-time domain heatmap** — Color-weight the Galaxy Pod by domain activity. If TRM is heavily using Math Galaxy, those stars glow brighter.

**Key files to modify:** `galaxyPodProjector.ts` (add highlight/pulse methods), `main.ts` (wire WS events to projector), `shapes.ts` (instanced rendering for highlights).

---

### Task V3: Ask TRM Questions (Priority: MEDIUM)

**Goal:** Let users type a natural language question, have the daemon route it to TRM, and display the answer with full thinking trace.

**What exists:** The chat input and `/ask-thoughts` button exist. The daemon's `execute_task()` already returns answers + thinking traces for any query. The HUD chat at the bottom already displays chat messages.

**What to build:**
1. **Query input with thinking display** — When user types a question, it goes through WS to daemon. As TRM processes, thinking steps stream to the Thinking Panel (V1). When complete, the answer appears in chat.

2. **Visual journey** — While TRM answers, the Galaxy Pod (V2) highlights the entries being consulted. The spatial CoT overlay shows the reasoning path in 3D.

3. **Answer card** — Display the final answer in a styled card (not just chat text) showing:
   - The question
   - Answer
   - Confidence score
   - Specialist used
   - Number of Galaxy entries consulted
   - Expandable thinking trace

**Key files to modify:** `chat.ts` (ensure query routing), `main.ts` (answer card rendering), `apps.ts` (enhance ConsoleApp or create AnswerApp).

---

### Task V4: Benchmark Dashboard (Priority: MEDIUM)

**Goal:** Show live benchmark progress and historical scores.

**What to build:**
1. **Live progress bar** — During a benchmark run, show suite progress: "ARC: 2/42 | Math: 3/500 | GSM8K: 7/462 | LHE: 1/35 | MMLU: 1106/4915"
2. **Score ticker** — Running accuracy percentage per suite.
3. **Historical chart** — Load from health log files, show score trends over runs.
4. **Per-question detail** — Click a suite to see individual question results (correct/wrong, thinking trace, specialist used).

**New WS message type needed:**
```json
{ "type": "benchmark", "suite": "MMLU", "index": 1107, "total": 4915, "correct": true, "score": 0.2252 }
```

**Key files:** New `benchmarkDashboard.ts` component, wire to WS events in `main.ts`.

---

### Task V5: TRM State Visualization (Priority: MEDIUM)

**Goal:** Visualize what's happening inside the TRM's "brain" — specialist routing weights, swarm worker states, and contrastive learning progress.

**What to build:**
1. **Specialist wheel** — Circular diagram showing all specialists (visual, math, chat, grammar, code, ocr) with their current routing weights. Brighter = more active.
2. **Swarm worker display** — Show the nine-chain swarm: 9 boxes/circles, each showing which candidate they're evaluating, their confidence, and whether they've converged.
3. **Contrastive learning indicator** — After sleep-time, show which specialists got positive/negative training pairs. Green = improved, red = no signal.

**New WS message types needed:**
```json
{ "type": "trm_state", "specialists": { "math": 0.85, "visual": 0.12, ... }, "swarm": [...] }
{ "type": "sleep_update", "stage": "B", "specialist": "math", "positive_pairs": 12, "negative_pairs": 38 }
```

**Key files:** New `trmStateView.ts`, integrate into tablet as new app or as separate HUD panel.

---

### Task V6: Polish Existing UX (Priority: LOW — Do After V1-V3)

1. **Move dev panel to game-style HUD** — The `?dev=1` panel is functional but ugly. Rebuild key controls into the game-style HUD overlay (bottom bar). Keep advanced controls behind a toggle.
2. **Minimap enhancement** — Show current room highlighted, show TRM activity indicator on minimap (which room the TRM is "thinking about").
3. **Room transition animations** — Smooth camera lerp when keyboard-navigating between rooms (instead of snap).
4. **Tablet visual polish** — The canvas-rendered tablet text is functional but basic. Consider HTML overlay for better typography.
5. **Mobile/touch support** — Touch navigation for tablets/phones.

---

## 4. Recommended Task Order

```
V1 (Thinking Panel) → V2 (Galaxy Live View) → V3 (Ask TRM Questions) → V4 (Benchmark Dashboard) → V5 (TRM State) → V6 (Polish)
```

V1 and V2 are the most impactful — they're what Daniel means by "show the thinking tag." Once those work, the viewer becomes a window into TRM's mind, not just a static scene viewer.

---

## 5. Communication Protocol Summary

For V1-V5, the viewer needs new WS message types from the daemon. Claude+Codex will implement the daemon-side emission. Christoph implements the viewer-side reception and rendering.

**New message types Christoph should expect:**

| Type | Direction | Purpose |
|------|-----------|---------|
| `thinking` | daemon → viewer | Incremental thinking step during any task |
| `benchmark` | daemon → viewer | Per-question benchmark progress |
| `trm_state` | daemon → viewer | Specialist routing weights, swarm state |
| `sleep_update` | daemon → viewer | Sleep-time contrastive learning results |
| `galaxy_highlight` | daemon → viewer | Which Galaxy entries TRM is visiting |

Christoph can start building the viewer-side handlers and UI with mock data while waiting for the daemon-side implementation.

---

## 6. Testing

22 test files exist in `viewer/tests/`. Run with:
```bash
cd viewer && npm run test
```

For new components, follow the existing pattern: one test file per module in `viewer/tests/`. Use `jest-environment-jsdom` for DOM tests.

---

## 7. Key Principles

- **The viewer is the user client** — it shows what the TRM is doing, not what the TRM is. The TRM lives on GPU; the viewer is a window.
- **House = the 3D world** — navigate rooms, click objects, interact with tablet. This is the shared spatial reality.
- **Galaxy Pod = TRM's brain visualized** — the stellarium in the bathroom shows what's inside the TRM's head.
- **HoloDesk = focused projection** — holographic wireframe display for inspecting individual objects/programs.
- **Thinking = the key differentiator** — showing HOW the TRM reasons (not just WHAT it answers) is what makes this viewer unique.
