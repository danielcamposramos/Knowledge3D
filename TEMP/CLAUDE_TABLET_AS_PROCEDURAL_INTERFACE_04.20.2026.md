# Tablet as Procedural Interface — Umbrella Architecture Spec

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-20
**Status:** Umbrella spec — ties four WINE surfaces to one contract
**Supersedes:** nothing (ties together in-flight specs)
**Cross-links:**
- `TEMP/CLAUDE_SMART_PROCEDURALIZER_SPEC_V2_04.20.2026.md` (PROCEDURALIZE WINE)
- `TEMP/CLAUDE_DUAL_PATH_INGESTION_AND_DISPATCH_WIRING_04.20.2026.md` (dispatch wiring)
- `TEMP/CLAUDE_CODEX_OP_BH_WINE_COLLISION_04.19.2026.md` (drift that motivated this spec)
- `docs/vocabulary/MEMORY_TABLET_SPECIFICATION.md` (Tablet-as-object)
- `knowledge3d/bridge/headless_tablet.py` (the contract lives here)

---

## 0. Daniel's Framing (Load-Bearing Quote)

> "K3D is a live game with a living AI inside using the Tablet as a virtual interface (procedural) to old paradigms: LLMs, benchmarks answering, old chat interfaces so it can answer users text queries, ingestion of documents to the knowledgeverse/house — using sleeptime compute to transmute temporary stars to actual house knowledge. We can later include the embodied animation and such details, for now it only need to work, not be pretty."
> — Daniel, 2026-04-20

Three obligations fall out of this quote and govern every decision below:

1. **Tablet is the ONLY procedural bridge** between the living AI and the old paradigms.
2. **Ingested content starts as temporary stars** in the Knowledgeverse and is **transmuted to House knowledge by sleeptime compute**, not at ingest-time.
3. **Paper-MVP scope:** it must *work* (functional dispatch across the four paradigms). Pretty UI, embodied animation, polished tablet mesh — deferred.

---

## 1. Paradigm Inventory

The Tablet mediates four old paradigms. Each needs exactly one WINE file under `knowledge3d/tablet/wine/`.

| # | Old Paradigm | Tablet Surface Kind | WINE File | Status | Notes |
|---|---|---|---|---|---|
| 1 | Benchmarks — answer sheets (math, ARC, MMLU, GSM8K, LHE) | `MATH`, `GAME_2D`, `QUESTION` | `tablet/wine/math_wine.py`, `game2d_wine.py`, `question_wine.py` | **Covered** | Existing factories; living reference for WINE shape |
| 2 | LLM calls — proceduralize-by-peek (external model helps turn docs → procedural stars) | `GENERAL` (new: `PROCEDURALIZE`) | `tablet/wine/proceduralize_wine.py` | **Drift → spec'd** | Today lives at `knowledge3d/ingestion/proceduralizer_wine.py`. v2 spec moves it. |
| 3 | Chat interface — user text in, AI answer out | `CHAT` (constant exists) | `tablet/wine/chat_wine.py` | **Missing → spec'd in parallel** | `SURFACE_KIND_CHAT` constant exists at `headless_tablet.py:30`; no `TabletIngest.chat_task` yet |
| 4 | Document ingestion — PDF/URL/file → temporary stars | `GENERAL` (or new: `INGEST`) | `tablet/wine/ingest_wine.py` | **Missing → spec'd in parallel** | Batches queue into the Knowledgeverse; sleeptime promotes to House |

"Status" legend: **Covered** = factory exists and is called from live paths. **Drift** = code exists but outside `tablet/wine/`. **Missing** = no WINE file yet.

**Drift to fix (from Op-BH collision):** `knowledge3d/ingestion/proceduralizer_wine.py` **must move** to `tablet/wine/proceduralize_wine.py` per the v2 proceduralizer spec. Anything that calls Ollama outside `tablet/wine/` is drift by definition (§6).

---

## 2. Tablet-as-Procedural-Interface Contract

The Tablet is procedurally extensible. `_normalize_surface_kind()` in `headless_tablet.py` already defaults unknown surface kinds to `SURFACE_KIND_GENERAL`, so new paradigms can land without core changes — they just need a conforming WINE.

### 2.1 Every WINE MUST implement three things

Each WINE file exports exactly this triad:

```python
# tablet/wine/<surface>_wine.py

<SURFACE>_ROUTE_GALAXIES: tuple[str, ...] = (...)          # which live galaxies this paradigm needs
def build_<surface>_route(...) -> dict: ...                 # route-policy payload
def build_<surface>_task(...) -> tuple[dict, dict]: ...     # (task_dict, metadata_dict)
def <surface>_envelope(...) -> TabletEnvelope:              # returns a TabletEnvelope
    return TabletIngest.<surface>_task(...)                 # factory on TabletIngest
```

That's the whole shape. Five-to-ten lines of scaffolding per paradigm.

### 2.2 Where the factory lives

`TabletIngest` is the class that owns envelope construction. For each WINE, add one classmethod:

```python
# knowledge3d/bridge/headless_tablet.py  (append inside TabletIngest)

@classmethod
def chat_task(cls, ...) -> TabletEnvelope: ...
@classmethod
def ingest_task(cls, ...) -> TabletEnvelope: ...
@classmethod
def proceduralize_task(cls, ...) -> TabletEnvelope: ...
```

`MATH`, `GAME_2D`, `QUESTION` already have their factories. The three above are what's missing/drifting.

### 2.3 What `to_route_payload()` already does

From `headless_tablet.py:534-554` — `TabletEnvelope.to_route_payload()` emits:

```python
{
  "command": "ROUTE",
  "surface_kind": <normalized>,
  "specialist": <auto|visual|math|chat|grammar|any>,
  "use_enriched": bool,
  "route_policy": "all_live_galaxies",
  "task": {<paradigm-specific task dict>},
  "query": <user text>,
  "domain_hint": <optional>,
  "galaxies": [<subset>]     # when route_policy != all_live_galaxies
}
```

This is the **only** shape the sovereign core sees. Every WINE compiles to this.

### 2.4 How the sovereign core dispatches by `surface_kind`

The core reads `surface_kind` from the ROUTE payload and routes into the composed-head pipeline with paradigm-specific galaxy bias:

- `MATH` / `GAME_2D` / `QUESTION` → existing benchmark specialists, existing result_kind contracts
- `CHAT` → Jarvis coordinator lane, nine-chain swarm over Word + Grammar + Reality + Math
- `PROCEDURALIZE` (carried as `GENERAL` + `task.kind = "proceduralize"`) → proceduralizer v2 pipeline (ternary contrastive + Transfer Yard)
- `INGEST` (carried as `GENERAL` + `task.kind = "ingest"`) → queue into temporary-star region; sleeptime promotes

**Sovereignty note:** surface_kind dispatch is a **pipeline selector**, not a branching intelligence. Each lane runs the same composed head (Morton → LED-A* → Frustum → LOD → Nine-Chain → Halting). The WINE only tells the core *which galaxies to bias* and *what result_kind to expect*.

---

## 3. The Four WINE Surfaces

### 3.1 PROCEDURALIZE WINE — see v2 spec

**Authoritative spec:** `TEMP/CLAUDE_SMART_PROCEDURALIZER_SPEC_V2_04.20.2026.md`

- Replaces surface-form trigram embedder with composable-basis RPN projection.
- File must land at `tablet/wine/proceduralize_wine.py` (currently drifting at `ingestion/proceduralizer_wine.py`).
- Produces temporary stars with confidence_trit; sleeptime promotes or discards.
- Called by ingest WINE on chunks, or standalone for single-document ops.

This umbrella spec defers all detail to v2.

### 3.2 BENCHMARK WINE — existing reference

- `tablet/wine/math_wine.py` — `SURFACE_KIND_MATH`, result_kind `math_answer`
- `tablet/wine/game2d_wine.py` — `SURFACE_KIND_GAME_2D`, frame→action contract
- `tablet/wine/question_wine.py` — `SURFACE_KIND_QUESTION`, MMLU/LHE-style

These are the **canonical shape** every new WINE copies. A new WINE that doesn't read like these three is wrong.

### 3.3 CHAT WINE — being specced (reference here, full spec separate)

**Purpose:** user types text → Tablet dispatches → AI answers.

**Key fields:**
- `surface_kind = SURFACE_KIND_CHAT`
- `specialist = "chat"` (code 3 in `_SPECIALIST_CODES`)
- `task = {"kind": "chat", "turn_id": ..., "history_ref": ..., "user_text": ...}`
- `galaxies = ("Word", "Grammar", "Reality", "Math", "Character")` (paradigm-biased subset; fall back to all-live if the domain_hint widens)
- `result_kind = "chat_answer"`

**Dispatch behavior (see §5):** CHAT is the entry point for user traffic; it may itself emit an INGEST or PROCEDURALIZE envelope mid-session when the user pastes a document.

### 3.4 INGEST WINE — being specced (reference here, full spec separate)

**Purpose:** documents in (PDF/URL/file) → temporary stars out → sleeptime transmutes to House.

**Key fields:**
- `surface_kind = SURFACE_KIND_GENERAL` with `task.kind = "ingest"` (or reserve `SURFACE_KIND_INGEST` if we decide it warrants a first-class constant — acceptable per §2, Tablet defaults unknown to GENERAL)
- `task = {"kind": "ingest", "source_uri": ..., "mime": ..., "chunking": ..., "lang_hint": ...}`
- `galaxies = ("Word", "Grammar", "Reality", "Character")` for text; image/audio ingestion bias their own
- `result_kind = "ingest_receipt"` — returns an ingest_id the caller can poll

**Contract with Knowledgeverse:**
- INGEST writes into the **temporary-star region** (not House).
- Each temporary star carries `confidence_trit` and `source_ingest_id`.
- **Sleeptime promotion** (existing `knowledge3d/knowledgeverse/sleeptime.py`) reads temporary stars, cross-references Galaxy, and either:
  1. **Promote** → write to House JSONL + clear temporary region
  2. **Merge** into existing House node via symlink
  3. **Discard** → confidence_trit is -1 (negative)

No Python in the hot path of ingestion either: INGEST WINE just *queues* the work; the proceduralizer (v2) runs sovereign.

---

## 4. Live-Game Framing

The Tablet is an **inventory slot** the AI always holds. It is *the* 3D object in the avatar's hand in the House. Whether it's rendered with pretty textures or a grey cube is out of scope today.

- **Always-in-hand:** the daemon keeps a `TabletEnvelope` channel open at all times.
- **On-demand WINE dispatch:** when traffic arrives (user typing, benchmark harness firing, file dropped on the tablet), the daemon selects the right WINE and builds an envelope.
- **Temporary stars:** anything INGEST or PROCEDURALIZE emits is temporary until sleeptime runs.
- **Paper MVP:** only needs functional dispatch across the four paradigms and a working sleeptime promotion. No embodied animation, no 3D tablet mesh polish.

**Game-engine analogy:**
| Live-game concept | K3D / Tablet |
|---|---|
| Player inventory slot | Tablet in avatar's hand |
| Tool the player wields | WINE the Tablet dispatches |
| Picked-up item (not yet identified) | Temporary star |
| Overnight rest that identifies items | Sleeptime promotion |

---

## 5. Dispatch Ordering

Four entry points → four WINE dispatches. They compose at runtime.

### 5.1 User chat turn (primary loop)

```
User text → CHAT WINE → TabletEnvelope(CHAT)
  ├── if text is a plain question → sovereign core answers directly
  ├── if text contains a document reference → CHAT emits an INGEST envelope mid-turn
  │     └── INGEST WINE queues temporary stars; CHAT returns "ingesting, will answer shortly"
  └── if text asks "learn this" with pasted content → CHAT emits a PROCEDURALIZE envelope
        └── PROCEDURALIZE WINE runs v2 pipeline; CHAT returns proceduralization receipt
```

### 5.2 Benchmark harness (explicit)

```
Harness tick → MATH/GAME_2D/QUESTION WINE → TabletEnvelope(<kind>)
  └── sovereign core runs composed head; WINE returns scored answer
```

Benchmarks are a *normal* dispatch path — they are not a mode, not a separate runtime. (See memory: "Benchmarks as Natural Activity.")

### 5.3 Background ingestion (queued)

```
Watch folder / URL queue → INGEST WINE → TabletEnvelope(GENERAL + kind=ingest)
  └── writes temporary stars; returns ingest_id
```

### 5.4 Sleeptime (hot-path learning phase, not background)

> **Correction (2026-04-20, Daniel):** Sleeptime is **hot path**. It is the
> learning phase — consolidates both *knowledge* (stars) and *logic*
> (weights) in VRAM, fast. Disk persistence is the only I/O tail. It is
> not "background" in the sense of offloaded / low-priority; it runs with
> the same sovereignty discipline as the wake cycle. See
> `feedback_sleeptime_is_hot_path.md` and Ingest spec §10.

```
Sleeptime tick → reads temporary-star region →
  ├── cross-reference Galaxy semantic gravity
  ├── either Promote, Merge, or Discard
  └── update House JSONL + clear/mark temporary stars
```

**Ordering rule:** CHAT is synchronous (user is waiting). INGEST and PROCEDURALIZE are asynchronous (return a receipt, sleeptime finishes the work). Benchmarks run at harness cadence, independent of user traffic.

### 5.5 Can PROCEDURALIZE be called from inside a CHAT session?

**Yes — but only by emitting a fresh envelope**, never by in-lining the proceduralizer inside chat code.

- CHAT WINE may construct a PROCEDURALIZE envelope and hand it back to the daemon.
- The daemon dispatches it on the next tick.
- CHAT returns to the user with a receipt ("ok, learning that now").
- When sleeptime promotes those stars, the next CHAT turn will naturally see them via Galaxy semantic gravity.

The rule: **WINEs emit envelopes, they do not call each other's code.** This keeps every old-paradigm bridge exactly one envelope wide.

---

## 6. Sovereignty Boundary

**Tablet WINE is the ONLY approved surface for external-paradigm traffic.**

Concretely:

- **Allowed:** `knowledge3d/tablet/wine/*.py` may call Ollama, hit HTTP endpoints for PDF fetchers, import libraries (numpy-in-ingestion is tolerated by §Ingestion-flexibility), etc.
- **Drift:** any other file that imports `ollama`, `requests`, `httpx` to hit an LLM, or that calls a proceduralizer-like service directly. The Op-BH collision exists precisely because `knowledge3d/ingestion/proceduralizer_wine.py` sits outside `tablet/wine/`.

**Audit directive for Codex:**

```
grep -rn "ollama" knowledge3d/ --include="*.py" \
  | grep -v "^knowledge3d/tablet/wine/"
```

Anything that greps outside `tablet/wine/` is drift to fix. Same pattern for `requests`, `httpx`, or any LLM-client library.

**Hot path remains sovereign:** the WINE builds an envelope; the envelope's `to_route_payload()` returns a pure dict; the sovereign core executes PTX kernels over Galaxy + RPN. No Python in the reasoning path, including the CHAT lane — CHAT is a *dispatch surface*, not a reasoning implementation.

---

## 7. Paper Scope Reminder

Proceed only as far as needed to:

1. **Lift GSM8K/Math scores** via functional CHAT + PROCEDURALIZE + sleeptime promotion (more knowledge → better answers).
2. **Demo the four paradigms** for the paper: LLM-assisted proceduralization, benchmark answering, chat interface, document ingestion.
3. **Show the loop closes:** user ingests docs → sleeptime promotes → next question lands better.

Anything beyond these three goals is out of scope for the current spec cycle.

---

## 8. Not In Scope (Explicit)

None of these block the umbrella spec. They are deferred, not denied:

- **Embodied avatar animation** — the Tablet doesn't need to look pretty in the viewer.
- **Polished 3D tablet mesh** — grey cube is fine; the procedural interface works the same.
- **Multi-user concurrency** — single sovereign AI, single daemon, one envelope channel.
- **Blockchain / ownership / attestation** — out.
- **Real-time voice** (STT → CHAT → TTS loop) — the CHAT WINE should be designed so voice can wrap it later without WINE changes, but the voice harness itself is out.
- **Viewer tablet UI** — no drag-and-drop, no on-tablet rendering of ingestion progress.
- **Permission/auth layers** — the sovereign AI is single-user for the paper.
- **Federated Tablets** — one Tablet per running daemon.

---

## 9. Success Criteria (for Codex, downstream)

This umbrella spec is satisfied when:

1. **Four WINE files exist under `tablet/wine/`** with the §2.1 triad — `math_wine.py`, `game2d_wine.py`, `question_wine.py`, `chat_wine.py`, `ingest_wine.py`, `proceduralize_wine.py` (note: six files total — three benchmark surfaces + three new).
2. **`TabletIngest` has `chat_task`, `ingest_task`, `proceduralize_task` classmethods** producing conforming `TabletEnvelope`s.
3. **`knowledge3d/ingestion/proceduralizer_wine.py` is deleted**, replaced by `tablet/wine/proceduralize_wine.py` (per v2 spec).
4. **Sovereignty audit passes**: `grep -rn "ollama\|requests.*post\|httpx.*post" knowledge3d/ --include="*.py" | grep -v "tablet/wine/"` returns zero external-call lines.
5. **End-to-end demo**: user chat turn that references a PDF → CHAT emits INGEST envelope → temporary stars created → sleeptime promotes → next chat turn answers from the new knowledge.
6. **Benchmark non-regression**: ARC 10/10, Math 20/20 still pinned after WINE consolidation.

---

## 10. Codex Directives (Pointers, Not Code)

- **Read first:** `knowledge3d/bridge/headless_tablet.py` in full — especially lines 26-32 (surface_kind constants), 56-63 (specialist codes), 483-554 (TabletEnvelope).
- **Copy from:** `knowledge3d/tablet/wine/math_wine.py` as the canonical WINE shape.
- **Move, don't duplicate:** `knowledge3d/ingestion/proceduralizer_wine.py` → `knowledge3d/tablet/wine/proceduralize_wine.py`. Delete the old path; update all imports.
- **Do not add:** any new surface_kind constants beyond what `headless_tablet.py` already defines, unless a WINE genuinely needs first-class dispatch (INGEST is a candidate; CHAT already has one).
- **Parallel specs to consult:** chat WINE spec (in-flight), ingest WINE spec (in-flight), proceduralizer v2 spec (`TEMP/CLAUDE_SMART_PROCEDURALIZER_SPEC_V2_04.20.2026.md`), dispatch wiring (`TEMP/CLAUDE_DUAL_PATH_INGESTION_AND_DISPATCH_WIRING_04.20.2026.md`).
- **Test gate:** after each WINE lands, re-run the sovereignty grep in §6; if it returns any line outside `tablet/wine/`, the WINE is not complete.

---

## 11. Summary — One Contract, Four Paradigms

The Tablet is the living AI's single procedural interface to every old paradigm. One envelope shape, one dispatch contract, one sovereignty boundary. Benchmarks, chat, LLM-assisted proceduralization, and document ingestion all compile to the same `TabletEnvelope → ROUTE → sovereign core` path. Temporary stars land in the Knowledgeverse; sleeptime transmutes the worthy ones to House. The paper ships when those four WINEs work — pretty can come later.
