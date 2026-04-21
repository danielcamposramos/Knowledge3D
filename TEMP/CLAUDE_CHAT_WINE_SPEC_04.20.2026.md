# CLAUDE Chat WINE Spec — Tablet Chat Surface

**Date:** 2026-04-20
**Author:** Claude (Architecture Partner)
**Status:** Draft — ready for Codex
**Umbrella:** `TEMP/CLAUDE_TABLET_AS_PROCEDURAL_INTERFACE_04.20.2026.md`
**Paradigm tagline (Daniel, 2026-04-20):** "For now it only needs to work, not be pretty."

---

## 1. Purpose

K3D is a live, always-on embodied AI. Humans must be able to ask it free-text questions through the Tablet — the "old chat interface" paradigm. The daemon **already** accepts a `CHAT` command and dispatches to an existing `chat_specialist`. What is missing is the **WINE contract** that wraps chat as a first-class Tablet surface, symmetric to `math_wine.py` / `question_wine.py` / `game2d_wine.py`.

This spec does **not** invent a new specialist. It wraps existing behavior in the Tablet envelope pipeline so chat lives inside the same procedural-interface discipline as every other surface.

---

## 2. Success Criteria

1. `tablet/wine/chat_wine.py` exists, mirrors `math_wine.py` shape, and exposes `build_chat_route()`, `build_chat_task()`, `chat_envelope()`.
2. `TabletIngest.chat_task(...)` factory exists in `knowledge3d/bridge/headless_tablet.py`, symmetric to `math_task` / `question_task`.
3. Daemon `cmd == "CHAT"` path in `knowledge3d/daemon/main.py:980-1022` constructs its task envelope **via `TabletIngest.chat_task`** — no inline envelope construction.
4. `python -m knowledge3d.tablet.chat --text "what is 2+3?"` prints a JSON envelope containing `"response": "5"` and exits 0.
5. Existing benchmark regression: sovereign math path still returns `"5"` for `"what is 2+3?"` with `gpu_execution: true`.
6. Zero behavioral change for existing TCP/STDIO callers of `CHAT` — contract is additive.
7. Surface is marked `SURFACE_KIND_CHAT` (already declared at `headless_tablet.py:30`), never `SURFACE_KIND_QUESTION`.

---

## 3. Ground Truth (verbatim from Wave-1)

- **Daemon already chat-capable:** `knowledge3d/daemon/main.py:980-1022` handles `cmd == "CHAT"`, extracts `payload.get("messages")`, dispatches through `knowledgeverse.execute_task(specialist="chat")`, returns `{"status":"ok","response":...}`. Listens via `serve_stdio()` (line 1079) and `serve_tcp()` (line 1101).
- **Specialist exists:** `knowledge3d/knowledgeverse/chat_specialist.py` (galaxy-navigation based).
- **Constants declared, factories missing:** `SURFACE_KIND_CHAT` is defined in `headless_tablet.py` line 30. `TabletIngest.chat_task` does **not** exist.
- **`question_wine.py` accepts optional `options`**, so free-text Q&A is technically possible through QUESTION surface — but Daniel's framing separates CHAT from QUESTION (different paradigm). Use `SURFACE_KIND_CHAT`.
- **CLI wrapper missing** — no `python -m knowledge3d.tablet.chat` entry.

---

## 4. Paradigm Boundary: CHAT vs QUESTION

| Axis | `QUESTION` (existing) | `CHAT` (this spec) |
|------|----------------------|---------------------|
| Shape | Single prompt, possibly with answer options | Multi-turn `messages` list with `{role, content}` |
| Intent | One-shot query (benchmark-style) | Conversational thread, prior-turn context matters |
| Surface constant | `SURFACE_KIND_QUESTION` | `SURFACE_KIND_CHAT` |
| Specialist | `math` / `question` / router | `chat` |
| Turn state | None | Full `messages` list per turn (stateless server, see §9) |

**Rule:** Do not collapse CHAT into QUESTION even though the sovereign math path answers both. They are different Tablet surfaces for different paradigms. A benchmark run is a QUESTION. A human talking to the AI is a CHAT.

---

## 5. Module: `knowledge3d/tablet/wine/chat_wine.py`

### 5.1 Shape (mirror `math_wine.py`)

```python
"""Chat WINE — Tablet surface for multi-turn free-text conversation.

Mirrors math_wine.py / question_wine.py shape. Chat reads the House through
the internal `chat_specialist`; it does NOT emit stars (no ingest). If a turn
crosses into ingest (user uploads a doc), that is a PROCEDURALIZE handoff,
not a chat-side concern.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

# Default galaxies resident in VRAM during chat reasoning.
# Chat is broad — keep the full default set loaded. No selection, no capping.
# (See feedback_no_knowledge_caps.md — quantity caps NEVER; LOD + frustum cull
# handle the working-memory management on GPU.)
CHAT_ROUTE_GALAXIES: tuple[str, ...] = (
    "Drawing",
    "Character",
    "Word",
    "Number",
    "Grammar",
    "Math",
    "Reality",
    "Audio",
    "3DObjects",
    "Tool",
)

# Maximum bytes per turn content field (UTF-8 encoded). Input gate.
CHAT_MAX_CONTENT_BYTES: int = 16 * 1024  # 16 KiB per message — tune later
CHAT_MAX_MESSAGES: int = 64              # per-turn history cap — tune later


def build_chat_route() -> dict[str, Any]:
    """Return the galaxy route descriptor consumed by the chat specialist.

    Symmetric to math_wine.build_math_route(). Contains the resident galaxy
    set plus any route hints the specialist needs. No semantic gravity
    tuning here — that lives inside the specialist on GPU.
    """
    return {
        "galaxies": CHAT_ROUTE_GALAXIES,
        "lod_policy": "dynamic",      # composed-head default
        "frustum_cull": True,         # composed-head default
    }


def build_chat_task(
    messages: Sequence[Mapping[str, str]],
    *,
    context: Mapping[str, Any] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """Return the task payload dict for a single chat turn.

    Args:
        messages: Sequence of {"role": "user"|"assistant"|"system",
                  "content": str}. Client sends full history each turn
                  (stateless server — see §9).
        context: Optional dict of prior-turn references
                 (e.g. {"prior_program_ids": [...]}) the client wants
                 the specialist to consider.
        stream: If True, the specialist MAY emit partial envelopes.
                Out-of-scope for MVP (see §11); include the flag now so
                wire format is stable. Default False.
    """
    return {
        "specialist": "chat",
        "messages": list(messages),
        "context": dict(context) if context else {},
        "stream": bool(stream),
        "route": build_chat_route(),
    }


def chat_envelope(
    messages: Sequence[Mapping[str, str]],
    *,
    context: Mapping[str, Any] | None = None,
    stream: bool = False,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Return the full Tablet envelope for a chat turn.

    This is the factory TabletIngest.chat_task delegates to. Shape
    mirrors math_wine.math_envelope().
    """
    return {
        "surface_kind": "CHAT",
        "task_id": task_id,
        "task": build_chat_task(messages, context=context, stream=stream),
    }
```

### 5.2 Why these galaxies

Chat is broad-domain — a user can ask about math, physics, language, drawings, or everyday facts in the same thread. **All default galaxies resident**, per `feedback_no_knowledge_caps.md`. LOD + frustum culling are the management mechanism on GPU; Python does not pre-select.

### 5.3 What `chat_wine.py` must NOT do

- Must not call the specialist directly.
- Must not instantiate the daemon.
- Must not handle I/O (stdin, stdout, sockets).
- Must not mutate global state.
- Must not import numpy / cupy / scipy (`feedback_no_numpy_no_bulk_libraries_sovereign_only.md`).

It is a **pure envelope factory**, same contract shape as every other WINE module.

---

## 6. Addition to `TabletIngest` (`knowledge3d/bridge/headless_tablet.py`)

Add method **symmetric to `math_task` / `question_task`**. Do not re-implement envelope construction — delegate to `chat_wine.chat_envelope`.

```python
# inside class TabletIngest:

def chat_task(
    self,
    messages: Sequence[Mapping[str, str]],
    *,
    context: Mapping[str, Any] | None = None,
    stream: bool = False,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Build a CHAT surface envelope for the Tablet.

    Mirrors self.math_task / self.question_task. Sets
    surface_kind = SURFACE_KIND_CHAT and delegates envelope
    construction to chat_wine.chat_envelope.
    """
    from knowledge3d.tablet.wine.chat_wine import chat_envelope
    envelope = chat_envelope(
        messages,
        context=context,
        stream=stream,
        task_id=task_id,
    )
    # Stamp surface_kind as the canonical int/enum constant
    envelope["surface_kind"] = SURFACE_KIND_CHAT
    return envelope
```

**Codex directive:** If existing `math_task` / `question_task` normalize / validate the envelope before returning (e.g. attach timestamps, task IDs, trace handles), apply the same normalization here. Keep the delta in `chat_wine.py` minimal and the Tablet-layer concerns in `TabletIngest`.

---

## 7. Wire the Existing Daemon Through the WINE Contract

**File:** `knowledge3d/daemon/main.py`
**Range:** `cmd == "CHAT"` block (lines ~980–1022 per Wave-1).

### 7.1 Current (paraphrased from Wave-1)

```python
if cmd == "CHAT":
    messages = payload.get("messages")
    # extracts prompt, builds inline task, dispatches to knowledgeverse.execute_task(specialist="chat")
    response = {"status": "ok", "response": solved.get("response", ...), ...}
```

### 7.2 Target

```python
if cmd == "CHAT":
    messages = payload.get("messages") or []
    context = payload.get("context") or {}
    stream = bool(payload.get("stream", False))
    task_id = payload.get("task_id")

    # Gate inputs (see §10)
    _validate_chat_input(messages, context)

    # Route through the WINE contract — no inline envelope construction
    envelope = self._tablet_ingest.chat_task(
        messages,
        context=context,
        stream=stream,
        task_id=task_id,
    )

    solved = self._knowledgeverse.execute_task(
        specialist="chat",
        envelope=envelope,
    )

    response = {
        "status": "ok",
        "response": solved.get("response", ""),
        "program_id": solved.get("program_id"),
        "gpu_execution": bool(solved.get("gpu_execution", False)),
        "telemetry": solved.get("telemetry", {}),
        "task_id": envelope.get("task_id"),
    }
```

### 7.3 Contract guarantees

- **Behaviorally identical** for existing TCP/STDIO callers — they keep sending `{"command":"CHAT","messages":[...]}` and keep getting `{"status":"ok","response":...}`.
- **Additive fields** (`program_id`, `gpu_execution`, `telemetry`, `task_id`) are safe — callers that don't read them are unaffected.
- Envelope construction now lives in one place (`chat_wine.chat_envelope`). Any future surface-level change (e.g. adding a `tools` field for tool-use) is a one-line change in `chat_wine.py`, not a daemon diff.

---

## 8. Thin CLI Entry: `knowledge3d/tablet/chat.py`

### 8.1 Contract

```
python -m knowledge3d.tablet.chat --text "what is 2+3?"
```

- In-process: imports and calls the daemon handler directly, **not** over TCP.
- Prints JSON envelope to stdout (single line or pretty — pretty is fine for MVP).
- Exit 0 on `status == "ok"`, exit 1 on error.
- No interactive REPL in MVP (out-of-scope per §11).

### 8.2 Shape

```python
"""CLI: python -m knowledge3d.tablet.chat --text '...'

Sends a single-turn CHAT envelope to the in-process daemon handler.
Prints the response JSON to stdout. Exit 0 on ok, 1 on error.

This is a thin I/O wrapper — it does NOT implement chat logic.
"""
from __future__ import annotations

import argparse
import json
import sys

from knowledge3d.daemon.main import handle_command_inprocess  # existing or new helper

def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="knowledge3d.tablet.chat")
    ap.add_argument("--text", required=True, help="User message text")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return ap.parse_args()

def main() -> int:
    args = _parse()
    payload = {
        "command": "CHAT",
        "messages": [{"role": "user", "content": args.text}],
    }
    result = handle_command_inprocess(payload)
    indent = 2 if args.pretty else None
    sys.stdout.write(json.dumps(result, indent=indent, ensure_ascii=False))
    sys.stdout.write("\n")
    return 0 if result.get("status") == "ok" else 1

if __name__ == "__main__":
    raise SystemExit(main())
```

**Codex directive:** If `handle_command_inprocess` does not exist, extract the existing TCP/STDIO dispatch body into a pure function `handle_command_inprocess(payload: dict) -> dict` and have both `serve_stdio` and `serve_tcp` call it. This is a refactor to expose the handler without socket I/O. Zero behavioral change.

### 8.3 What the CLI must NOT do

- Must not open a TCP socket (in-process only for MVP).
- Must not persist history to disk (stateless).
- Must not handle streaming (out-of-scope).
- Must not authenticate (out-of-scope).

---

## 9. Turn State (Decision: Option A — Stateless Server)

**Chosen:** Client passes full `messages` list every turn. Server holds **no** chat session state.

**Why:**
- Matches the daemon's existing `payload.get("messages")` extraction — zero server-side surgery.
- Simpler semantics for the MVP — no session expiry, no session GC, no multi-user collisions.
- Aligns with "AI is always-on and embodied" — the Tablet surface is the context, not a per-session Python dict.
- If/when persistent sessions are needed, they can be introduced as an **additive** server-side cache keyed by `task_id` — the wire format already carries `task_id`.

**Turn boundaries:** The client is responsible for appending the assistant's prior `response` as `{"role":"assistant","content":...}` before sending the next user turn. CLI (§8) is single-turn and therefore does not manage history.

**`context` dict** is the escape hatch for clients that want the specialist to look up references from prior turns (e.g. `{"prior_program_ids": [...]}`) without re-sending full assistant replies. Optional.

---

## 10. Gates / Validation

### 10.1 Input gate (before dispatch)

Implement `_validate_chat_input(messages, context)` in the daemon (or in `chat_wine.py` as a pure function — Codex's call; I lean pure function for testability).

Rules:

| Rule | Failure mode |
|------|--------------|
| `messages` must be a non-empty list | Return `{"status":"error","error":"empty_messages"}` |
| `len(messages) <= CHAT_MAX_MESSAGES` | Return `{"status":"error","error":"too_many_messages"}` |
| Each message has `role in {"user","assistant","system"}` and string `content` | Return `{"status":"error","error":"bad_message_shape"}` |
| `len(content.encode("utf-8")) <= CHAT_MAX_CONTENT_BYTES` | Return `{"status":"error","error":"content_too_large"}` |
| Content must be valid UTF-8 (reject surrogates) | Return `{"status":"error","error":"bad_encoding"}` |

All validation is **pure Python string-length / type checks** — **not** reasoning logic. Sovereignty is intact (`feedback_python_dispatch_is_not_a_line_item.md` — these are I/O gates, not dispatch decisions).

### 10.2 Output envelope (expected back from `knowledgeverse.execute_task`)

```python
{
    "status": "ok",
    "response": str,          # the user-visible reply
    "program_id": str | None, # RPN program id if one was composed
    "gpu_execution": bool,    # True if hot path was sovereign GPU
    "telemetry": dict,        # per feedback_note_taking_everywhere.md
    "task_id": str | None,
}
```

`gpu_execution: false` is **not** a soft-fallback signal — it means the chat specialist returned without entering the composed-head pipeline (e.g. greeted or echoed). Sovereign path when the reply requires reasoning is still the hard requirement.

### 10.3 Telemetry

Per `feedback_note_taking_everywhere.md` — every chat turn emits a trace. At minimum:
- `specialist`: "chat"
- `galaxies_touched`: list[str]
- `rpn_opcodes_executed`: int
- `ptx_kernels_invoked`: int
- `halting_gate_converged`: bool
- `latency_ms`: float
- `turn_index`: int (derived from `len(messages)`)

Silence is a bug. If a chat turn produces no telemetry, the system is lying about its work.

---

## 11. Not In Scope (MVP)

Explicit out-of-scope list so Codex does not accidentally pull these in:

- **Streaming SSE output** — the `stream` flag is reserved in the wire format but the MVP ignores it (treats all turns as non-streaming). Design separately once non-stream path is proven.
- **WebSocket protocol** — `cli_client.py` is incomplete per Wave-1. Out of scope.
- **Authentication / authorization** — single-user local daemon assumption.
- **Rate limiting** — single-user local daemon assumption.
- **Multi-user sessions** — out of scope (stateless server, §9).
- **Persistent chat history on disk** — out of scope (House persistence handles knowledge, not transcripts).
- **Tool-use / function-calling extensions to the envelope** — add as additive fields later; do not design preemptively.
- **A new chat specialist** — one already exists at `knowledge3d/knowledgeverse/chat_specialist.py`. Do not rewrite it. Do not design a new router. Do not add it to the specialist registry under a new name.

---

## 12. Sovereignty Compliance

| Concern | Status |
|---------|--------|
| No Python fallbacks in hot path | Chat specialist is galaxy-nav based on GPU — this spec does not introduce any Python reasoning |
| No numpy / cupy / scipy | WINE module is pure Python dicts; CLI is pure Python |
| No Python regex / string ops for reasoning | Input validation uses `len()` + type checks + UTF-8 decode — **not reasoning**, pure I/O gating |
| Python ≤ boot + I/O | Envelope factory + CLI + daemon dispatch only. Zero reasoning added |
| Galaxy-first | `CHAT_ROUTE_GALAXIES` selects resident galaxies; no hardcoded logic |
| Composed head pipeline | Chat specialist runs inside the existing pipeline; this spec composes **into** it, does not bypass |

The spec adds **no new Python in the hot path**. It refactors dispatch through a WINE contract and adds a thin CLI — both are I/O/boot code.

---

## 13. Codex Directives (actionable)

1. Create `knowledge3d/tablet/wine/chat_wine.py` per §5. Mirror `math_wine.py` structure exactly.
2. Add `TabletIngest.chat_task` method in `knowledge3d/bridge/headless_tablet.py` per §6. Delegate to `chat_wine.chat_envelope`.
3. Refactor daemon `CHAT` handler in `knowledge3d/daemon/main.py:980-1022` per §7. No behavioral change — just route through the WINE contract.
4. Extract `handle_command_inprocess(payload)` helper in `knowledge3d/daemon/main.py` if not present, shared by `serve_stdio` / `serve_tcp` / CLI. Pure function, no I/O.
5. Create `knowledge3d/tablet/chat.py` CLI per §8. In-process call only.
6. Implement `_validate_chat_input` per §10 — pure function in `chat_wine.py` (preferred) or in daemon. Exportable, testable.
7. Run tests in §14 under `conda activate k3d-cranium`, `CUDA_VISIBLE_DEVICES=0`.

---

## 14. Test Plan

### 14.1 Integration test (daemon in-process)

**File:** `tests/tablet/test_chat_wine.py` (new)

```python
def test_chat_wine_math_passthrough():
    from knowledge3d.daemon.main import handle_command_inprocess
    result = handle_command_inprocess({
        "command": "CHAT",
        "messages": [{"role": "user", "content": "what is 2+3?"}],
    })
    assert result["status"] == "ok"
    assert result["response"].strip() == "5"
    assert result["gpu_execution"] is True
```

Gate: reuses the sovereign math path that already returns `"5"` (per `project_first_sovereign_math_answer.md`). If this test fails, either the sovereign path regressed or the CHAT wiring broke — both are hard failures.

### 14.2 CLI smoke test

**File:** `tests/tablet/test_chat_cli.py` (new)

```python
import json, subprocess, sys

def test_chat_cli_math():
    out = subprocess.check_output(
        [sys.executable, "-m", "knowledge3d.tablet.chat", "--text", "what is 2+3?"],
        text=True,
    )
    result = json.loads(out)
    assert result["status"] == "ok"
    assert result["response"].strip() == "5"
```

### 14.3 Input-gate unit tests

Exhaustive table tests for `_validate_chat_input`:
- empty messages list → error
- over-cap messages list → error
- bad role → error
- non-string content → error
- over-cap content bytes → error
- well-formed input → passes

### 14.4 Non-regression

Run the existing math benchmark suite (`benchmarks/math_*.py`) after the refactor. The `QUESTION` surface path and the composed-head pipeline must not change output. Pinned benchmark state (Math 20/20) must stay green.

### 14.5 Sovereignty grep

```
grep -RnE "import (numpy|cupy|scipy|sympy)" knowledge3d/tablet/wine/chat_wine.py \
    knowledge3d/tablet/chat.py
```

Must return zero matches.

---

## 15. Open Questions (for Daniel, non-blocking)

1. Should `CHAT_MAX_CONTENT_BYTES` scale with available VRAM / be driven by a House constant rather than a literal in `chat_wine.py`? (MVP: literal.)
2. Is there an expectation that chat responses eventually persist into House as knowledge traces, or are they transient? (MVP: transient — telemetry only, no House writes from chat-side.)
3. Streaming design (SSE vs chunked JSON vs custom) — deferred. When it's time, pick a delegate-to-ollama-specialist moment.

---

## 16. Files Touched (expected)

- **New:** `knowledge3d/tablet/wine/chat_wine.py`
- **New:** `knowledge3d/tablet/chat.py`
- **New:** `tests/tablet/test_chat_wine.py`
- **New:** `tests/tablet/test_chat_cli.py`
- **Edit:** `knowledge3d/bridge/headless_tablet.py` (add `TabletIngest.chat_task`)
- **Edit:** `knowledge3d/daemon/main.py` (route CHAT through WINE + extract `handle_command_inprocess`)

No deletions. No changes to `chat_specialist.py`. No changes to `knowledgeverse.py` reasoning surface (the 4000→200 line Python shrink is Phase D, not this spec).

---

## 17. Definition of Done

- [ ] `chat_wine.py` exists, mirrors `math_wine.py` shape, exports `CHAT_ROUTE_GALAXIES`, `build_chat_route`, `build_chat_task`, `chat_envelope`.
- [ ] `TabletIngest.chat_task` exists and builds envelopes via `chat_wine.chat_envelope`.
- [ ] Daemon `CHAT` handler builds envelopes through `TabletIngest.chat_task` (zero inline construction).
- [ ] `python -m knowledge3d.tablet.chat --text "what is 2+3?"` returns JSON with `"response":"5"` and exit 0.
- [ ] All tests in §14 pass.
- [ ] Sovereignty grep (§14.5) returns zero.
- [ ] Existing math benchmark non-regression confirmed.
- [ ] Telemetry emitted per turn per §10.3.
