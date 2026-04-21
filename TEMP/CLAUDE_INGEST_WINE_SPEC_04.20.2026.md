# CLAUDE Ingest WINE Spec — Tablet Document-Ingestion Surface + Sleeptime Transmutation

**Date:** 2026-04-20
**Author:** Claude (Architecture Partner)
**Status:** Draft — ready for Codex
**Umbrella:** `TEMP/CLAUDE_TABLET_AS_PROCEDURAL_INTERFACE_04.20.2026.md`
**Sibling specs:**
- `TEMP/CLAUDE_SMART_PROCEDURALIZER_SPEC_V2_04.20.2026.md` (PROCEDURALIZE WINE)
- `TEMP/CLAUDE_CHAT_WINE_SPEC_04.20.2026.md` (CHAT WINE)
- `TEMP/CLAUDE_DUAL_PATH_INGESTION_AND_DISPATCH_WIRING_04.20.2026.md` (Path A Layer-0 unblock)

**Paradigm tagline (Daniel, 2026-04-20):** "Ingestion of documents to the knowledgeverse/house — using sleeptime compute to transmute temporary stars to actual house knowledge."

---

## 0. One-line Summary

`ingest_wine.py` is the Tablet surface the AI uses to **queue** a document (PDF / URL / file path) into the Knowledgeverse as **temporary stars**; a background **sleeptime** tick later **transmutes** worthy temporary stars into persistent House knowledge. No reasoning happens inside the WINE — it is pure envelope construction + a receipt handshake.

---

## 1. Purpose & Scope

### 1.1 Purpose

1. Give the living AI (and any user-facing path) a first-class Tablet surface for *"here is a document, learn from it"*.
2. Keep ingestion **asynchronous** — the caller gets an `ingest_receipt` immediately; the proceduralization and House promotion happen on their own tracks.
3. Preserve the **temporary-star → sleeptime-promotion → House** flow so nothing leaks into House before it's earned its place.
4. Keep sovereignty boundaries intact: external libraries (requests, PDF parsers, OCR) are allowed **only** inside `tablet/wine/`.

### 1.2 In scope

- `knowledge3d/tablet/wine/ingest_wine.py` (new WINE module)
- `TabletIngest.ingest_task` classmethod on `knowledge3d/bridge/headless_tablet.py`
- A daemon `INGEST` command handler mirroring the `CHAT` wiring pattern (§7 of Chat spec)
- A thin CLI: `python -m knowledge3d.tablet.ingest --source <uri>`
- **Sleeptime transmutation contract** (§6) — how temporary stars become House entries
- Tests + sovereignty grep

### 1.3 Not in scope (explicit)

- New chunking algorithms, new OCR backends, new PDF parsers — reuse what `knowledge3d/ingestion/` already has (`enhanced_pdf_ingest.py`, `pdf_ocr_pipeline.py`, etc.).
- In-line proceduralization. When a chunk needs enrichment the WINE **emits a PROCEDURALIZE envelope**; it does not import the proceduralizer module (per umbrella §5.5).
- Multi-user ingestion queues, auth, persistent per-user state — single-daemon, single-user (per umbrella §8).
- New opcodes, new kernels, new RPN. None are needed for this spec.
- Viewer-side drag-and-drop UI — deferred with the rest of tablet prettifying.
- Deleting the existing `knowledge3d/ingestion/*.py` modules. Ingest WINE **wraps** them; the relocation surgery is proceduralizer-only (handled in v2 spec).

---

## 2. Success Criteria

1. `knowledge3d/tablet/wine/ingest_wine.py` exists with the canonical WINE triad (`<SURFACE>_ROUTE_GALAXIES`, `build_ingest_route`, `build_ingest_task`, `ingest_envelope`).
2. `TabletIngest.ingest_task(...)` classmethod exists on `headless_tablet.py`, symmetric to `math_task` / `question_task` / `chat_task`.
3. Daemon `cmd == "INGEST"` path exists in `knowledge3d/daemon/main.py`, constructed via `TabletIngest.ingest_task` — no inline envelope.
4. `python -m knowledge3d.tablet.ingest --source file:///tmp/sample.pdf --mime application/pdf` prints a JSON receipt containing `ingest_id` and exits 0.
5. Temporary stars land in the Knowledgeverse's **temporary-star region** with `confidence_trit` and `source_ingest_id` fields populated (see §5).
6. Sleeptime tick reads temporary stars and executes one of: **Promote** (write House JSONL + clear), **Merge** (symlink into existing House node), **Discard** (confidence_trit = -1). Deterministic, logged.
7. End-to-end smoke: `ingest` a tiny test document → `sleeptime` runs → `chat`'s next turn can reference new knowledge via semantic gravity (no explicit look-up path).
8. Sovereignty grep: no `ollama` / `requests` / `httpx` imports outside `tablet/wine/` added by this spec.
9. Pinned benchmark non-regression: ARC 10/10, Math 20/20 still green.

---

## 3. Ground Truth (what exists today)

| Fact | Evidence |
|---|---|
| `SURFACE_KIND_GENERAL` is the default fallback | `headless_tablet.py:31`, `_normalize_surface_kind` defaults unknown → `GENERAL` |
| **No** `SURFACE_KIND_INGEST` constant exists yet | grep of `headless_tablet.py:26-32` |
| `TabletIngest.ingest_task` does NOT exist | grep across `headless_tablet.py` |
| Sleeptime plumbing exists | `knowledge3d/knowledgeverse/sleep_time_micro.py` (PTX-based, inter-frame consolidation) |
| Full-document ingestion modules exist | `knowledge3d/ingestion/enhanced_pdf_ingest.py`, `pdf_ocr_pipeline.py`, `batch_orchestrator.py`, etc. |
| Existing proceduralizer transport | `knowledge3d/ingestion/proceduralizer_wine.py` (drift location; v2 spec moves it to `tablet/wine/`) |
| Galaxy JSONL home | `/K3D/Knowledge3D.local/galaxies/*.jsonl` (House-backing stores) |
| Temporary-star region | *Not* a first-class spec region yet — see §5 decision |

### 3.1 Decision: reserve `SURFACE_KIND_INGEST`

The umbrella spec (§3.4) leaves this open. **This spec locks it down as a first-class constant** because:

- INGEST has a distinct result contract (`ingest_receipt` with `ingest_id`) that's not compatible with `GENERAL`'s generic fallthrough.
- It keeps the dispatch table explicit (`_normalize_surface_kind`, `_SPECIALIST_CODES`) and makes grep-based drift detection trivial.
- Adding a constant costs one line; re-routing `GENERAL + task.kind="ingest"` through the same pipeline later costs more to reason about.

**Codex directive:** add `SURFACE_KIND_INGEST = "INGEST"` at `headless_tablet.py:~32`, extend `_SPECIALIST_CODES` and `_normalize_surface_kind` with `"INGEST"` / `"INGEST_TASK"` entries mapped to it. Do not renumber existing codes (per `feedback_expand_not_replace_opcodes.md` — registry is append-only).

---

## 4. Module: `knowledge3d/tablet/wine/ingest_wine.py`

### 4.1 Canonical WINE triad

```python
"""Ingest WINE — Tablet surface for document ingestion (PDF / URL / file).

Queues a source for proceduralization. Never imports the proceduralizer
directly. Never performs reasoning. Emits a TabletEnvelope with
surface_kind=INGEST; the daemon stamps an ingest_id and drops temporary
stars into the Knowledgeverse's temporary-star region. Sleeptime later
transmutes worthy temporary stars into House knowledge.

See: TEMP/CLAUDE_INGEST_WINE_SPEC_04.20.2026.md
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import (
    ROUTE_POLICY_ALL_LIVE_GALAXIES,
    SURFACE_KIND_INGEST,         # new constant per §3.1
    TabletEnvelope,
    TabletIngest,
)


# Broad — ingestion can cross domains (a PDF of physics, a web page of
# grammar, an audio transcript). Keep all default live galaxies biased in.
# LOD + frustum cull handle working-memory management on GPU per
# feedback_no_knowledge_caps.md.
INGEST_ROUTE_GALAXIES: tuple[str, ...] = (
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


# --- Gate constants (input validation; see §8) -----------------------------

# Max size of a source_uri string (not the document — just the URI itself).
INGEST_MAX_URI_BYTES: int = 4 * 1024

# Supported MIME classes. This is a coarse gate only — per-MIME parsers
# live in knowledge3d/ingestion/ and are selected by the proceduralizer
# pipeline, not by this WINE.
INGEST_SUPPORTED_MIME: frozenset[str] = frozenset({
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/html",
    "application/json",
    "image/png",
    "image/jpeg",
    "audio/wav",
    "audio/mpeg",
})

# Max chunk count per ingest (downstream, the proceduralizer enforces
# per-chunk cost gates). This cap protects the receipt/queue path from
# a single 100k-chunk submission starving everything else.
INGEST_MAX_CHUNKS: int = 4096


def build_ingest_route(
    *,
    specialist: str = "ingest",
    domain_hint: str | None = None,
    galaxies: Sequence[str] | None = None,
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
) -> dict[str, Any]:
    """Route descriptor for an ingest dispatch.

    Mirrors build_math_route(). INGEST does not run the composed head;
    it runs the proceduralizer-feeder chain. Specialist is fixed at
    "ingest" — the sovereign core recognizes this as a queue-write lane.
    """
    route: dict[str, Any] = {
        "specialist": str(specialist or "ingest"),
        "route_policy": str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
    }
    if domain_hint is not None and str(domain_hint).strip():
        route["domain_hint"] = str(domain_hint).strip()
    galaxy_names = [str(name) for name in (galaxies or INGEST_ROUTE_GALAXIES) if str(name).strip()]
    if galaxy_names:
        route["galaxy_names"] = galaxy_names
    return route


def build_ingest_task(
    *,
    task_id: str,
    source_uri: str,
    mime: str,
    chunking: Mapping[str, Any] | None = None,
    lang_hint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (task_payload, route_payload) for a single ingest job.

    Args:
        task_id: Caller-supplied identifier (CLI / daemon generates UUID
            if caller omits). Used as the ingest_id seed.
        source_uri: file:// or https:// URI of the document. Must be
            reachable from the daemon process. No raw bytes in the
            envelope — the fetch happens inside the ingest pipeline.
        mime: IANA MIME type. Must be in INGEST_SUPPORTED_MIME.
        chunking: Optional overrides for chunker policy:
            {"strategy": "pdf_pages"|"md_headers"|"fixed_chars",
             "size": int, "overlap": int}.
            Default policy is inferred from mime inside the proceduralizer.
        lang_hint: Optional BCP-47 language hint (e.g. "pt-BR", "en").
            Used by the proceduralizer to bias multilingual embeddings;
            NOT a filter (knowledge is meaning-centric per MEMORY.md).
        metadata: Free-form caller metadata — provenance, user tags, etc.
    """
    envelope = TabletIngest.ingest_task(
        task_id=task_id,
        source_uri=source_uri,
        mime=mime,
        chunking=chunking,
        lang_hint=lang_hint,
        metadata=metadata,
    )
    return dict(envelope.task), build_ingest_route(
        specialist=envelope.specialist,
        domain_hint=envelope.domain_hint,
        galaxies=envelope.galaxies,
        route_policy=envelope.route_policy,
    )


def ingest_envelope(
    *,
    task_id: str,
    source_uri: str,
    mime: str,
    chunking: Mapping[str, Any] | None = None,
    lang_hint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TabletEnvelope:
    """Return the full TabletEnvelope for an ingest job.

    Mirrors math_wine.math_dataset_envelope(). This is the factory the
    daemon and CLI call; TabletIngest.ingest_task does the actual
    construction.
    """
    return TabletIngest.ingest_task(
        task_id=task_id,
        source_uri=source_uri,
        mime=mime,
        chunking=chunking,
        lang_hint=lang_hint,
        metadata=metadata,
    )
```

### 4.2 What `ingest_wine.py` must NOT do

- Must not import `ollama`, `requests`, `httpx`, `urllib.request` — the actual fetching lives inside the sovereign proceduralizer pipeline invoked via a downstream PROCEDURALIZE envelope.
- Must not import `knowledge3d.ingestion.*` — the WINE is the *envelope builder*, not the orchestrator.
- Must not open files on disk (no `open(source_uri)`). The source_uri is a *string reference*; the pipeline resolves it.
- Must not call `TabletIngest.proceduralize_task(...)` inline. That's the daemon's job after receiving the INGEST envelope (see §5.3).
- Must not mutate global state.

This is a **pure envelope factory**, identical in discipline to `math_wine.py`.

---

## 5. Addition to `TabletIngest` (`knowledge3d/bridge/headless_tablet.py`)

```python
# inside class TabletIngest: (after question_task, symmetric shape)

@staticmethod
def ingest_task(
    *,
    task_id: str,
    source_uri: str,
    mime: str,
    chunking: Mapping[str, Any] | None = None,
    lang_hint: str | None = None,
    galaxies: Sequence[str] | None = None,
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
    metadata: Mapping[str, Any] | None = None,
) -> TabletEnvelope:
    """Build an INGEST surface envelope for the Tablet.

    Mirrors math_task / question_task / chat_task. Sets
    surface_kind = SURFACE_KIND_INGEST. Task payload carries the
    source reference; the daemon is responsible for assigning an
    ingest_id and queueing the job.
    """
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("mime", str(mime))
    if lang_hint is not None:
        merged_metadata.setdefault("lang_hint", str(lang_hint))
    task_payload = {
        "surface_kind": SURFACE_KIND_INGEST,
        "task_id": str(task_id),
        "query": str(source_uri),
        "source_uri": str(source_uri),
        "mime": str(mime),
        "chunking": dict(chunking) if chunking else {},
        "lang_hint": str(lang_hint) if lang_hint is not None else None,
    }
    return TabletEnvelope(
        surface_kind=SURFACE_KIND_INGEST,
        task_id=str(task_id),
        query=str(source_uri),
        specialist="ingest",
        domain_hint=str(lang_hint) if lang_hint is not None else None,
        galaxies=tuple(str(name) for name in (galaxies or ()) if str(name).strip()),
        route_policy=str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
        result_kind="ingest_receipt",
        task=task_payload,
        metadata=merged_metadata,
    )
```

**Codex directive:** keep the field set minimal — the proceduralizer v2 spec already defines the heavy shape. This factory is pure construction; if any normalization is needed (e.g. UUID stamping), do it here so daemon callers stay declarative.

---

## 6. Daemon Wiring (`knowledge3d/daemon/main.py`)

Mirror the CHAT spec's §7 pattern.

### 6.1 New handler

```python
if cmd == "INGEST":
    source_uri = payload.get("source_uri")
    mime = payload.get("mime")
    chunking = payload.get("chunking") or {}
    lang_hint = payload.get("lang_hint")
    task_id = payload.get("task_id") or _make_ingest_id()  # helper, §6.2

    _validate_ingest_input(source_uri, mime, chunking)  # §8

    envelope = self._tablet_ingest.ingest_task(
        task_id=task_id,
        source_uri=source_uri,
        mime=mime,
        chunking=chunking,
        lang_hint=lang_hint,
    )

    # Queue. This does NOT run the proceduralizer; it writes a
    # pending-ingest entry that the proceduralizer lane will drain.
    receipt = self._knowledgeverse.enqueue_ingest(envelope=envelope)

    response = {
        "status": "ok",
        "result_kind": "ingest_receipt",
        "ingest_id": receipt["ingest_id"],
        "task_id": task_id,
        "queued_chunks_estimate": receipt.get("queued_chunks_estimate"),
        "telemetry": receipt.get("telemetry", {}),
    }
```

### 6.2 Helpers to add

- `_make_ingest_id() -> str`: deterministic UUIDv7-style id (timestamp prefix + random suffix) so IDs sort chronologically.
- `_validate_ingest_input(source_uri, mime, chunking)`: see §8.
- `knowledgeverse.enqueue_ingest(envelope)`: the knowledgeverse API for writing into the **temporary-star region** (§5 of this spec / §7 of the umbrella). Synchronous — returns the receipt. Internal queueing / proceduralizer dispatch is the Knowledgeverse's concern.

### 6.3 Contract guarantees

- No PROCEDURALIZE envelope is constructed *by the daemon* in the INGEST path. The Knowledgeverse's ingest queue drains into the proceduralizer (via its WINE) on its own cadence — this is the asynchronous boundary.
- Receipt returns immediately; callers poll or wait for sleeptime.

---

## 7. Temporary-Star Region (Knowledgeverse Contract)

### 7.1 Shape

Each temporary star carries the **MeaningCentricStar §2.1 shape** (per v2 proceduralizer spec), **plus** these ingest-only fields:

```json
{
  "star_id": "<content hash>",
  "meaning_rpn": "...",
  "class": "...",
  "domain": "...",
  "surface_forms": {...},
  // Temporary-region-only fields:
  "temporary": true,
  "source_ingest_id": "<ingest_id>",
  "confidence_trit": 0,       // -1 discard, 0 pending, +1 promote (ternary-first, per feedback_ternary_first_where_cheaper.md)
  "first_seen_at": "<iso>",
  "last_touched_at": "<iso>",
  "sleeptime_passes": 0
}
```

**Why ternary confidence?** One opcode (`TERNARY_PROMOTE` / `TERNARY_DISCARD` decision) resolves promotion on GPU without softmax. Aligns with `feedback_attention_is_ternary_plus_contrastive.md`.

### 7.2 Storage

- Path: `/K3D/Knowledge3D.local/galaxies/_temporary/<ingest_id>.jsonl`
- Append-only within a single ingest.
- After sleeptime promotes or discards, the per-ingest JSONL is either merged into the target galaxy JSONL (Promote / Merge) or truncated (Discard).

### 7.3 Why not write straight to House

- House is persistent. Unverified ingestion content must earn its place — Christoph's semantic-gravity pass runs at sleeptime and can reject low-coherence stars.
- Temporary region isolates ingestion noise from benchmark state. If a PDF contains contradictions, the temporary region absorbs them; the sleeptime pass resolves or discards.

---

## 8. Input Gates / Validation

`_validate_ingest_input(source_uri, mime, chunking)` — pure Python, no reasoning.

| Rule | Failure |
|---|---|
| `source_uri` is a non-empty string | `{"status":"error","error":"missing_source_uri"}` |
| `len(source_uri.encode("utf-8")) <= INGEST_MAX_URI_BYTES` | `{"status":"error","error":"uri_too_long"}` |
| URI scheme is one of `file://`, `https://`, `http://` (localhost only), `s3://` | `{"status":"error","error":"unsupported_scheme"}` |
| `mime` is in `INGEST_SUPPORTED_MIME` | `{"status":"error","error":"unsupported_mime"}` |
| `chunking` (if present) has keys in `{"strategy","size","overlap"}` and values of expected types | `{"status":"error","error":"bad_chunking"}` |
| Estimated chunk count (from doc size / chunking) `<= INGEST_MAX_CHUNKS` | `{"status":"error","error":"too_many_chunks"}` |

Size-estimation step is optional for MVP; the real cap enforces during the proceduralizer lane. The cap here is a cheap pre-flight.

These are **I/O gates, not reasoning**. Sovereignty holds per `feedback_python_dispatch_is_not_a_line_item.md`.

---

## 9. CLI Entry: `knowledge3d/tablet/ingest.py`

```
python -m knowledge3d.tablet.ingest \
    --source file:///tmp/sample.pdf \
    --mime application/pdf \
    [--lang-hint pt-BR] \
    [--pretty]
```

```python
"""CLI: python -m knowledge3d.tablet.ingest --source <uri> --mime <type>

Sends a single INGEST envelope to the in-process daemon handler.
Prints the ingest_receipt JSON to stdout. Exit 0 on ok, 1 on error.

Thin I/O wrapper — no ingestion logic.
"""
from __future__ import annotations

import argparse
import json
import sys

from knowledge3d.daemon.main import handle_command_inprocess


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="knowledge3d.tablet.ingest")
    ap.add_argument("--source", required=True, help="Source URI (file://, https://, s3://)")
    ap.add_argument("--mime", required=True, help="IANA MIME type")
    ap.add_argument("--lang-hint", default=None, help="BCP-47 language hint (optional)")
    ap.add_argument("--pretty", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse()
    payload = {
        "command": "INGEST",
        "source_uri": args.source,
        "mime": args.mime,
        "lang_hint": args.lang_hint,
    }
    result = handle_command_inprocess(payload)
    indent = 2 if args.pretty else None
    sys.stdout.write(json.dumps(result, indent=indent, ensure_ascii=False))
    sys.stdout.write("\n")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

**Reuses `handle_command_inprocess`** extracted in the Chat WINE spec (§8). If not yet extracted, Codex extracts it once and both CLIs share it.

---

## 10. Sleeptime Transmutation Contract

Ingestion is half of the loop; **sleeptime finishes it**.

> **Correction (2026-04-20, Daniel):** Sleeptime is **hot path**. It is the
> model's **learning phase** — it consolidates both *knowledge* (temporary
> stars) and *logic* (TRM + specialist region weights), in VRAM, fast. Only
> the final save-to-disk tail is I/O where outside libraries are allowed.
> This section is written to that contract. See
> `feedback_sleeptime_is_hot_path.md`.

### 10.1 Trigger

- **Idle-trigger:** daemon idle > `SLEEPTIME_IDLE_S` (default 60s) AND temporary region non-empty.
- **Count-trigger:** temporary region crosses `SLEEPTIME_MAX_PENDING` stars (default 1024) regardless of idle.
- **Explicit-trigger:** CLI `python -m knowledge3d.tablet.sleeptime --run-once` (for tests and paper-run scripting).

Sleeptime does NOT run during active benchmark harnesses or active CHAT turns. The scheduler pauses sleeptime while any synchronous-surface envelope is in flight.

### 10.2 Per-tick pass (two lanes, both in VRAM)

Every tick runs **two consolidation lanes** in the same VRAM pass — knowledge
and logic. They share Galaxy memory and the composed-head pipeline.

**Lane A — Star consolidation** (for each temporary star, in order of
`first_seen_at`):

1. **Semantic-gravity probe (GPU kernel).** Compute gravity against the
   target galaxy (per `project_semantic_gravity_coinage.md`:
   `F = T(s1,s2) × M(s1) × M(s2) / d²`). If the top existing star has
   gravity above `MERGE_GRAVITY_THRESHOLD`, mark `merge_into=<star_id>`.
2. **Defeasibility check (PTX kernel).** Run `gre_defeasible_resolver.cu`
   (per `feedback_exploratory_grammar_deferred.md`) against candidate
   Grammar Galaxy entries the temporary star would create or imply. If it
   contradicts a high-strength rule, prefer the existing rule (mark
   `confidence_trit = -1`).
3. **Decision (trit output from resolver).** One of:
   - **Promote:** `confidence_trit = +1`, queue append to target galaxy
     JSONL, queue House 3D node via existing House-write path.
   - **Merge:** `confidence_trit = +1`, queue symlink insert into existing
     star (`symlink_targets += [<merge_into>]`), bidirectional (per
     `feedback_bidirectional_symlinks_norm.md`).
   - **Discard:** `confidence_trit = -1`, leave in temporary region for
     audit but tombstone so it won't be re-considered.

**Lane B — Weight consolidation** (runs alongside Lane A, same VRAM pass):

1. **Shadow-copy aggregation (GPU kernel).** Gather successful traces from
   the wake cycle — the per-lane deltas the nine-chain swarm + halting
   gate recorded as "this composition helped." Aggregate into TRM and
   specialist-region delta tensors in VRAM. Incremental — not a full
   retrain. Bounded by tick budget.
2. **Apply-or-reject (trit decision per delta).** Each delta goes through
   a contrastive acceptance pass (ternary + contrastive per
   `feedback_attention_is_ternary_plus_contrastive.md`). Accept (+1), hold
   for next tick (0), or reject (-1). Silence = hold. No default-apply.
3. **In-place weight update (GPU kernel).** Accepted deltas fold into the
   live TRM + specialist weights in VRAM. The updated weights are now the
   ones the next wake-cycle tick reads.

**Tail — Persist (I/O only, the one permitted outside-library step):**

After both lanes finish in VRAM:

- Append promoted stars to their target `/galaxies/*.jsonl`.
- Insert merge symlinks (bidirectional) into the existing JSONL rows.
- Checkpoint updated weights to `/K3D/Knowledge3D.local/weights/<ts>/`.
- Truncate `/galaxies/_temporary/<ingest_id>.jsonl` to an audit stub
  (first line = summary receipt). Clear the pending-ingest entry.

Lane B runs **every tick**, not just after an ingest. Wake-cycle shadow
copies accumulate continuously and get folded in during the next sleeptime
tick. Ingest is what triggers Lane A; Lane B runs because the wake cycle
ran.

### 10.3 Sovereignty — Sleeptime IS hot path (VRAM-first, disk-last)

Sleeptime is the **learning phase** — it consolidates two things simultaneously:

1. **Knowledge** — temporary stars → Galaxy entries / House symlinks.
2. **Logic** — TRM + specialist-region weight deltas earned during the
   wake cycle are folded in here.

Both happen **in VRAM**, fast, with the same sovereign discipline as the
wake-cycle composed head. Only the **final save to disk** (JSONL append for
stars, weight-checkpoint write) uses outside libraries, and only as I/O at
the tail of the pass. **From inside to outside:** compute first, persist
last. See `feedback_sleeptime_is_hot_path.md`.

Concretely:

- **Semantic-gravity probe** runs as a PTX/CUDA kernel over Galaxy data in
  VRAM — not a Python loop over stars.
- **Defeasibility resolution** runs `gre_defeasible_resolver.cu` on GPU;
  Python only launches.
- **Weight consolidation** (shadow-copy → TRM + specialist LoRA-like deltas)
  is a GPU pass, not a Python gradient update. Incremental — not a full
  retrain — and bounded in time so ticks stay short.
- **Decision output** (promote / merge / discard per star, weight delta
  accepted / rejected) comes out of the PTX resolvers, not from Python
  branches.
- **Only after** the VRAM pass completes, the I/O tail appends the promoted
  stars to their target Galaxy JSONL and checkpoints the updated weights.
  This is the one place outside libraries (json, pathlib, torch-save or
  equivalent) are allowed — file write only, no reasoning.
- **No Python fallbacks** (per `feedback_no_fallbacks_ever_including_sleeptime.md`).
  If the gravity probe fails, the star stays `confidence_trit = 0` and
  waits for the next tick. If the weight-delta kernel fails, weights stay
  where they were. **No default-promote. No default-discard. No
  default-apply.** Silence = stays pending.
- **Fast cadence.** Short ticks that run often beat long ticks that run
  rarely. A tick that takes minutes is a bug.
- **No sleep-phase fallback to a "Python consolidator" variant.** Any
  implementation where Python iterates stars, computes gravity, or updates
  weights is drift — flag and fix, don't route around.

### 10.4 Telemetry (mandatory)

Per `feedback_note_taking_everywhere.md`, every sleeptime tick emits:

```json
{
  "tick_id": "<uuid>",
  "pending_before": <int>,
  "promoted": <int>,
  "merged": <int>,
  "discarded": <int>,
  "pending_after": <int>,
  "duration_ms": <float>,
  "ingest_ids_touched": [...]
}
```

Silence is a bug.

### 10.5 Files

- **New or reused:** `knowledge3d/knowledgeverse/sleeptime_ingest.py` (or extend `sleep_time_micro.py`). Codex picks based on LOC / separation.
- **CLI:** `knowledge3d/tablet/sleeptime.py` — single-run-mode wrapper for testing.
- **No viewer changes.**

---

## 11. End-to-End Flow (Illustrative)

```
User                Tablet CLI          Daemon              Knowledgeverse       Sleeptime          House
 │                     │                    │                      │                   │               │
 │ ingest sample.pdf   │                    │                      │                   │               │
 ├────────────────────▶│ INGEST envelope    │                      │                   │               │
 │                     ├───────────────────▶│                      │                   │               │
 │                     │                    │ enqueue_ingest       │                   │               │
 │                     │                    ├─────────────────────▶│                   │               │
 │                     │                    │ ingest_receipt       │                   │               │
 │                     │◀───────────────────┤                      │                   │               │
 │ {"ok","ingest_id"}  │                    │                      │                   │               │
 │◀────────────────────┤                    │                      │                   │               │
 │                     │                    │                      │ proceduralize     │               │
 │                     │                    │                      │ via WINE v2       │               │
 │                     │                    │                      ├──(async)──────────┤               │
 │                     │                    │                      │ temp stars        │               │
 │                     │                    │                      │                   │               │
 │                     │                    │                      │     idle > 60s    │               │
 │                     │                    │                      │◀──────────────────┤               │
 │                     │                    │                      │ sleeptime tick    │               │
 │                     │                    │                      │ promote/merge     │               │
 │                     │                    │                      ├──────────────────────────────────▶│
 │                     │                    │                      │                   │ House write   │
 │ next chat turn      │                    │                      │                   │               │
 ├────────────────────▶│ CHAT envelope      │                      │                   │               │
 │                     │                    │ semantic gravity     │                   │               │
 │                     │                    │ finds new star       │                   │               │
 │ {"response":"..."}  │                    │                      │                   │               │
 │◀────────────────────┴────────────────────┴──────────────────────┴───────────────────┴───────────────┘
```

The loop closes without any new user action.

---

## 12. Sovereignty Compliance

| Concern | Status |
|---|---|
| No Python reasoning in hot path | WINE is envelope construction; daemon is dispatch; proceduralization runs via its own WINE |
| No numpy/cupy/scipy in WINE | WINE is pure dict/tuple construction |
| No external HTTP from WINE | `source_uri` is a string; fetching happens in `tablet/wine/proceduralize_wine.py` (approved), not here |
| No Python fallbacks in sleeptime | §10.3 — pending stays pending if probe fails; no default decision |
| Registry drift avoided | New `SURFACE_KIND_INGEST` is additive (§3.1); no opcode renumbering |
| Galaxy-first | All ingestion products land as Galaxy entries; nothing hardcoded |
| Composed head untouched | INGEST does not run the composed head — it's a queue-write lane; CHAT/MATH/QUESTION continue to run composed head unchanged |

---

## 13. Test Plan

### 13.1 Factory test (unit)

**File:** `tests/tablet/test_ingest_wine.py`

```python
def test_ingest_envelope_shape():
    from knowledge3d.tablet.wine.ingest_wine import ingest_envelope
    env = ingest_envelope(
        task_id="t-abc",
        source_uri="file:///tmp/sample.pdf",
        mime="application/pdf",
        lang_hint="pt-BR",
    )
    assert env.surface_kind == "INGEST"
    assert env.task["source_uri"] == "file:///tmp/sample.pdf"
    assert env.task["mime"] == "application/pdf"
    assert env.result_kind == "ingest_receipt"
    assert env.specialist == "ingest"
```

### 13.2 Daemon dispatch test (integration)

```python
def test_daemon_ingest_returns_receipt(tmp_path):
    from knowledge3d.daemon.main import handle_command_inprocess
    sample = tmp_path / "sample.md"
    sample.write_text("# hello\nfoo bar baz\n", encoding="utf-8")
    result = handle_command_inprocess({
        "command": "INGEST",
        "source_uri": f"file://{sample}",
        "mime": "text/markdown",
    })
    assert result["status"] == "ok"
    assert result["result_kind"] == "ingest_receipt"
    assert result["ingest_id"]
```

### 13.3 CLI smoke test

```python
def test_ingest_cli(tmp_path):
    import json, subprocess, sys
    sample = tmp_path / "sample.md"
    sample.write_text("# hello\n", encoding="utf-8")
    out = subprocess.check_output(
        [sys.executable, "-m", "knowledge3d.tablet.ingest",
         "--source", f"file://{sample}", "--mime", "text/markdown"],
        text=True,
    )
    result = json.loads(out)
    assert result["status"] == "ok"
    assert "ingest_id" in result
```

### 13.4 Input-gate exhaustive table

Tests for `_validate_ingest_input`:

- missing source → error
- oversized URI → error
- unsupported scheme (`gopher://`) → error
- unsupported MIME (`application/x-custom`) → error
- bad chunking shape → error
- well-formed → passes

### 13.5 Sleeptime end-to-end

```python
def test_sleeptime_promotes_small_ingest(tmp_path):
    # 1. ingest one tiny markdown with a fresh, unique claim
    # 2. call sleeptime.run_once()
    # 3. assert the House JSONL gained at least one row with source_ingest_id==<id>
    # 4. assert the temporary JSONL is truncated to its audit stub
```

Gated behind `pytest.mark.gpu` — requires `k3d-cranium` env + CUDA.

### 13.6 Sovereignty grep

```
grep -rnE "^(import |from )(ollama|requests|httpx|urllib\.request)" \
    knowledge3d/tablet/wine/ingest_wine.py knowledge3d/tablet/ingest.py
```

Must return zero matches.

### 13.7 Non-regression

Run `benchmarks/math_*.py` and the ARC sovereignty smoke after the new constants land. Pinned ARC 10/10, Math 20/20 must hold.

---

## 14. Open Questions (for Daniel, non-blocking)

1. **s3:// support.** Worth turning on by default, or gate behind a config flag? (MVP: gate behind flag, default off.)
2. **Stream-ingest for large PDFs.** If a PDF would exceed `INGEST_MAX_CHUNKS`, do we error or split into multiple ingest_ids? (MVP: error; stakeholders rarely need > 4k chunks per doc.)
3. **Temporary region persistence across daemon restart.** Survive? (MVP: yes — JSONL on disk. Sleeptime resumes where it left off.)
4. **Sleeptime cadence.** Is 60s idle right, or should it be House-constant driven? (MVP: literal, tune later.)
5. **Auditability vs. hygiene.** Do we keep discarded temporary stars on disk forever (auditable) or purge after N days? (MVP: keep — disk is cheap, audit is valuable for the paper.)

---

## 15. Files Touched (expected)

- **New:** `knowledge3d/tablet/wine/ingest_wine.py`
- **New:** `knowledge3d/tablet/ingest.py` (CLI)
- **New:** `knowledge3d/tablet/sleeptime.py` (CLI for explicit run-once)
- **New:** `knowledge3d/knowledgeverse/sleeptime_ingest.py` (or extend `sleep_time_micro.py`)
- **New:** `tests/tablet/test_ingest_wine.py`
- **New:** `tests/tablet/test_ingest_cli.py`
- **New:** `tests/tablet/test_sleeptime_ingest.py` (gpu-marked)
- **Edit:** `knowledge3d/bridge/headless_tablet.py` — add `SURFACE_KIND_INGEST`, extend `_SPECIALIST_CODES` + `_normalize_surface_kind`, add `TabletIngest.ingest_task`
- **Edit:** `knowledge3d/daemon/main.py` — add `cmd == "INGEST"` handler + `_make_ingest_id` + `_validate_ingest_input`; extract `handle_command_inprocess` if CHAT spec hasn't
- **Edit:** `knowledge3d/knowledgeverse/knowledgeverse.py` — add `enqueue_ingest(envelope)` entry point writing to temporary region

No deletions. No changes to existing ingestion modules (`enhanced_pdf_ingest.py`, etc.) — they remain the per-MIME parsers the proceduralizer WINE drives.

---

## 16. Codex Directives (actionable, in order)

1. Add `SURFACE_KIND_INGEST = "INGEST"` and extend the `_normalize_surface_kind` mapping / `_SPECIALIST_CODES` in `headless_tablet.py` per §3.1.
2. Implement `TabletIngest.ingest_task` per §5. Symmetric to `math_task` / `chat_task`.
3. Create `knowledge3d/tablet/wine/ingest_wine.py` per §4. Mirror `math_wine.py` exactly.
4. Create `knowledge3d/tablet/ingest.py` CLI per §9. Reuse `handle_command_inprocess` (extract it if not yet done per CHAT spec §8).
5. Extend daemon with `cmd == "INGEST"` handler per §6. Add `_make_ingest_id` and `_validate_ingest_input` helpers.
6. Add `knowledgeverse.enqueue_ingest(envelope) -> receipt` that writes the pending-ingest entry to `/K3D/Knowledge3D.local/galaxies/_temporary/<ingest_id>.jsonl` and returns `{"ingest_id": ..., "queued_chunks_estimate": ..., "telemetry": ...}`.
7. Implement sleeptime_ingest (§10) with idle/count/explicit triggers. Use existing `gre_defeasible_resolver.cu` and the semantic-gravity routines.
8. Write tests per §13. Run sovereignty grep (§13.6) last.
9. Run the pinned-benchmark non-regression (§13.7).
10. Report back with: WINE module diff, daemon handler diff, one end-to-end demo ingesting a small markdown + sleeptime run + a subsequent CHAT turn that references the new knowledge.

---

## 17. Definition of Done

- [ ] `SURFACE_KIND_INGEST` constant exists and is wired into `_normalize_surface_kind` + `_SPECIALIST_CODES`.
- [ ] `TabletIngest.ingest_task` exists and returns a conforming `TabletEnvelope`.
- [ ] `knowledge3d/tablet/wine/ingest_wine.py` exists and exports `INGEST_ROUTE_GALAXIES`, `build_ingest_route`, `build_ingest_task`, `ingest_envelope`.
- [ ] Daemon `INGEST` handler builds envelopes through `TabletIngest.ingest_task` — zero inline construction.
- [ ] `python -m knowledge3d.tablet.ingest --source file://... --mime ...` returns a JSON receipt with `ingest_id` and exit 0.
- [ ] Temporary-star region writes conform to §7.1 shape.
- [ ] Sleeptime tick promotes / merges / discards deterministically (§10.2); telemetry per §10.4 is emitted.
- [ ] All tests in §13 pass; sovereignty grep returns zero.
- [ ] ARC 10/10 and Math 20/20 still pinned post-landing.
- [ ] End-to-end demo (§11) reproducible — one ingest → one sleeptime tick → next chat turn sees new knowledge via semantic gravity.

---

## 18. Cross-Reference Map

| Concern | Authoritative doc |
|---|---|
| Envelope construction + TabletIngest contract | `knowledge3d/bridge/headless_tablet.py`, umbrella §2 |
| Proceduralization (downstream of this spec) | `TEMP/CLAUDE_SMART_PROCEDURALIZER_SPEC_V2_04.20.2026.md` |
| Chat dispatch pattern (mirror of) | `TEMP/CLAUDE_CHAT_WINE_SPEC_04.20.2026.md` |
| MeaningCentricStar schema | `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §2.1` |
| House write path | existing stargate_feeder / galaxies JSONL |
| Sleeptime kernel precedent | `knowledge3d/knowledgeverse/sleep_time_micro.py`, `knowledge3d/cranium/cuda/sleep_time_micro.cu` |
| Defeasibility at sleeptime | `feedback_exploratory_grammar_deferred.md`, `gre_defeasible_resolver.cu` |
| Semantic gravity | `project_semantic_gravity_coinage.md`, `feedback_semantic_gravity_between_stars.md` |
| Bidirectional symlinks | `feedback_bidirectional_symlinks_norm.md` |
| No-fallback sleeptime rule | `feedback_no_fallbacks_ever_including_sleeptime.md` |
| Telemetry mandate | `feedback_note_taking_everywhere.md` |
| Ternary confidence | `feedback_ternary_first_where_cheaper.md`, `feedback_attention_is_ternary_plus_contrastive.md` |
| Append-only registry (no opcode renumber) | `feedback_expand_not_replace_opcodes.md` |

---

## 19. Summary

One new WINE (`ingest_wine.py`). One new classmethod on `TabletIngest` (`ingest_task`). One new daemon command (`INGEST`). One new surface-kind constant (`SURFACE_KIND_INGEST`). One new temporary-star region shape (§7). One sleeptime transmutation contract (§10). Five CLIs/tests wrapping the above.

With this spec + the CHAT WINE spec + the PROCEDURALIZE v2 spec, all four old paradigms the umbrella names have a conforming Tablet WINE surface, and the living-AI ingestion loop closes without leaking Python into the hot path.
