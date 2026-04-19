# Claude Architecture Directive — Absolute Sovereignty Purge

**Date:** 2026-04-18
**Author:** Claude (Architecture Partner)
**Scope:** Hot path (`knowledge3d/cranium/**`, `knowledge3d/knowledgeverse/**`).
**Directive origin:** Daniel, 2026-04-18 —
> *"Take out all numpy, cupy and torch code to Old_Attempts folder
> (this was ruled before - who knows when we might need, but this is
> not how k3d works!) - this is a different architecture even on how
> we store things inside it (from the layered symlinked style), not
> old paradigm… so, it's absolute (was supposed to be since the
> beginning…)"*

**Supersedes:** the soft-reading of Gate 0C in
`TEMP/CODEX_LAYER0_SEED_AND_PARALLEL_INGEST_04.18.2026.md`. Gate 0C is
**absolute**, not delta-scoped.

**Orchestration:** Claude-supervised, maximum delegation to **haiku**
sub-agents (mechanical moves, import enumeration) and **ollama
specialists** (refactor planning). **Sonnet** only where an actual
design judgment is required. No Codex in this purge (Daniel's
explicit directive: *"I do not trust Codex"*).

---

## 1. Why (the architectural clarification)

K3D stores knowledge differently from a numpy/torch-style AI system:

- **Layered + symlinked** — knowledge is a graph of content-hashed
  stars with bidirectional symlinks from Layer 0 (drawing primitives)
  up through Meaning, Rules, Meta.
- **Procedural RPN all the way** — every star carries `meaning_rpn`
  that executes on PTX, not on a CPU numpy path.
- **Multi-modal single procedural head** — image = procedural
  composition of drawing primitives + glyphs; audio = a dot-vector
  map over that procedural image; video = a sequence of those images.
  **One pipeline, not three.**
- **Hot path = PTX + Galaxy + RPN + TRM.** No Python arithmetic, no
  numpy arrays, no torch tensors, no cupy buffers.

Every `import numpy as np` on the hot path is an old-paradigm fallback
— a local optimum that blocks the global optimum. Moving this code to
`Old_Attempts/` is not deletion (the bits survive in git + in the
folder) — it is a clean break so the sovereign path has no
competitor.

---

## 2. Daniel's Direct Rulings (2026-04-18, this session)

**Ruling A — memory.py image/audio branches.**
> *"Take those python out after understanding the real intention
> behind it (we'll have a multi-model single procedural all the way
> head - audio and video are realized using the dot vector map -
> procedural image - and from there video is a sequence of those
> images)"*

Means: remove the `if o.kind == "image" | elif o.kind == "audio"`
branches in `knowledge3d/cranium/memory.py:154-157`. The real
intention is that all modalities collapse into a single procedural
surface — image is the primary, audio is a dot-vector map on top of
image, video is a time-indexed sequence of images. There is no
per-modality Python branching in the sovereign head.

**Ruling B — absolute, not delta.**
> *"so, it's absolute (was supposed to be since the beginning...)"*

The Gate 0C grep must return **zero** hits over the entire hot path.
Not zero new hits; zero hits. Period.

**Ruling C — move to `Old_Attempts/`, do not delete.**
> *"who knows when we might need"*

Git history preserves everything, but a physical folder also keeps
the bits one `ls` away, which matters for the engineer who ever needs
to diff against the old path.

---

## 3. The Purge — Classification Rules

Every file under `knowledge3d/cranium/**` and
`knowledge3d/knowledgeverse/**` that matches the violator grep gets
exactly one tag:

| Tag | Applies to | Action |
|-----|-----------|--------|
| **EXEMPT (tests)** | `**/tests/**`, `*.md`, `*.rst` | Leave in place. Per memory `feedback_no_numpy_no_bulk_libraries_sovereign_only.md`: *"Tests may use numpy; production CANNOT."* |
| **EXEMPT (ingestion)** | `knowledge3d/cranium/ocr/**` (ingestion-flexible per CLAUDE.md §Sovereignty Boundary) | Leave in place. |
| **REFACTOR** | Boot-critical production files: `knowledgeverse.py`, `cranium/memory.py`, `cranium/ptx/**`, `cranium/ptx_runtime/**`, `cranium/sovereign/**` | Remove the numpy/cupy/torch imports; replace logic with sovereign equivalent or explicit `NotImplementedError` stub pointing at the successor. |
| **MOVE** | Everything else (production code in cranium or knowledgeverse with a banned import) | `git mv <file> Old_Attempts/<same-relative-path>/<file>`. Preserve directory structure under `Old_Attempts/`. |

**Inventory source of truth:**
`TEMP/SOVEREIGNTY_PURGE_INVENTORY_04.18.2026.md` (built by the haiku
inventory sub-agent dispatched this session). That file is the
authoritative list. This directive references it; it is not
duplicated here.

---

## 4. `Old_Attempts/` Folder Contract

**Location:** `/K3D/GitHub/Knowledge3D/Old_Attempts/`

**Structure:**
```
Old_Attempts/
  README.md                          # why this exists + how to read it
  knowledge3d/
    cranium/
      <mirrored file tree>
    knowledgeverse/
      <mirrored file tree>
  INDEX.md                           # generated — one line per moved file with original path + move date + reason
```

**README.md contents (authored by this directive):**
1. Why this folder exists (absolute sovereignty).
2. These files are archived, not dead — bits may be relevant to future
   sovereign-path design (e.g., as test oracles for the replacement).
3. Do NOT import from `Old_Attempts/` in live code. It is not on the
   Python path for runtime; if an import resolves, that is a drift
   regression and must be flagged.
4. Pointer to `INDEX.md` for the move log.

**Git mv requirement:** use `git mv <src> <dst>` so history follows.
Never `cp && rm` — that loses blame.

---

## 5. REFACTOR Targets — Replacement Notes

### 5.1 `knowledge3d/cranium/memory.py:154-157`

**Current (violator):**
```python
if o.kind == "image":
    # numpy path
elif o.kind == "audio":
    # numpy path
```

**Replacement (per Daniel's Ruling A):**
- Remove the branches entirely. No per-modality Python dispatch.
- Every object entering memory goes through the unified procedural
  surface: `procedural_image_rpn` (primary), which carries an optional
  `audio_dot_vector_map` field for audio-bearing objects and an
  optional `frame_sequence_refs` for video-bearing objects.
- The hot path fetches the RPN and executes it on PTX. There is no
  kind-based branch in Python.

### 5.2 `knowledge3d/knowledgeverse/knowledgeverse.py:21`

**Current (violator):** top-level `import numpy as np`.

**Replacement approach:**
- Enumerate every numpy use in the file.
- For each use, identify whether it is:
  - **(a) load-bearing math** → replace with RPN-on-PTX call through
    `rpn_math_core` (sovereign) or `ctypes` raw pointer passing.
  - **(b) type hint / typing only** → replace with Python builtin or
    `typing.Any`; drop the import.
  - **(c) dead code** → delete the branch.
- Boot must continue to succeed after the refactor. If a numpy use is
  genuinely load-bearing and has no sovereign replacement yet, insert
  `raise NotImplementedError("sovereign successor pending — see
  TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md §5.2")` rather
  than keeping the import. Fail loud, not silently.

### 5.3 `knowledge3d/cranium/ptx/**`, `ptx_runtime/**`

Expected numpy usage: buffer shape inspection, dtype conversion for
PTX kernel args. Replacement: direct `ctypes` array pointers +
explicit dtype strings. The PTX launchers do not need numpy; they
need `(void*, size, dtype_code)`.

### 5.4 `knowledge3d/cranium/sovereign/**`

Ironic that "sovereign"-named modules have cupy/numpy imports. Per
inventory, loader.py has ~7 nested cupy imports. Refactor each to
`ctypes.CDLL("libcuda.so.1")` calls directly, matching the v1/v2
binding pattern memory-noted in
`project_sovereign_bitnet_attention_smoke_pass.md`.

---

## 6. Execution Orchestration

Claude supervises. Sub-agent dispatch plan:

### Phase 1 — Inventory (haiku, dispatched this session)

Produces `TEMP/SOVEREIGNTY_PURGE_INVENTORY_04.18.2026.md`. Claude
reviews and confirms classification. If any file is mis-classified,
Claude reclassifies in a single edit to the inventory.

### Phase 2 — Build Old_Attempts skeleton (haiku, parallel-safe)

One haiku sub-agent creates `Old_Attempts/README.md`, the empty
directory skeleton, and an empty `Old_Attempts/INDEX.md`. Does NOT
move any files yet.

### Phase 3 — Execute MOVE batch (haiku, parallelized by top-dir)

Multiple haiku sub-agents in parallel, each scoped to one top-level
directory (e.g., `cranium/bridges/`, `cranium/specialists/`,
`cranium/codecs/`, `cranium/actions/`, `cranium/ptx_runtime/` —
though ptx_runtime is REFACTOR not MOVE; see §5.3).

Each sub-agent:
1. Reads its slice of the inventory (files tagged MOVE under its dir).
2. For each file: `git mv <src> Old_Attempts/<src>`.
3. Appends one line to `Old_Attempts/INDEX.md`:
   `- <original_path> → Old_Attempts/<original_path> (2026-04-18)`
4. Returns a terse completion report.

### Phase 4 — REFACTOR boot-critical files (ollama `ask_coder` or sonnet)

Each REFACTOR target is its own dispatch:
- `knowledgeverse.py` refactor → **sonnet** (load-bearing; needs
  design judgment on what to stub vs what to replace).
- `cranium/memory.py` refactor → **ollama ask_coder** (scoped:
  remove the image/audio branches, replace with the unified
  procedural path from §5.1).
- `ptx/**` and `ptx_runtime/**` refactors → **haiku** batch (mostly
  mechanical: numpy → ctypes arrays).
- `sovereign/**` refactor → **sonnet** (the v1/v2 libcuda binding
  pattern is non-trivial; Claude reviews every diff).

Claude reviews every REFACTOR diff before it lands.

### Phase 5 — Verify (Claude)

```bash
# MUST return zero:
grep -rn --include='*.py' -E "^\s*(import (numpy|cupy|scipy|sympy|torch)\b|from (numpy|cupy|scipy|sympy|torch)\b)" \
    knowledge3d/cranium/ knowledge3d/knowledgeverse/ \
    --exclude-dir=tests --exclude-dir=ocr

# Nested-import check:
grep -rn --include='*.py' -E "^\s+(import (numpy|cupy|scipy|sympy|torch)\b|from (numpy|cupy|scipy|sympy|torch)\b)" \
    knowledge3d/cranium/ knowledge3d/knowledgeverse/ \
    --exclude-dir=tests --exclude-dir=ocr
```

Both must return zero. If non-zero, iterate on the offender.

### Phase 6 — Boot-break report (expected state)

After purge, `python -c "import knowledge3d.knowledgeverse"` will
almost certainly fail with ImportErrors because Phase 3 moved
transitively-imported files. **This is expected and correct.** The
failure trace is the work queue for rebuilding the sovereign head.

Claude produces a `TEMP/POST_PURGE_BOOT_BREAK_REPORT_04.18.2026.md`
with the first 50 ImportErrors in topological order. That becomes
the input to the next phase: sovereign-head rebuild.

---

## 7. What This Purge Does NOT Do

- Does not touch ingestion path (`knowledge3d/ingestion/**`) — still
  flexible per CLAUDE.md sovereignty boundary.
- Does not touch `cranium/ocr/**` — ingestion-flexible.
- Does not touch `**/tests/**` — tests may use numpy per memory.
- Does not touch viewer / tablet frontend / daemon entry points.
- Does not rewrite the sovereign head end-to-end — it creates the
  clean slate on which the rewrite proceeds.

---

## 8. Sovereignty Invariants (permanent, starting now)

After this purge lands, the following pre-commit check runs on every
commit touching `knowledge3d/cranium/**` or
`knowledge3d/knowledgeverse/**`:

```bash
#!/bin/bash
# scripts/sovereignty_preflight.sh
OUT=$(grep -rn --include='*.py' -E \
  "^\s*(import (numpy|cupy|scipy|sympy|torch)\b|from (numpy|cupy|scipy|sympy|torch)\b)" \
  knowledge3d/cranium/ knowledge3d/knowledgeverse/ \
  --exclude-dir=tests --exclude-dir=ocr 2>/dev/null)
if [ -n "$OUT" ]; then
  echo "SOVEREIGNTY_VIOLATION:"
  echo "$OUT"
  exit 1
fi
```

Hook this into `.git/hooks/pre-commit` (local) and CI (if applicable).
Once the hot path is clean, no numpy ever re-enters without a
ruling-level decision.

---

## 9. Success Criteria

1. Zero hits on the Phase-5 greps (both top-level and nested).
2. `Old_Attempts/` exists with correct structure, README, INDEX.
3. Every REFACTOR diff reviewed by Claude before landing.
4. No silent workarounds. No `# type: ignore` tricks to hide
   imports. No conditional `try: import numpy except: pass`.
5. Boot-break report produced and hands the next phase a concrete
   work queue.

---

## 10. Forbidden

- Using `cp && rm` instead of `git mv` (loses blame/history).
- Leaving any `import numpy`/`cupy`/`torch` in hot-path production
  code, even commented out or inside `try/except` stubs.
- Silent fallback wrappers (`if numpy_available: ... else: ...`)
  that re-introduce the library on a configuration flag.
- Deleting files in lieu of moving. Old_Attempts preserves bits.
- Codex involvement in this directive (per Daniel's explicit
  instruction).

---

## 11. One-sentence summary

Absolute purge: move every hot-path numpy/cupy/torch production file
to `Old_Attempts/`, refactor the handful of boot-critical survivors,
ship a pre-commit guard, accept the boot break as the input queue
for sovereign rebuild — supervised by Claude, executed by haiku and
ollama specialists, sonnet only where true design judgment is
needed.
