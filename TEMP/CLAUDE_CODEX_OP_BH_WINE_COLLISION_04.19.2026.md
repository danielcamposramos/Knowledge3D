# Codex Spec — Real Opcode Collision at 0x180-0x18C (OP_BH_* vs WINE)

**Date**: 2026-04-19
**Owner**: Claude (spec author) — surfaced during an attempted fix of a MVCIC-reported AVATAR_ACTION collision that turned out to be a hallucination
**Severity**: P0 for registry integrity, P1 for runtime (no sovereignty violation yet — both users of the range exist but are in different modules)
**Scope**: One real collision between a registry reservation and live behavior-op definitions, both at 0x180-0x18C.

**STATUS: ADJUDICATED 2026-04-19 — Option 1 (OP_BH_* wins, WINE relocates to 0x220-0x22F).**

**Daniel's ruling (verbatim):** *"we add opcodes, not delete, the one that was born first holds the lower addresses, we move the new feature to latter addresses."*

- OP_BH_* is already live in `rpn_opcodes.py` and `modular_rpn_kernel.cu` at 0x180-0x18C (born first).
- WINE I/O block relocates to **0x220-0x22F** (new reservation, append-only).
- Registry §7.3 / §11.2 row for 0x180-0x18F is marked `superseded` with VACATED note and the new WINE row is appended.
- No opcode numbers are deleted; OP_BH_* is canonical at the old WINE slots going forward.

---

## Context — why this exists as a separate spec

During the attempted execution of Paper A P0.2 (relocate AVATAR_ACTION opcodes from 0x150-0x154 to 0x180-0x19F, per MVCIC chain review), a sub-agent verification pass found:

1. **The AVATAR_ACTION 0x150-0x154 collision MVCIC reported does not exist.** No numeric AVATAR_ACTION opcodes are defined in the live codebase. The only `AVATAR_ACTION` reference is a Python list of Galaxy atom *string IDs* at `knowledge3d/knowledgeverse/action_embedding_loader.py:42`, which are not opcodes.
2. **But the proposed target range 0x180-0x19F is triply occupied** — and *that* is a real problem that pre-dates this session.

This spec documents the real collision.

---

## The three claimants to 0x180-0x18F

### Claimant A — Registry §7.3 / §11.2: WINE I/O Contract Block (authoritative, date 2026-04-18)

`docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` §7.3 and §11.2 reserve:

- `WINE_INGRESS_DECODE = 0x180`
- `WINE_EGRESS_ENCODE  = 0x181`
- `WINE_RESOLVE        = 0x182`
- `0x183-0x18F`: reserved for WINE expansion

Status: `active`. Reservation date: 2026-04-18. Per `feedback_opcode_range_reservation_protocol.md`, **the registry is the single source of truth** for opcode allocations.

### Claimant B — `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py:302-313`: OP_BH_* (live definitions)

```
OP_BH_PERCEIVE          = 0x180
OP_BH_SEEK              = 0x181
OP_BH_FLEE              = 0x182
OP_BH_ARRIVE            = 0x183
OP_BH_SEPARATE          = 0x184
OP_BH_APPLY_FORCE       = 0x185
OP_BH_BT_TICK           = 0x186
OP_BH_GOAP_PLAN         = 0x188
OP_BH_SLEEP_CHECK       = 0x189
OP_BH_BLACKBOARD_READ   = 0x18A
OP_BH_BLACKBOARD_WRITE  = 0x18B
OP_BH_PATHFIND          = 0x18C
```

These are runtime-live — any Python code importing from this module binds to these numeric values.

### Claimant C — `knowledge3d/cranium/kernels/modular_rpn_kernel.cu:2860-2943`: case handlers

```
case 0x180: /* BH_PERCEIVE    */ …
case 0x181: /* BH_SEEK        */ …
case 0x182: /* BH_FLEE        */ …
case 0x183: /* BH_ARRIVE      */ …
case 0x184: /* BH_SEPARATE    */ …
case 0x185: /* BH_APPLY_FORCE */ …
```

(Range 0x186-0x18C pending Codex verification — sub-agent reported handlers "through 0x185" explicitly; treat 0x186+ as possibly handler-less until Codex confirms.)

The kernel dispatches on these numeric values. If a program encodes `WINE_INGRESS_DECODE (0x180)` expecting WINE behavior, it gets `BH_PERCEIVE` behavior instead. Silent, wrong.

---

## Why this is a real collision, not a false alarm

- Both Claimant A and Claimant B claim the *same numeric slots*.
- The registry (A) is the canonical source per sovereignty doctrine; the live code (B, C) is what actually runs.
- A reasoning tick that dispatches opcode 0x180 goes to **whatever the kernel thinks it is** — which is `BH_PERCEIVE`, not `WINE_INGRESS_DECODE`. The registry's intent (WINE) and the runtime behavior (behavior-ops) disagree.
- This is the **0x1AD incident pattern** in `feedback_opcode_range_reservation_protocol.md`: opcodes got assigned outside the reservation protocol, and the registry was updated after-the-fact with a conflicting reservation.

---

## Adjudication — RULED Option 1 (2026-04-19)

Daniel's directive applies the **"born first holds lower addresses"** invariant from
`feedback_expand_not_replace_opcodes.md`:

> *"we add opcodes, not delete, the one that was born first holds the lower addresses,
> we move the new feature to latter addresses."*

### Who was born first?

- **OP_BH_*** is live in `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py:302-313` AND in
  `knowledge3d/cranium/kernels/modular_rpn_kernel.cu:2860+` case handlers. Runtime
  dispatch goes to these on 0x180-0x18C today. This is the "born" artifact.
- **WINE I/O block** (0x180-0x18F) is a reservation in `RPN_DOMAIN_OPCODE_REGISTRY.md`
  §7.3 / §11.2 dated 2026-04-18 with no runtime binding yet (no kernel dispatches
  WINE_INGRESS_DECODE as opcode 0x180; the only consumers are spec files and the
  pre-relocation WINE adapter wiring).

Therefore OP_BH_* holds the lower addresses. WINE is the "new feature" that moves.

### The ruling in one paragraph

- OP_BH_* **keeps** 0x180-0x18C (and any future sibling ops extend UP within
  0x18D-0x18F as Claimant B's natural home).
- WINE I/O block **relocates** to **0x220-0x22F** — a clean 16-slot window past the
  VIRTUAL_PAGE_* reservation overflow room (0x1D0-0x1FF + likely 0x200-0x21F
  headroom per §11.2 note).
- Concrete WINE reassignments:
  - `WINE_INGRESS_DECODE`: 0x180 → **0x220**
  - `WINE_EGRESS_ENCODE`:  0x181 → **0x221**
  - `WINE_RESOLVE`:        0x182 → **0x222**
  - 0x223-0x22F: reserved for WINE expansion (mirrors old 0x183-0x18F intent)
- Registry `§11.2` receives TWO new rows in a single commit:
  1. **0x180-0x18F row**: status `superseded` — note:
     *"VACATED 2026-04-19 — range already held by OP_BH_* (`rpn_opcodes.py:302-313`,
     `modular_rpn_kernel.cu:2860+`, born before 2026-04-18 registry entry). WINE I/O
     block relocated to 0x220-0x22F per Daniel's 'born-first' ruling. Opcodes
     previously listed here are NOT deleted — OP_BH_* is the canonical owner going
     forward. See `TEMP/CLAUDE_CODEX_OP_BH_WINE_COLLISION_04.19.2026.md`."*
  2. **0x220-0x22F row**: status `active` — owner `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §4.2-4.3 (WINE I/O, relocated 2026-04-19)` — date_reserved `2026-04-19`.
- Registry `§7.3` (WINE I/O block definition table) has its opcode-number column
  updated in place (the names and semantics are unchanged — only the numeric values
  change), with a footnote pointing to this spec and the §11.2 VACATED row.
- **No row is deleted. No opcode number is deleted.** OP_BH_* stays bound to
  0x180-0x18C; WINE_* re-binds to 0x220-0x222 everywhere it appears.

### Why 0x220-0x22F (not 0x1B0-0x1BF)

The original non-binding recommendation floated 0x1B0-0x1BF. That range turned out
to be largely claimed after re-grep: 0x1B0 single-slot reservation, 0x1B1-0x1B5
Attention Future Expansion, 0x1B6-0x1B9 bulk-lib-purge minted, 0x1BA-0x1BF
normalization/attention headroom. 0x1C0-0x1C5 IMAGE/SPARSE, 0x1C6-0x1CF physics
headroom, 0x1D0-0x1FF VIRTUAL_PAGE_*. First actually-clean 16-slot window is
0x220-0x22F (0x200-0x21F left as overflow room for VIRTUAL_PAGE_* which §11.2
anticipates).

---

## What Codex should do — execute Option 1 in ONE commit

Do NOT split across multiple commits. One atomic edit. Before/after greps in the
commit message.

### Step 1 — Verify claimant inventory is current

Run these first and paste the output into the commit body:

```bash
grep -nE '0x18[0-9A-F]' knowledge3d/cranium/ptx_runtime/rpn_opcodes.py
grep -nE '0x18[0-9A-F]' knowledge3d/cranium/kernels/modular_rpn_kernel.cu
grep -rnE '0x18[0-9A-F]' knowledge3d/cranium/ptx/ || true
grep -rnE '(WINE_INGRESS|WINE_EGRESS|WINE_RESOLVE)' knowledge3d/ docs/ TEMP/
grep -rnE 'OP_BH_' knowledge3d/ docs/ TEMP/
```

If any new PTX dispatch of 0x180-0x18F is found that isn't OP_BH_*-shaped, STOP
and report back before committing.

### Step 2 — Edit registry (`docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`)

1. **§7.3 WINE I/O block table**: change the numeric column for
   `WINE_INGRESS_DECODE` (0x180 → 0x220), `WINE_EGRESS_ENCODE` (0x181 → 0x221),
   `WINE_RESOLVE` (0x182 → 0x222). Update surrounding prose that hardcodes
   "0x180-0x18F" to "0x220-0x22F". Add a footnote: *"Relocated 2026-04-19 from
   0x180-0x18F per `TEMP/CLAUDE_CODEX_OP_BH_WINE_COLLISION_04.19.2026.md` —
   born-first ruling."*
2. **§11.2 block-reservation table**: do NOT edit the existing 0x180-0x18F row in
   place. Append two new rows at the end of §11.2 table (preserving chronological
   order):
   ```
   | `0x180` | `0x18F` | `TEMP/CLAUDE_CODEX_OP_BH_WINE_COLLISION_04.19.2026.md` — VACATED 2026-04-19. OP_BH_* is canonical owner (born first in `rpn_opcodes.py`/`modular_rpn_kernel.cu`). WINE relocated to 0x220-0x22F. | 2026-04-19 | superseded |
   | `0x220` | `0x22F` | `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §4.2-4.3` (WINE I/O, relocated from 0x180-0x18F) | 2026-04-19 | active |
   ```
   Mark the original 2026-04-18 `0x180-0x18F` row's status `active → superseded`
   (this is the only in-place edit to §11.2 and it is explicitly allowed by
   `§11.3` rule 2: `active → released → superseded`).

### Step 3 — Edit Python opcode table (`knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`)

OP_BH_* section at 302-313: **no change**. These stay at 0x180-0x18C.

Add a new WINE section (append-only, below OP_BH_* block):
```python
# --- WINE I/O (relocated 2026-04-19 from 0x180-0x182 per born-first ruling) ---
WINE_INGRESS_DECODE = 0x220
WINE_EGRESS_ENCODE  = 0x221
WINE_RESOLVE        = 0x222
# 0x223-0x22F reserved for WINE expansion
```

If any prior code in this file already binds `WINE_INGRESS_DECODE` / `WINE_EGRESS_ENCODE`
/ `WINE_RESOLVE` to 0x180-0x182, replace those bindings with the new values
(do NOT keep duplicate bindings).

### Step 4 — Edit CUDA kernel dispatch (`knowledge3d/cranium/kernels/modular_rpn_kernel.cu`)

OP_BH_* case handlers at 2860+: **no change**. They stay at `case 0x180` … `case 0x18C`.

If the kernel has existing WINE case handlers (grep for `WINE_INGRESS|WINE_EGRESS|WINE_RESOLVE`
in this file), update their case labels: 0x180 → 0x220, 0x181 → 0x221, 0x182 → 0x222.
If no WINE handlers exist yet in the kernel, skip this sub-step (they will be
minted against the new numbers when the WINE runtime lands).

### Step 5 — Edit any PTX files that dispatch WINE numerically

Grep for `0x180|0x181|0x182` in `knowledge3d/cranium/ptx/`. If any PTX file encodes
these values with WINE semantics (comments or symbol names will identify), relabel
to the new numbers. If zero PTX files reference WINE numerically, skip.

### Step 6 — Edit CANONICAL_REGISTRY_SPECIFICATION.md

Line references found: `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md:516, 524, 530, 534, 562`.
Update all `0x180 WINE_INGRESS_DECODE` / `0x181 WINE_EGRESS_ENCODE` / `0x182 WINE_RESOLVE`
to the new numbers. Keep the `0x180-0x183` range mention at line 562 pointing to the
new `0x220-0x222` range.

### Step 7 — Grep-sweep TEMP/ for stale WINE-at-0x180 references

```bash
grep -rnE '0x18[012][^0-9A-Fa-f]' TEMP/ docs/ knowledge3d/ | grep -iE 'WINE'
```

Any match: fix or annotate with a pointer to this spec. `TEMP/consistency_sweep_v3_04.18.2026.md:32`
is a known stale reference that should be updated.

### Step 8 — Commit in one atomic commit

Commit message template:

```
opcodes: relocate WINE I/O block 0x180-0x18F → 0x220-0x22F (born-first ruling)

Resolves collision between OP_BH_* (live at 0x180-0x18C in rpn_opcodes.py +
modular_rpn_kernel.cu) and WINE I/O reservation (registry-only since 2026-04-18).

Per Daniel's 2026-04-19 ruling: born-first holds lower addresses; OP_BH_* was
born in code before the WINE registry row, so WINE is the "new feature" that moves.

BEFORE:
  WINE_INGRESS_DECODE = 0x180  (conflict with OP_BH_PERCEIVE)
  WINE_EGRESS_ENCODE  = 0x181  (conflict with OP_BH_SEEK)
  WINE_RESOLVE        = 0x182  (conflict with OP_BH_FLEE)

AFTER:
  WINE_INGRESS_DECODE = 0x220
  WINE_EGRESS_ENCODE  = 0x221
  WINE_RESOLVE        = 0x222
  0x223-0x22F reserved for WINE expansion

Registry §11.2: original 0x180-0x18F row marked superseded (append-only); new
0x220-0x22F row appended with date 2026-04-19. OP_BH_* bindings unchanged.

Spec: TEMP/CLAUDE_CODEX_OP_BH_WINE_COLLISION_04.19.2026.md
```

### Step 9 — Post-commit verification

After the commit, run:

```bash
grep -rnE '0x18[012]' knowledge3d/ docs/ | grep -iE 'WINE'
grep -rnE '(WINE_INGRESS|WINE_EGRESS|WINE_RESOLVE)' knowledge3d/ docs/ | grep -vE '0x22[012]'
```

Both should return zero WINE hits (the first confirms no WINE code lives at 0x18x;
the second confirms no WINE name is still bound to a non-0x22x number).

---

## Related files

- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` — §7.3 and §11.2 (the reservation)
- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py:302-313` — the live Python-side definitions
- `knowledge3d/cranium/kernels/modular_rpn_kernel.cu:2860-2943` — the CUDA kernel case handlers
- `feedback_opcode_range_reservation_protocol.md` — the protocol this incident is another instance of
- `feedback_expand_not_replace_opcodes.md` — registry is append-only; freed slots stay `RESERVED` with a note
- `TEMP/mvcic_chain_paper_a_review_04.19.2026.md` — source of the (false) AVATAR_ACTION 0x150 alarm that led here
- `TEMP/CLAUDE_PAPER_A_SKELETON_DRAFT_04.19.2026.md` §Addendum / P0.2 — corrected record of how this was found

---

**Estimated effort**: 60-90 min once Daniel picks an option (code edits + registry update + grep verification).
**Blocks**: Tablet WINE adapter reliability (runtime may be silently dispatching BH_PERCEIVE instead of WINE_INGRESS_DECODE if anything ever encodes 0x180 expecting WINE).
**Blocked by**: Daniel adjudication (Option 1 / 2 / 3).
**Location**: `TEMP/CLAUDE_CODEX_OP_BH_WINE_COLLISION_04.19.2026.md`
