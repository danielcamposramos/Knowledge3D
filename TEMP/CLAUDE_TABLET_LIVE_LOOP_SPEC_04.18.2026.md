# Tablet Live-Loop Wiring Spec (2026-04-18)

**Author:** Claude (architecture partner)
**Purpose:** Close the last sovereignty gap between `import knowledge3d.knowledgeverse`
(green) and `k3d-daemon` actually tick-ing a living AI that answers math,
generalities, and ARC-1/2 via its virtual tablet — the thing Daniel asked
for when he said *"a live game with a living AI, for now using its
virtual tablet to solve these questions."*

**Prerequisite:** Waves 1/2/3 are complete — see
`CLAUDE_LIVE_GAME_WAVE123_COMPLETE_04.18.2026.md`. Hot path
(`cranium/**` + `knowledgeverse/**`) is clean of numpy/cupy/scipy/sympy/torch.

**Blocker found (2026-04-18 during live-loop probe):** The tablet bridge
layer (`knowledge3d/bridge/**`) is **outside** the preflight hot-path
filter but is the transport between the sovereign kernel and the
orchestrator. Three files still import banned libraries, and the
`action_types.py` contract module (191 lines, 39 numpy refs, 1 cupy
fallback) was archived by the purge without a sovereign successor.

---

## 1. Violations Blocking the Live Loop

### 1.1 `action_types.py` — the PTX↔orchestrator contract

**Location when the purge started:** `knowledge3d/cranium/actions/action_types.py`
**Current location:** `Old_Attempts/2026-04-18/knowledge3d/cranium/actions/action_types.py`
**Loaded via:** `importlib.util.spec_from_file_location(...)` in
`knowledge3d/bridge/headless_tablet.py:16-24`
**Failure mode right now:** `FileNotFoundError` at daemon boot.

**Why we can't stub this one:**
`ACTION_BUFFER_DTYPE`, `ActionBuffer`, `ActionType` define the 288-byte
binary layout the PTX output layer writes to. Every tablet mutation
(nav vectors, dialogue tokens, memory writes) passes through this
struct. A NotImplementedError-stub would not just break one code path
— it would sever the entire PTX→Tablet transport.

**Sovereign successor:** pure-ctypes `Structure` with `_fields_` matching
the 288-byte layout described in Step 7.2 of the swarm chain spec. No
numpy dtype, no cupy fallback. The IntEnum `ActionType` is already
pure-stdlib.

**Size of the rewrite:** 191 lines → roughly the same, but
numpy.dtype(...) arrays become `ctypes.Structure` + `ctypes.Array`
types. The 39 numpy refs reduce to ~15 sites once you replace the
dtype descriptor with a `_fields_` list. cupy fallback is deleted.

---

### 1.2 `knowledge3d/bridge/headless_tablet.py`

**Banned imports:** `import numpy as np` (line 12)
**Why it's there:** orchestration of ActionBuffers returned from PTX,
plus tape recording and replay. Most numpy usage should be reducible
to ctypes or bytes + struct.

**Action:** audit all numpy call sites, map each to ctypes/bytes
equivalents, rewrite. This is ingestion-adjacent work (runs at each
tick, not mid-kernel), but it lives on the live-game hot path, so it
qualifies for preflight enforcement once scope is extended.

---

### 1.3 `knowledge3d/bridge/live_server.py` + `replay_builder.py`

**Banned imports:** `numpy` in both.
**Why they're there:** live WebSocket tape replay and history
serialization. Lower criticality than `headless_tablet` but still on
the tablet surface.

**Action:** same treatment — ctypes/bytes rewrite per file.

---

## 2. Proposed Scope Extension for Preflight

Once the three bridge files above are purged, extend the preflight
filter in `scripts/sovereignty_preflight.sh`:

```bash
FILES="$(find knowledge3d/cranium knowledge3d/knowledgeverse knowledge3d/bridge \
  -type f -name '*.py' \
  -not -path 'knowledge3d/cranium/tests/*' \
  -not -path 'knowledge3d/cranium/ocr/*' \
  -not -path 'knowledge3d/bridge/tests/*' 2>/dev/null || true)"
```

(Same for the staged-mode grep pattern.)

`knowledge3d/tablet/**` and `knowledge3d/daemon/**` have already been
scanned — **zero violations**. The tablet Python ceiling is thin; the
bridge layer is where the remaining debt sits.

---

## 3. Work Order for Codex (Phase 7.5 — Tablet Purge)

| # | Task | Files | Est. lines changed |
|---|------|-------|---------------------|
| 1 | Resurrect `action_types.py` as pure-ctypes `Structure` | `knowledge3d/cranium/actions/action_types.py` (new) | ~200 |
| 2 | Update `cranium/actions/__init__.py` to re-export the new class | 1 file | ~3 |
| 3 | Remove the `importlib.util.spec_from_file_location` hack in `headless_tablet.py` — replace with a normal `from knowledge3d.cranium.actions.action_types import ...` | `knowledge3d/bridge/headless_tablet.py:16-27` | ~10 |
| 4 | Purge `numpy` from `headless_tablet.py` — audit call sites, rewrite with ctypes/bytes | `knowledge3d/bridge/headless_tablet.py` | ~50–80 |
| 5 | Purge `numpy` from `live_server.py` and `replay_builder.py` | 2 files | ~30–60 |
| 6 | Extend preflight to cover `knowledge3d/bridge/**` (minus `bridge/tests/**`) | `scripts/sovereignty_preflight.sh` | ~4 |
| 7 | Re-verify: `python -c "import knowledge3d.daemon.main"` green | — | 0 |
| 8 | Wire math query → Tablet → PTX → Tablet round-trip | existing infrastructure | ~30 (glue) |
| 9 | 50×5 validation gate run | — | 0 (uses existing benchmark runner) |

**Recommended sequencing:** do #1–#7 as one PR (mechanical purge), then
#8–#9 as a second PR (live-loop integration + validation). The split
matches Claude/Codex lane separation — the purge work is mechanical
and well-scoped for Codex; the live-loop integration is where
architecture judgment comes back into play.

---

## 4. Embodiment Gap Positioning

Per `project_embodiment_gaps_identified.md`, the six embodiment gaps
that block MVP live-in are:

1. Perception loop (sensor tick)
2. Action loop (actuator tick)
3. House↔Galaxy symlink binding
4. Tablet as primary interface
5. Observability trace per solve
6. Always-on daemon

The work in this spec covers **gaps 1 + 2 + 4** (perception via
Tablet, action via ActionBuffer round-trip, Tablet as the MVP
interface). **Gap 5** is already delivered (`execution_events.py`
resurrected in Wave 2). **Gaps 3 + 6** remain after Phase 7.5 lands.

---

## 5. Claude's Assessment

This is the right shape of work for Codex's next batch. It's
mechanical (pure struct-layout rewrite + ctypes/bytes substitution),
well-scoped (3 files + 1 hack removal), and has a clear completion
gate (preflight extension passes + daemon imports green). I do not
recommend Claude write this code — not because the edits are hard,
but because they are mechanical, and the Claude lane should stay
focused on the live-loop integration architecture (#8) and the
validation-gate design (#9) that follow.

**If Daniel wants Claude to drive through Phase 7.5 directly rather
than handing to Codex, the spec above is the minimum diff. No
fallbacks, no try/except ImportError, no numpy lingering.**

---

## 6. Open Questions for Daniel

1. **Codex lane load** — is Phase 7 (canonical-ID + symlinks +
   meaning-star dedup) already enough for Codex, or can Phase 7.5
   stack on top?
2. **Preflight scope ceiling** — extend to `knowledge3d/bridge/**`
   now, or add it as a separate hardening PR after 7.5 ships?
3. **ActionBuffer layout freeze** — the 288-byte record in Step 7.2
   of the swarm chain spec; is that still authoritative, or has any
   post-purge ruling (BitNet b1.58, attention=ternary+contrastive,
   etc.) changed the field set?
