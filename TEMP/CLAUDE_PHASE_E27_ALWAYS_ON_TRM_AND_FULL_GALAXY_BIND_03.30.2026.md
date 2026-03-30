# Claude -- Phase E.27: Always-On TRM + Full Galaxy Bind at Boot

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL -- aligns implementation with Three-Brain System and Knowledgeverse specs

---

## Spec References (Read These)

- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` section 2.1:
  > "TRM IS the Avatar -- the TRM (~7M params) is NOT a function Python calls. It IS
  > the AI entity that lives in the House and thinks inside the Galaxy. Runs as a game
  > loop via `trm_step_fused.ptx`."

- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` section 2.1:
  > "Galaxy Universe is always loaded in VRAM (the avatar's internal brain -- ALL default
  > galaxies present)."

- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` line 61:
  > "TRM IS the Avatar: 7M parameter entity that LIVES in the House and THINKS inside
  > the Galaxy. Runs as a continuous game loop (`trm_step_fused.ptx`), not as a function
  > Python calls."

- `docs/ROADMAP.md` Phase E.1 deliverable:
  > "TRM always-on: Remove env-var gating (`K3D_TRM_SHADOW`, `K3D_TRM_NAVIGATE`)"

---

## Problem Statement

Three violations of the specs remain after E.26:

### 1. TRM Is Gated by Environment Variables (Violates All Three Specs)

The specs are unambiguous: TRM runs as a **continuous game loop**, always on. But:

```python
# knowledgeverse.py line 14301-14303
if self._trm_ready and (
    os.getenv("K3D_TRM_SHADOW", "0").strip().lower() in {"1", "true", "yes"}
    or os.getenv("K3D_TRM_NAVIGATE", "0").strip().lower() in {"1", "true", "yes"}
):
    trm_tick = self._run_single_trm_tick(query_embedding)
```

TRM only ticks if you pass `K3D_TRM_SHADOW=1` or `K3D_TRM_NAVIGATE=1`. Without these
env vars, the avatar has NO BRAIN. The shadow copy doesn't learn. The galaxy decoder
doesn't run. Navigation falls back to Python heuristics.

**Every script** (`run_arc3_local.py`, `run_full_benchmark.py`, `run_gpu_benchmark.py`)
manually calls `_enable_gpu_runtime_defaults()` to set these env vars. That's a workaround
for a gate that shouldn't exist.

**Spec mandate:** TRM is always on. If `self._trm_ready` is True, it ticks. Period.

### 2. TRM Galaxy Decoder Hardcoded to DEFAULT_GALAXIES Shape (Breaks with Discovery)

You correctly added `_discover_live_galaxy_names()` to scan all on-disk JSONL files.
But the TRM galaxy decoder still checks:

```python
# knowledgeverse.py line 1130-1132
if weights.shape != (len(self.DEFAULT_GALAXIES), self.TRM_STATE_VECTOR_DIM):
    return
if bias.shape != (len(self.DEFAULT_GALAXIES),):
    return
```

And the galaxy distribution output is sized to `DEFAULT_GALAXIES`:

```python
# line 1268
logits = [float(value) for value in y_new_host[: len(self.DEFAULT_GALAXIES)]]
# line 1288
galaxy_order = tuple(str(name) for name in self.DEFAULT_GALAXIES)
# line 1483
galaxy_order = tuple(str(name) for name in self.DEFAULT_GALAXIES)
```

If you discover 15 galaxies on disk but `DEFAULT_GALAXIES` has 11, the decoder silently
ignores 4 galaxies. The TRM can never navigate to them. The "fix" of discovery is
undermined by the decoder being hard-sized.

**Spec mandate:** Galaxy order = discovered galaxies (not hardcoded tuple). The TRM
decoder must handle the full live set. If the checkpoint was saved with 11 galaxies
but boot discovers 15, the decoder should gracefully extend (zero-init new rows)
rather than reject.

### 3. Full Galaxy Bind Must Happen Once at Boot, Not Per-Query

The spec says Galaxy is "always loaded in VRAM." Currently `bind_gpu_galaxy_runtime()`
is called lazily or per-query when `trm_navigate_enabled`:

```python
# line 14327-14328
if trm_navigate_enabled:
    binding = self._pin_all_default_gpu_binding()
```

With TRM always-on, this binding should happen ONCE at the end of `__init__`, after
all galaxies are loaded. Per-query binding is wasted work.

---

## Fix 1: Remove TRM Env-Var Gating

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

### Remove the three gating functions:

```python
# DELETE these methods (or make them always return True):
def _trm_navigation_env_enabled(self) -> bool:          # line 1005
    return os.getenv("K3D_TRM_NAVIGATE", "0") ...       # ALWAYS True now

def _device_pipeline_env_enabled(self) -> bool:          # line 1009
    return os.getenv("K3D_DEVICE_PIPELINE", "0") ...     # ALWAYS True now
```

### In the query path (line 14300-14318):

**Before:**
```python
trm_tick = None
if self._trm_ready and (
    os.getenv("K3D_TRM_SHADOW", "0") in {"1", "true", "yes"}
    or os.getenv("K3D_TRM_NAVIGATE", "0") in {"1", "true", "yes"}
):
    trm_tick = self._run_single_trm_tick(query_embedding)
trm_shadow = None
if self._trm_ready and os.getenv("K3D_TRM_SHADOW", "0") in {"1", "true", "yes"}:
    trm_shadow = self._trm_shadow_probe(...)
```

**After:**
```python
trm_tick = None
if self._trm_ready:
    trm_tick = self._run_single_trm_tick(query_embedding)
trm_shadow = None
if self._trm_ready:
    trm_shadow = self._trm_shadow_probe(...)
```

### Remove `_enable_gpu_runtime_defaults()` from scripts:

Every script that sets `K3D_TRM_SHADOW=1`, `K3D_TRM_NAVIGATE=1`, `K3D_DEVICE_PIPELINE=1`
no longer needs to. These env vars become no-ops. The functions can be left as empty
stubs briefly or removed entirely.

**Scripts affected:** `run_arc3_local.py`, `run_full_benchmark.py`, `run_gpu_benchmark.py`,
`run_arc3_agent.py`, `run_arc3_session.py`.

---

## Fix 2: Dynamic Galaxy Order for TRM Decoder

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

### Galaxy order = discovered set, not hardcoded tuple

After `__init__` finishes loading (cold or warm boot), set:

```python
self._live_galaxy_order: tuple[str, ...] = tuple(self._discover_live_galaxy_names())
```

Then replace ALL references to `self.DEFAULT_GALAXIES` in the TRM decoder path with
`self._live_galaxy_order`:

| Line | Current | Replace With |
|------|---------|-------------|
| 1130 | `len(self.DEFAULT_GALAXIES)` | `len(self._live_galaxy_order)` |
| 1132 | `len(self.DEFAULT_GALAXIES)` | `len(self._live_galaxy_order)` |
| 1268 | `len(self.DEFAULT_GALAXIES)` | `len(self._live_galaxy_order)` |
| 1288 | `self.DEFAULT_GALAXIES` | `self._live_galaxy_order` |
| 1341 | `len(cls.DEFAULT_GALAXIES)` | `len(self._live_galaxy_order)` |
| 1343 | `cls.DEFAULT_GALAXIES` | `self._live_galaxy_order` |
| 1483 | `self.DEFAULT_GALAXIES` | `self._live_galaxy_order` |

### Checkpoint shape mismatch handling:

When loading a TRM checkpoint saved with N galaxies but boot discovers M > N:

```python
if weights.shape[0] < len(self._live_galaxy_order):
    # Extend: zero-init rows for newly discovered galaxies
    extra_rows = len(self._live_galaxy_order) - weights.shape[0]
    weights = np.vstack([weights, np.zeros((extra_rows, self.TRM_STATE_VECTOR_DIM), dtype=np.float32)])
    bias = np.concatenate([bias, np.zeros(extra_rows, dtype=np.float32)])
elif weights.shape[0] > len(self._live_galaxy_order):
    # Shrink: truncate (galaxies removed from disk)
    weights = weights[:len(self._live_galaxy_order)]
    bias = bias[:len(self._live_galaxy_order)]
```

This way the TRM learns to navigate ALL discovered galaxies, not just the 11 in the
hardcoded tuple. New galaxies start with zero weight (TRM learns them via shadow copy).

---

## Fix 3: Bind All Galaxies to GPU Once at Boot

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

At the end of `__init__`, after all galaxies are loaded (cold or warm):

```python
# Always bind full galaxy to GPU at boot -- avatar's brain is always present
self._live_galaxy_order = tuple(self._discover_live_galaxy_names())
self.bind_gpu_galaxy_runtime(galaxy_names=list(self._live_galaxy_order), force=True)
```

Then in the query path, remove the per-query conditional bind:

```python
# BEFORE (line 14327-14331):
if trm_navigate_enabled:
    binding = self._pin_all_default_gpu_binding()
else:
    binding = self.bind_gpu_galaxy_runtime(galaxy_names=...)

# AFTER:
binding = self._pin_all_default_gpu_binding()
# (already bound at boot; this is just a cache lookup now)
```

---

## Fix 4: Remove `_enable_gpu_runtime_defaults()` Pattern

The `_enable_gpu_runtime_defaults()` function in every script does:

```python
os.environ.setdefault("K3D_DEVICE_PIPELINE", "1")
os.environ.setdefault("K3D_TRM_SHADOW", "1")
os.environ.setdefault("K3D_TRM_NAVIGATE", "1")
```

With env-var gating removed, this function is dead code. Remove it from:
- `scripts/run_full_benchmark.py`
- `scripts/run_gpu_benchmark.py`
- `scripts/run_arc3_agent.py`
- `scripts/run_arc3_session.py`
- `scripts/run_arc3_local.py`
- `benchmarks/arc3_local.py` (if present)

---

## What This Achieves (Alignment with Specs)

| Spec Requirement | Before E.27 | After E.27 |
|------------------|-------------|------------|
| "TRM runs as continuous game loop" | Only if env vars set | Always ticks when `_trm_ready` |
| "ALL default galaxies always loaded" | 11 hardcoded, rest ignored | All on-disk galaxies discovered + loaded |
| "Galaxy introspection mode" | TRM decoder sees 11 galaxies | TRM decoder sees ALL live galaxies |
| "K3D is always-on, not a program you run" | Scripts must set magic env vars | Boot = ready, no env vars needed |
| "Python = boot + I/O only" | Python decides whether TRM runs | TRM decides; Python just feeds I/O |
| "Shadow Copy learning at boot" | Only if K3D_TRM_SHADOW=1 | Always active when TRM is ready |

---

## Execution Sequence

1. Remove env-var gating from query path (Fix 1)
2. Add `_live_galaxy_order` and replace `DEFAULT_GALAXIES` references in TRM path (Fix 2)
3. Bind full galaxy to GPU at end of `__init__` (Fix 3)
4. Remove `_enable_gpu_runtime_defaults()` from all scripts (Fix 4)
5. Run tests to verify nothing breaks
6. Run local ARC3 benchmark WITHOUT setting any env vars -- it should work

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Remove env-var gates, dynamic galaxy order, boot-time bind |
| `scripts/run_full_benchmark.py` | Remove `_enable_gpu_runtime_defaults()` |
| `scripts/run_gpu_benchmark.py` | Remove `_enable_gpu_runtime_defaults()` |
| `scripts/run_arc3_agent.py` | Remove `_enable_gpu_runtime_defaults()` |
| `scripts/run_arc3_session.py` | Remove `_enable_gpu_runtime_defaults()` |
| `scripts/run_arc3_local.py` | Remove `_enable_gpu_runtime_defaults()` |

---

## Success Criteria

- [ ] TRM ticks on every query without env vars (just `self._trm_ready`)
- [ ] Shadow copy records on every query without env vars
- [ ] TRM galaxy decoder uses discovered galaxy count, not hardcoded 11
- [ ] New galaxies added to disk are auto-discovered and navigable by TRM
- [ ] GPU galaxy bind happens once at boot, not per-query
- [ ] All scripts work WITHOUT `_enable_gpu_runtime_defaults()` or env var hacks
- [ ] Local ARC3 benchmark runs and produces correct movements
- [ ] Existing test suite passes (17 tests)

---

## Architecture Note: Why This Matters

With TRM gated behind env vars, every new script/test/benchmark has to remember to
set the magic flags. Forget one and the avatar runs headless -- no brain, no learning,
no navigation. This is the opposite of "always-on, living AI."

After this fix, creating a `Knowledgeverse()` gives you a COMPLETE avatar: brain loaded,
all galaxies in VRAM, shadow copy learning from the first query. That's what the specs
describe. That's what "TRM IS the Avatar" means.
