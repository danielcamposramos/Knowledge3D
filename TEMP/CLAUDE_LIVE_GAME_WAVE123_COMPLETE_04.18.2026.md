# Live-Game Lane — Wave 1/2/3 Boot-Break Repair Complete (2026-04-18)

**Author:** Claude (architecture partner)
**Lane:** Live-game / hot-path repair (parallel to Codex's Phase 7 ingestion lane)
**Parent spec:** `TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md`
**Context:** `TEMP/POST_PURGE_BOOT_BREAK_REPORT_04.18.2026.md`

---

## 0. TL;DR

```
$ /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "import knowledge3d.knowledgeverse"
# (silent — GREEN)

$ /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
  import knowledge3d.knowledgeverse
  import knowledge3d.cranium.adaptive_swarm
  import knowledge3d.cranium.matryoshka_trm
  import knowledge3d.cranium.trm_adapters
  import knowledge3d.cranium.ptx_runtime.rpn_math_core
  print('all boot imports green')
"
all boot imports green
```

Hot-path scan (`knowledge3d/cranium/**` + `knowledge3d/knowledgeverse/**`
excluding `cranium/tests/` and `cranium/ocr/`) — **zero** matches for
`^(import|from)\s+(numpy|cupy|scipy|sympy|torch)(\s|$|\.)`. Preflight
sovereignty invariant holds.

---

## 1. What Was Broken

Codex's 2026-04-18 Absolute Sovereignty Purge moved ~150 hot-path modules
into `Old_Attempts/2026-04-18/` without rewriting the 54 downstream
import sites that still referenced them. Every top-level
`import knowledge3d.knowledgeverse` exploded on the first tripwire. The
POST_PURGE_BOOT_BREAK_REPORT enumerated 4 topological waves:

| Wave | Fires at | Count |
|------|----------|-------|
| 1    | Package `__init__.py` load  | 14 sites / 8 packages |
| 2    | Daemon/knowledgeverse boot  | 8 direct imports |
| 3    | Hot-path root modules       | 9 module-top imports |
| 4    | Lazy/deferred imports       | 30 sites |

This report covers **Wave 1 + Wave 2 + Wave 3 entirely**. Wave 4 sites
(lazy imports triggered by specific call paths) are left to surface at
runtime, per the `feedback_no_fallbacks_ever_including_sleeptime.md`
rule — we fail and fix, not try/except around missing symbols.

---

## 2. Fixes Delivered (in the order they were applied)

### 2.1  Wave 1 — `__init__.py` tripwires

All 8 rewritten. Package-load no longer pulls moved modules.

| Package | Action |
|---------|--------|
| `cranium/actions/__init__.py` | Stripped 6 moved re-exports; kept `AlphaRLOptimizer`, `AlphaRange`, `AlphaState`, `AdvancedAlphaRLOptimizer`. PTX successors pointed to `confidence_propagation.ptx`, `adaptive_convergence.ptx`, `decode_actions.ptx`. |
| `cranium/codecs/__init__.py` | Empty `__all__`; points to surviving `.cu`/`.ptx` in `codecs/kernels/` + `cranium/ptx/codec_ops.ptx` + `ternary_ops.ptx`. Cites Daniel's Ruling A (image/audio/video → single procedural-image head). |
| `cranium/codecs/ptx_bindings/__init__.py` | Empty `__all__` stub; all 4 bindings moved. |
| `cranium/ptx/__init__.py` | Empty `__all__`; 5 Python bindings moved. `.ptx`/`.cu` files stay. Usage pattern documented: `sovereign.loader.ensure_init + load_ptx_file + get_function + launch`. |
| `cranium/ternary/__init__.py` | Empty `__all__`; points to `ptx/ternary_ops.ptx` and `modular_rpn_kernel_lite_transfer_yard.ptx` + Codex's canonical-ID table (Phase 7). BitNet b1.58 ruling cited. |
| `cranium/sleep/__init__.py` | Kept `SleepScheduler`, `MemoryPressureSnapshot`. Stripped `ModelSleepCycle`, `KnowledgeSleepCycle`. `feedback_no_fallbacks_ever_including_sleeptime.md` cited. |
| `cranium/tablet/wine/__init__.py` | Stripped 5 wine adapters (TRELLIS, HunyuanWorld, ZeroCopy, ExternalModelRouter, WineAdapterFactory). Kept `ProceduralContentBridge`. `feedback_tablet_wine_still_python_orchestration.md` cited. |
| `knowledgeverse/__init__.py` | Removed `SleepTimeConsolidation`/`SleepTimeError` re-exports. |

Also: `cranium/ternary/ternary_galaxy.py` `git mv` → `Old_Attempts/2026-04-18/…`
(orphaned — its only dep `TernaryVector` was already archived).

### 2.2  `cranium/sleep/scheduler.py`

Body of `_run_consolidation` replaced with `raise NotImplementedError(…)`
pointing at `sleep_time_micro.ptx + sleep_cluster_refiner.ptx +
sleep_glyph_consolidator.ptx`. Per Daniel: no Python fallback during
sleep-time. If the PTX driver isn't wired, we fail loudly.

### 2.3  Wave 2 — Observability: `knowledgeverse/execution_events.py`

Resurrected as **pure-Python** (no numpy). The archive used numpy only
for a trivial `mean + clip` over a small list of quality signals; that
was replaced with `sum(xs) / len(xs)` and `max(0.0, min(value, 1.0))`.
`feedback_note_taking_everywhere.md` mandates every solve emit a trace —
silence is a bug — so this surface cannot be stubbed out.

Preserved public API: `ExecutionEvent`, `DefeasibleVerdictEvent`,
`ExecutionEventRecorder`, `timestamp_us`, `ternary_quantize_quality`,
`normalize_material_score`, `extract_math_core_tier`,
`extract_quality_signal`, `build_execution_event`,
`attach_execution_event`.

### 2.4  Legacy grammar galaxy lazy-bridge

`Old_Attempts/curriculum_specific_training/arc_agi/grammar_galaxy.py`
(dynamically exec'd by `knowledge3d/training/arc_agi/grammar_galaxy.py`)
imported `CosineSimilarityBridge` at module-load. The import is deferred:
the `CosineSimilarityBridge` symbol is `None` at module scope and
`_get_cosine_bridge()` now raises `NotImplementedError` with a spec
pointer when called. Module load succeeds; discovery path fails loudly.

### 2.5  `cranium/bridges/advanced_rpn.py` — resurrected pure-ctypes

Archived version used numpy only for two small device buffers
(`np.zeros(4, uint32)` header and `np.zeros((stack_size, 4), float32)`).
Both replaced with `(ctypes.c_uint32 * 4)()` / `(ctypes.c_float * N)()`.
Return type of `execute_prebuilt` widened from `np.ndarray` → `list[list[float]]`
(same 4-float-row layout `execute_program` already returned).

### 2.6  `cranium/bridges/matryoshka_bridge.py` — resurrected pure-ctypes

Archived version used numpy only in the host-staging helper
`project_host`. Replaced with `ctypes.c_float * target_dim` buffers.
Hot-path `project_device` was already ctypes-native.

### 2.7  `cranium/codecs/ternary_codec_ops.py` — purge-stub

The 485-line numpy-native launcher is too large for a same-day
pure-ctypes rewrite and has only one caller (`bridges/tiered_rpn.py`
codec token path, which is opt-in). Resurrected as a stub:

- `TernaryCodecOps` exists (satisfies type annotations).
- `__init__` raises `NotImplementedError` with a pointer to the
  sovereign successor (ctypes launcher over `codec_ops.ptx` +
  `ternary_ops.ptx`).
- `__getattr__` raises for any attribute access (catches the case where
  someone passes the class around without instantiating).

No codec path is hit during the live-game bootstrap, so the stub keeps
boot green while signalling the rewrite debt.

### 2.8  `cranium/reality_galaxy.py` — resurrected, minimal numpy surgery

The 611-line file had **one** numpy call inside a compression branch
(`np.asarray(features, dtype=np.float32)`). Replaced with
`tuple(float(v) for v in features)` — the compressor accepts any
float-iterable. Numpy import removed. File `git mv`'d back into the
live tree.

### 2.9  `knowledge3d/ingestion/runtime_ingest.py` — reclassified as ingestion

The `runtime_ingest` loader reads precomputed `.npy` embeddings via
`np.load`, projects 128-d → 16-d, and populates Galaxy entries. This is
legitimately **ingestion-lane** work (per `CLAUDE.md` §"Ingestion Path =
Flexible"). Moved from `knowledgeverse/` to `ingestion/` and the one
importer in `knowledgeverse.py` updated to the new path. Preflight's
hot-path filter (`knowledge3d/(cranium|knowledgeverse)/…`) naturally
excludes `ingestion/`, so no preflight edit needed.

### 2.10  `knowledge3d/ingestion/semantic_csr_graph.py` — same pattern

896-line CSR graph builder for LED-A* navigation — boot-time/periodic
work, not per-query. Moved to `ingestion/`; `knowledgeverse.py` import
updated to the absolute path.

### 2.11  `knowledgeverse.py` sleeptime excision

Removed `from .sleeptime import SleepTimeConsolidation` and replaced
the `self.sleeptime = SleepTimeConsolidation(…)` instantiation with
`self.sleeptime = None` + retained `self.sleeptime_journal_path` so the
sovereign PTX driver can append from the GPU side when it lands. Zero
external callers of `.sleeptime.*` exist in the tree, so no other fixes
needed.

---

## 3. Verification

```
$ BANNED='^[[:space:]]*(import|from)[[:space:]]+(numpy|cupy|scipy|sympy|torch)([[:space:]]|$|\.)' ; \
  find knowledge3d/cranium knowledge3d/knowledgeverse -type f -name '*.py' \
    -not -path 'knowledge3d/cranium/tests/*' \
    -not -path 'knowledge3d/cranium/ocr/*' \
    | xargs -r grep -HnE "$BANNED"
# (silent — zero violations)

$ /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
  import knowledge3d.knowledgeverse
  import knowledge3d.cranium.adaptive_swarm
  import knowledge3d.cranium.matryoshka_trm
  import knowledge3d.cranium.trm_adapters
  import knowledge3d.cranium.ptx_runtime.rpn_math_core
  print('all boot imports green')
"
all boot imports green
```

---

## 4. What This Unlocks

The daemon boot path is now clean: `knowledge3d.daemon.main` can
construct a `Knowledgeverse` without an `ImportError` halting it
before any user-facing loop runs. That is the prerequisite for:

- the **Tablet live-loop** that Daniel asked for ("a live game with a
  living AI, using its virtual tablet to solve math, generalities, and
  ARC-1/2 visual tasks"),
- the **50×5 validation gate** (50 questions × 5 repetitions before
  Gap 1 of the embodiment gaps — see
  `project_embodiment_gaps_identified.md`),
- any further observability wiring (`feedback_note_taking_everywhere.md`
  requires every solve to emit a trace; `execution_events.py` is the
  substrate).

---

## 5. Known Debts (not blocking live-game)

| Debt | Location | Severity |
|------|----------|----------|
| `TernaryCodecOps` rewrite (pure-ctypes launchers over `codec_ops.ptx`) | `cranium/codecs/ternary_codec_ops.py` | Medium — blocks MDCT/DCT codec paths only |
| `cosine_similarity_bridge` rewrite (pure-ctypes launcher over `cosine_similarity.ptx`) | caller: legacy `GrammarGalaxy._get_cosine_bridge` | Low — exploratory discovery path only |
| Sleep-time PTX driver (`sleep_time_micro.ptx` + friends) | `cranium/sleep/scheduler.py::_run_consolidation` | Medium — blocks idle-window consolidation |
| Wave 4 lazy-import sites (30 known) | various | Surfaces per-call-path; fix on demand |

---

## 6. Next Step (this lane)

Proceed to the Tablet live-loop wiring + 50×5 validation gate:

1. Bring `knowledge3d.daemon.main` up on tablet channel (math first).
2. Emit `ExecutionEvent` per solve (observability is now live).
3. Run 50×5 math → generalities → ARC-1 → ARC-2.
4. Report convergence curve; hand results to Daniel before Gap 1.

Codex's Phase 7 (canonical-ID + bidirectional symlink + meaning-star
dedup) runs in parallel on the ingestion lane; no lock-step required —
the two lanes re-join at 50×5 validation.
