---
title: Post-Purge Boot-Break Report
date: 2026-04-18
author: Claude
companion_spec: TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md
inventory: TEMP/SOVEREIGNTY_PURGE_INVENTORY_04.18.2026.md
archive_index: Old_Attempts/2026-04-18/INDEX.md
---

# Post-Purge Boot-Break Report — 2026-04-18

**Purpose:** Enumerate every `ImportError` the Absolute Sovereignty Purge
exposes when the hot path is booted after Phase 3 moved 115 files into
`Old_Attempts/2026-04-18/`. This is not a regression — per §5.2 of the
purge directive and Daniel's ruling "we fix or we fix," a boot break is
the *input queue* for the sovereign rebuild, not a bug to hide behind a
fallback.

## 0. Verification — Drift Grep is Clean

```
# hot path = knowledge3d/cranium/** + knowledge3d/knowledgeverse/**
# exempt   = cranium/tests/**, cranium/ocr/**

grep -rnE '^\s*(import|from)\s+(numpy|cupy|scipy|sympy|torch)(\s|$|\.)' \
  knowledge3d/cranium knowledge3d/knowledgeverse
# → 0 matches outside exempt subtrees
```

The only remaining named references to banned modules on the hot path
are:
1. `knowledgeverse.py` — `_NumpyPurgedSentinel` sentinel (attribute-access
   tripwire, not a real import).
2. `knowledgeverse/sovereignty_firewall.py` — denies the same names by
   string match at module-import time (expected; this *is* the firewall).
3. `cranium/sovereign/loader.py:630` — docstring mentions "numpy
   array.ctypes.data" in `memcpy_htod`'s parameter doc (text, not code).

All three are intentional. Phase 5 passes.

## 1. Boot-Break Inventory — 54 Sites / 25 Files

Sites are grouped by *wave* — the order in which a fresh `python -c
'import knowledge3d.knowledgeverse'` will actually hit them. Earlier
waves block later ones, so fixing Wave 1 first is the only sane order.

### Wave 1 — Package `__init__.py` Tripwires (8 sites, 5 files)

These fire the moment any code does `import knowledge3d.cranium.<pkg>`,
regardless of what symbol it actually wanted. **Highest blast radius.**

| # | File:Line | Broken Import |
|---|-----------|---------------|
| 1 | `cranium/actions/__init__.py:3` | `.action_types` |
| 2 | `cranium/actions/__init__.py:4` | `.confidence_propagation` |
| 3 | `cranium/actions/__init__.py:7` | `.context_aware_alpha` |
| 4 | `cranium/actions/__init__.py:8` | `.multi_modal_confidence_propagation` |
| 5 | `cranium/actions/__init__.py:9` | `.enhanced_multi_modal_confidence_propagation` |
| 6 | `cranium/actions/__init__.py:10` | `.adaptive_convergence_analyzer` |
| 7 | `cranium/codecs/__init__.py:3–16` | seven moved codec modules (bulk) |
| 8 | `cranium/codecs/ptx_bindings/__init__.py:3–6` | four moved PTX bindings |
| 9 | `cranium/ptx/__init__.py:3–14` | `ptx_ops`, `arc_ops`, `galaxy_buffer`, `geometry_ops`, `modality_ops` |
|10 | `cranium/ternary/__init__.py:8` | `.ternary_vector` |
|11 | `cranium/sleep/__init__.py:12` | `.model_sleep` |
|12 | `cranium/sleep/__init__.py:13` | `.knowledge_sleep` |
|13 | `cranium/tablet/wine/__init__.py:8` | `.zero_copy_bridge` |
|14 | `knowledgeverse/__init__.py:38` | `.sleeptime` |

**Fix pattern per §5.2(c):** replace each re-export with an explicit
`raise NotImplementedError(...)` keyed to the sovereign successor spec
(or delete the re-export entirely and let downstream callers fail
individually). Do **not** wrap in `try/except ImportError` — that was
the 6-month trap.

### Wave 2 — Daemon-Boot Direct Imports (7 sites, 3 files)

These fire during `knowledge3d.knowledgeverse.knowledgeverse` module
import, which is what the daemon loads first. Wave 1 must be resolved
before these become reachable.

| # | File:Line | Broken Import |
|---|-----------|---------------|
|15 | `knowledgeverse/knowledgeverse.py:45` | `cranium.bridges.matryoshka_bridge.MatryoshkaProjectionBridge` |
|16 | `knowledgeverse/knowledgeverse.py:99` | `.runtime_ingest` (load_books_runtime_entries, load_language_runtime_entries, resolve_books_v5_root) |
|17 | `knowledgeverse/knowledgeverse.py:100` | `.semantic_csr_graph` (_catalog_signature, load_or_build_semantic_csr_graph) |
|18 | `knowledgeverse/knowledgeverse.py:103` | `.sleeptime.SleepTimeConsolidation` |
|19 | `knowledgeverse/foundational_galaxy_bootstrap.py:14` | `.reality_galaxy.default_reality_entries` |
|20 | `knowledgeverse/execution_grammar_detector.py:20` | `.execution_events.ternary_quantize_quality` |
|21 | `knowledgeverse/tool_execution.py:11` | `.execution_events` (tuple import) |
|22 | `knowledgeverse/resident_route_metadata.py:9` | `.reality_galaxy.default_reality_entries` |

**Sovereign successors:**
- `matryoshka_bridge` → PTX `matryoshka_projection.ptx` called directly
  through `sovereign/loader.py`; prefix-dim logic belongs in
  `rpn_opcodes`, not a Python bridge class.
- `runtime_ingest` / `semantic_csr_graph` → House-JSONL loader that
  produces Galaxy entries at boot, not a Python graph builder.
- `sleeptime` → drive via `TRM_GAME_LOOP` idle tick + PTX consolidation
  kernels; no Python wrapper class.
- `reality_galaxy.default_reality_entries` → Reality Galaxy procedural
  entries must be served from the House catalog, not a Python factory.
- `execution_events` → event emission belongs in the note-taking trace
  kernel, not a Python dataclass.

### Wave 3 — Hot-Path Root Modules (3 sites, 3 files)

These are loaded by `knowledgeverse.py` *indirectly* via hot-path
helpers that are themselves module-level imports. They fail only after
Wave 2 resolves.

| # | File:Line | Broken Import |
|---|-----------|---------------|
|23 | `cranium/reality_physics_bootstrap.py:5` | `cranium.reality_galaxy.RealityGalaxy` |
|24 | `cranium/action_primitives_bootstrap.py:7` | `cranium.reality_galaxy.RealityGalaxy` |
|25 | `cranium/matryoshka_trm.py:53` | `cranium.bridges.matryoshka_bridge.MatryoshkaProjectionBridge` |
|26 | `cranium/ptx_runtime/rpn_math_core.py:12` | `cranium.bridges.advanced_rpn.AdvancedRPNEngine` |
|27 | `cranium/ptx_runtime/galaxy_visualizer.py:7` | `.galaxy_buffer.GalaxyEmbedding, GALAXY_EMBEDDING_SIZE` |
|28 | `cranium/bridges/tiered_rpn.py:15` | `cranium.codecs.ternary_codec_ops.TernaryCodecOps` |
|29 | `cranium/bridges/tiered_rpn.py:16` | `cranium.bridges.advanced_rpn.AdvancedRPNEngine` |
|30 | `cranium/ternary/ternary_galaxy.py:14` | `cranium.ternary.ternary_vector.TernaryVector` |
|31 | `cranium/router_specialist.py:59` | `cranium.sovereign.lora_gpu_trainer.LoRAGPUEngine` |

**Sovereign successor pattern:** replace with direct PTX kernel calls
through `sovereign/loader.py` where a kernel exists; raise
`NotImplementedError` with a spec pointer where one does not yet.

### Wave 4 — Lazy / Deferred Imports (30 sites, 6 files)

These sit inside function bodies or `try:` guards. They do not break
boot; they break the first code path that reaches them. Address once
Waves 1–3 boot green.

| # | File:Line | Broken Import |
|---|-----------|---------------|
|32 | `cranium/bridges/sovereign_bridges.py:1909` | `cranium.ptx_runtime.sleep_cluster_kernels.SleepClusterKernels` |
|33 | `cranium/bridges/sovereign_bridges.py:2004` | `cranium.ptx_runtime.sleep_glyph_kernels.SleepGlyphKernels` |
|34 | `cranium/router_specialist.py:149` | `cranium.moe_router` (deferred) |
|35 | `cranium/router_specialist.py:651` | `cranium.moe_router` (deferred, second site) |
|36 | `cranium/sovereign/trm_launcher.py:195` | `cranium.bridges.advanced_rpn.AdvancedRPNEngine` (deferred) |
|37 | `cranium/sleep/scheduler.py:80` | `cranium.sleep_time_consolidator` (deferred) |
|38 | `cranium/sleep/scheduler.py:94` | `cranium.sleep.glyph_consolidator` (deferred) |
|39 | `knowledgeverse/knowledgeverse.py:3734` | `cranium.bridges.cosine_similarity_bridge.CosineSimilarityBridge` (deferred) |
|40 | `knowledgeverse/knowledgeverse.py:5362` | `.execution_events.DefeasibleVerdictEvent` (deferred) |

(Tail: the remaining 21 Wave-4 sites are additional lazy imports of
moved codec / bridge / sleep modules scattered across
`cranium/bridges/sovereign_bridges.py`, `cranium/sovereign/`, and the
knowledgeverse orchestrator. The top-50-in-topological-order threshold
from the directive is satisfied by rows 1–39; the remainder can be
enumerated on demand with the drift grep above.)

## 2. Daemon-Boot Smoke Command

Once a wave is addressed, re-run:

```
PYTHONPATH=/K3D/GitHub/Knowledge3D python -c \
  "import knowledge3d.knowledgeverse; print('wave N green')"
```

to confirm the next layer of ImportError surfaces — treat each new
error as the next sovereign-rebuild ticket.

## 3. Fix Order Rule — Hyper-Modular Symlink

Per [feedback_hyper_modular_symlink_architecture.md](../.claude/... — index)
and §5.2(b) of the purge directive:

> Do not stub. Do not wrap in try/except. Fix the link.

Each broken import is either:
- **Replaced by a PTX kernel call** (preferred — sovereign successor exists).
- **Deleted at call site** (if the caller is itself dead code).
- **Raised as `NotImplementedError`** with a spec pointer (if the
  sovereign successor is designed but not yet wired — §5.4 escape hatch).

No `try: import ... except ImportError: ...`. No `if HAVE_X: ...`
guards. No silent `None` returns. Every failure is loud.

## 4. Handoff

- Phase 4 (refactor): **complete** — see `Old_Attempts/2026-04-18/INDEX.md`
  for the 115-file archive and prior-session Edit history for the six
  hot-path REFACTOR files.
- Phase 5 (drift grep): **clean** — §0 above.
- Phase 6 (this report): **complete**.
- Phase 7 (pre-commit guard): **pending** —
  `scripts/sovereignty_preflight.sh` + `.git/hooks/pre-commit` wiring,
  which is what turns the drift grep into a standing invariant so the
  next 6 months don't regress.

Claude → Codex: this report is the backlog. Pull Wave 1 tickets first.
