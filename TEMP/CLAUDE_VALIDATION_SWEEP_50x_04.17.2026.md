# Validation Sweep — 50× across Benchmarks (pre-embodiment baseline)

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-17
**For:** Codex
**Scope:** One run, 50 questions per benchmark (except ARC-3, which is live and already recorded). Purpose: prove the router + ring + daemon + TickDriver compose cleanly before we start the embodiment work (perceive/act/House↔Galaxy symlinks).

Runs are training. Whatever this sweep produces, sleep-time absorbs.

---

## 0. Pre-flight

1. **Daemon must be live** for the full sweep. One session, one tick driver, one galaxy runtime.
   - Start via `scripts/k3d_daemon.py` (or the systemd unit under `deploy/systemd/k3d-daemon.service` if that's the current entry point — grep to confirm).
   - Verify: `TICK_STATUS` over the daemon socket reports `ticks_total` advancing and `active=true`.
2. **One `bind_gpu_galaxy_runtime(galaxy_names=[...all default galaxies...])` at boot**, then do not rebind between benchmarks. The engine is supposed to hold all defaults resident.
3. **`K3D_BYPASS_GAME_LOOP` must be unset.** We are validating the ring path, not the legacy bypass.
4. **Sleep-time enabled throughout.** Each benchmark's completion triggers its own consolidation wave per the usual contract — do not disable.
5. Record `git rev-parse HEAD` and `wc -l knowledge3d/knowledgeverse/knowledgeverse.py` at the top of the report.

---

## 1. What to run

Five benchmarks, 50 items each, in this order (light-to-heavy so early failure is cheap):

| Order | Benchmark | Harness / Sender | Notes |
|-------|-----------|------------------|-------|
| 1 | MMLU | [benchmarks/mmlu_sender.py](../benchmarks/mmlu_sender.py) | 50 items across subjects, not 50 per subject |
| 2 | GSM8K | [benchmarks/math_sender.py](../benchmarks/math_sender.py) or equivalent | word problems — stresses navigator symlink-vote |
| 3 | Math competitions (AMC/AIME/MATH blend) | [benchmarks/math_sender.py](../benchmarks/math_sender.py) | 50 items; keep the blend natural |
| 4 | LHE (Last Humanity Exam) | [benchmarks/lhe_sender.py](../benchmarks/lhe_sender.py) | 50 items — multi-hop stress |
| 5 | ARC-AGI 1 (offline) | current local ARC-1 sender (check `benchmarks/arc_agi_*`) | 50 items |

Skip ARC-3 — it's live and scored already per Daniel.

All envelopes go through the daemon ROUTE path (ring). **No direct Python calls to `execute_task` from the senders.** If any sender still shortcuts the daemon, fix the sender before running — this sweep is meaningless if it doesn't exercise the live path.

---

## 2. Sanity criteria (fail-fast signals — not score thresholds)

Score thresholds are outcomes. These are **compositional health checks**. Any one failing means stop, diagnose, don't rush to the next benchmark.

1. **Ring is actually used.** For each benchmark, sample 3 random items and assert (via daemon logs or a trace) that each went through `enqueue_task → tick → wait_output_buffer`. Zero samples bypassing the ring is a pass.
2. **Meaning-class distribution is not degenerate.** Across all 250 items, the 8-class `meaning_class_dist` softmax must have argmax spread across ≥ 4 distinct classes. If every question collapses onto `FACTUAL_RECALL`, the navigator is broken regardless of score.
3. **No benchmark-label leakage in the wire envelope.** Pick 5 random wire-payloads across the 5 benchmarks (log them). Assert none contain `competition`, `dataset`, `source`, `task_type`, `surface_kind`. If any do, the drift regressed — stop and fix.
4. **TickDriver didn't starve or runaway.** At sweep end, `stats()["ticks_total"]` should be between `N × 50` and `N × 50000` where `N` is the wall-seconds of the run. Outside that window means driver is either paused or busy-looping.
5. **Janet = 18 still holds** when re-submitted at the start and at the end of the sweep. Drift during a sweep = sleep-time poisoning the galaxy; we need to know.
6. **No new `token in set(...)` reasoning** introduced by hot fixes during the run. Quick grep:
   ```
   grep -rnE "any\(token in \{|token in {" knowledge3d/knowledgeverse knowledge3d/daemon | wc -l
   ```
   Expected: **1** (the "a"/"an" article check at [navigator_specialist.py:1016](../knowledge3d/knowledgeverse/navigator_specialist.py#L1016)). Any higher = drift during the sweep.

---

## 3. What to capture per benchmark

For each of the 5 benchmarks, write `TEMP/validation_sweep_2026-04-17/<bench>.json`:

```json
{
  "benchmark": "mmlu",
  "items": 50,
  "correct": 0,
  "incorrect": 0,
  "errors": 0,
  "accuracy": 0.0,
  "median_latency_ms": 0,
  "p95_latency_ms": 0,
  "meaning_class_argmax_counts": {"FACTUAL_RECALL": 0, "...": 0},
  "sampled_envelopes": [ { "...": "3 random envelopes, verbatim" } ],
  "sampled_outputs":   [ { "...": "3 random outputs,   verbatim" } ],
  "ring_samples": [ { "request_id": "...", "enqueue_ts": "...", "wait_ticks": N, "output_ts": "..." } ],
  "tick_stats_at_end": { "ticks_total": N, "idle_ticks": N, "active_ticks": N, "error_ticks": 0 }
}
```

Plus one top-level `TEMP/validation_sweep_2026-04-17/SUMMARY.md` with:
- The 5 accuracies side-by-side.
- The six §2 sanity criteria as ✅/❌ with one-line justification each.
- Janet = 18 at T0 and T_end: both PASS/FAIL.
- `wc -l knowledge3d/knowledgeverse/knowledgeverse.py` + `git rev-parse HEAD` at the top.
- One-paragraph free-form commentary: anything unexpected, any pattern you noticed in failures, anything the sweep made you want to fix before embodiment.

---

## 4. Do not do

- **Do not re-tune anything mid-sweep.** If accuracy on MMLU is low, note it, keep going. We want one clean shot at the engine as-is.
- **Do not disable sleep-time.** Training is part of the signal.
- **Do not add new Python classifiers** to patch a failing item. If something's broken at the meaning-class layer, write it up — do not patch it in Python.
- **Do not run in parallel.** One sequential loop keeps the galaxy state coherent between items.
- **Do not burn tokens on a new spec.** If you discover a bug, note it in `SUMMARY.md`; Claude will spec the fix next round.

---

## 5. Acceptance — how I'll read the report

I care, in order:
1. **All six §2 sanity criteria green.** That's the composition health check; scores are secondary.
2. **Janet = 18 at T_end.** No galaxy poisoning.
3. **Meaning-class distribution non-degenerate.** The navigator is actually navigating.
4. **Scores as they are.** I'll compare to prior Phase B+ numbers (ARC-AGI 1 ~10/10 on the curated set, Math 20/20, LHE 10/10, GSM8K —/50, MMLU —/50 from CLAUDE.md §"Current State") to decide whether embodiment is safe to start or a regression needs to be fixed first.
5. **Anything surprising** in the free-form commentary.

If §2 is green and Janet holds, we move to embodiment (gaps #1-#3) regardless of absolute score, because scores follow knowledge density and we haven't added any new knowledge this round — we only cleaned the paths.

---

## 6. Standing protocol reminders

- Rule of three still applies when you hit something confusing during the sweep (MCP specs → MCP ptx → plan_task cloud).
- `kimi_swarm` / deep `ask_cloud` timeout = **240000 ms**.
- No numpy. No stubs. No Python reasoning. No fallbacks. One AI, one mind, one ring.

Report landing path: `TEMP/validation_sweep_2026-04-17/SUMMARY.md` + the 5 per-benchmark JSONs.
