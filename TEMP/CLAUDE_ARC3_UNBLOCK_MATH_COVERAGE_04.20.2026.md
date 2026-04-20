---
date: 2026-04-20
author: Claude (pilot mode, Codex limit-locked)
status: ARC3 import unblocked; math galaxy coverage expanded; zero regression
commit: 70c465c3
---

# ARC3 Unblock + Math Coverage Expansion — 04.20.2026

## What landed

### 1. `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` (restored)

Was moved to `Old_Attempts/2026-04-18/` during the sovereignty purge
(torch import). Restored at its live path with:

- `_detect_gpu_count()` re-implemented without torch: reads
  `CUDA_VISIBLE_DEVICES` directly (sovereign rigs report 1; tests still
  monkeypatch).
- New class `ARC3EpisodeGPURing` — VRAM-resident ring of 128 frames ×
  64-byte `ARC3FrameStruct` = **8 KiB VRAM** per episode. `seed_frame()`
  now also does a single `memcpy_htod` per tick through the ring. No
  `dtoh` on the hot path.
- `ARC3EpisodeGalaxy.bind_for_trm_inference()` returns `{gpu_ptr,
  capacity, stride_bytes, frame_count, write_index, total_bytes}` for
  PTX kernel binding.
- Graceful degradation: ring is built lazily; if CUDA is unavailable or
  `K3D_ARC3_DISABLE_GPU_RING=1`, the Python-side deque path still works
  (that's how the 15 `test_arc3_living_memory.py` tests pass without a
  CUDA context).

### 2. `knowledge3d/knowledgeverse/knowledgeverse.py` (math coverage)

`_runtime_materialize_math_answer` now calls new helper
`_math_query_galaxies()` which returns `Math/Grammar/Reality/Tool` **+**
every discovered `proceduralized_*` / `Book_*` galaxy on disk. At
`/K3D/Knowledge3D.local/galaxies/` that adds 5 Book_ corpora
(BiologyAtlas, LanguageFoundations, MathematicsPrimer, PhysicsHandbook,
ToolManual) and 2 proceduralized corpora (`gsm8k_train_10`,
`mmlu_val_10`) — previously unreached.

Fallback extension:
- `_pick_math_operator_from_grammar` ranks operator candidates by
  keyword-hit count, not first-match. "than / more than / fewer than"
  forces subtraction even if additive cues appear (comparative
  difference bias).
- `_build_math_fallback_program` emits multi-operand `a b + c +` chains
  when ≥3 numeric literals + additive/multiplicative cues are present.

## Verification

| Benchmark | Before | After | Δ mat. | Δ acc. |
|-----------|--------|-------|--------|--------|
| GSM8K 10q | 1/10, 8 numeric + 2 empty | 1/10, 9 numeric + 1 empty | **+1** | 0 |
| MMLU 10q  | 2/10 | 2/10 | 0 | 0 |
| Math 10q  | 1/10 | 1/10 | 0 | 0 |

Zero regression. One more GSM8K case now materializes (via the
multi-operand fallback), though the numeric answer is still wrong. The
true lift is the **RPN template frontier** — template coverage for
comparative-difference, unit-conversion, ratio, and multi-step problem
classes — not fallback-widening.

Logs:
- `/tmp/gsm8k_10q_run2.log`
- `/tmp/mmlu_10q_run2.log`
- `/tmp/math_10q_run2.log`

## What this unlocks

1. **ARC3 live run** — `benchmarks/arc3_sdk_agent.py` imports cleanly
   again. The SDK agent can be driven against the live game server
   (`three.arcprize.org`) with the ARC-3 API key. Smoke-import confirmed.
2. **Book / proceduralized corpora** are now in the math query scope.
   When the RPN engine can interpret their `template_ref` fields (next
   phase), accuracy should lift.

## What's still open (audit backlog)

1. **Symlink dereferencing** — inert `metadata.symlink` fields on word
   and character stars (e.g. `"character_galaxy|word_galaxy"`) are not
   dereferenced anywhere. Needed for true cross-galaxy composition.
2. **Comparative-difference RPN template** — `math_template_comparative_difference_gpu`
   doesn't exist in Math.jsonl. The fallback now biases subtraction on
   "than" but a proper template would handle `(a - b)` with ordered
   operand selection (a = larger, b = smaller).
3. **Multi-GPU sleep consolidation** — `device_roles` routes crystallize
   / reinforce / object to separate GPUs if ≥4 are present, but the
   per-role code currently runs on the inference device. Needs PTX sleep
   kernels to actually span devices.
4. **Encoder-to-ring fusion** — `arc3_frame_encoder.py` does HtoD → PTX
   → DtoH per frame; the ring could receive the embedding directly via
   a PTX kernel that writes to ring-slot offset without any DtoH. Saves
   ~256 bytes/frame of CPU-host traffic.

## Sovereignty posture

- No new Python compute on the hot path. The ring is
  write-only-to-GPU; `seed_frame` emits exactly one `memcpy_htod` per
  call.
- Python-side rule crystallization (`_crystallize_rules`,
  `_classify_objects`) still lives in CPU land but is scoped to
  `run_micro_sleeptime` and `run_deep_consolidation` — the
  sleep/ingestion path, never inference. This matches the existing
  sovereignty debt in `knowledgeverse.py` (Phase D migration target).
- No new dependencies; torch stripped from the restored module.
