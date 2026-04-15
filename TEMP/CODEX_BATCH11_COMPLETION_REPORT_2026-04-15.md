# Codex Batch 11 Completion Report

**Date:** 2026-04-15  
**Spec:** `TEMP/CODEX_BATCH11_KNOWLEDGE_WAVES_AND_OBSERVABILITY_2026-04-15.md`  
**Status:** Partially green. Track A/B/C landed enough to expose the next real blocker. The batch is not fully green because the required `GAME_2D` warm-up probe fails with `EMPTY_RECALL`.

---

## 1. What landed

### Track A — Knowledge waves

#### Parser / ingestion path

Landed:

- `knowledge3d/ingestion/hs_curriculum_parser.py`
- `scripts/ingest_hs_curriculum_remaining.py`
- `knowledge3d/ingestion/canonical_curriculum_loader.py`

Key changes:

- The curriculum parser now accepts the second markdown dialect already present in the Kimi files:
  - `**canonical_id:** ...`
  - `- **canonical_id**: ...`
  - `canonical_id: ...`
- Markdown field extraction now also accepts:
  - `**field:**`
  - `**field**:`
  - `field:`
- `surface_forms` parsing now handles bullet-style language rows (`- en: ...`).
- `source_file -> domain` inference now recognizes ARC primitives and tags them as `arc`.
- `scripts/ingest_hs_curriculum_remaining.py` now includes:
  - `TEMP/KIMI_ARC_REASONING_PRIMITIVES_CLUSTER_2026-04-14.md`
  - subkind: `arc_reasoning_primitives`

#### Real canonical write

Executed:

```bash
python3 scripts/ingest_hs_curriculum_remaining.py --write
```

Result:

- `written=82`
- `confirmed=82`
- `misses=[]`

Breakdown:

- `arc_reasoning_primitives = 33`
- `hs_applied_cs_health_psych_sociology = 14`
- `hs_crosscultural_glue = 5`
- `hs_earth_space_environmental = 16`
- `hs_history_geography_civics_economics = 12`
- `hs_languages_linguistics = 1`
- `hs_natural_sciences = 1`
- `hs_humanities_lit_phil_religion_arts = 0` (still parser-empty)

Artifact:

- `/K3D/Knowledge3D.local/reports/hs_curriculum_remaining_ingest.json`

### Track B — Observability

Landed:

- `knowledge3d/bridge/headless_tablet.py`
- `scripts/run_headless_tablet_benchmarks.py`

Key changes:

- Tablet session confirmation now writes a second event type:
  - `tablet_session_trace`
- Per-trace record now includes:
  - `item_id`
  - `task_id`
  - `suite`
  - `route_family`
  - `specialist_lane`
  - `stars_loaded_count`
  - `stars_touched`
  - `stars_recalled`
  - `opcodes_fired`
  - `halting_reason`
  - `raw_answer`
  - `normalized_answer`
  - `latency_ms`
  - `correct`
  - `program_id`
  - `surface_kind`
- Benchmark runner now emits:
  - `trace.{suite}.jsonl`
  - `trace.{suite}.coverage.json`
- Coverage report now includes:
  - `missing_item_ids`
  - `touched_but_never_recalled`
  - `recalled_but_wrong`
  - collapse attractor detection
  - specialist lane coverage
  - route family coverage
  - program id coverage
  - opcode coverage
  - PTX kernel inventory cross-check

### Track C — Routing and load verification

Landed:

- `knowledge3d/ingestion/canonical_curriculum_loader.py`
- `scripts/run_headless_tablet_benchmarks.py`

Key changes:

- `_target_galaxy_with_subkind()` now covers the missing Batch 11 domains:
  - civics
  - economics
  - macroeconomics
  - microeconomics
  - government
  - politics
  - social_studies
  - health
  - psychology
  - sociology
  - anthropology
  - arc
  - visual_reasoning
  - drawing/literature/philosophy/religion/arts/etc.
- `assert_canonical_curriculum_loaded()` is live.
- Loaded-by-galaxy accounting now counts by the actual resident galaxy, not by a possibly empty entry field.
- Warm-up probes were added for:
  - `GAME_2D`
  - `MATH`
  - `MMLU`
  - `LHE`
- Warm-up probes now persist `warmup_probes.json` before raising.

---

## 2. Verification

### Focused tests

Executed:

```bash
pytest -q tests/test_batch11_knowledge_waves.py tests/test_batch11_observability.py tests/bridge/test_headless_tablet.py
```

Result:

- `22 passed`

Also clean:

- `python3 -m py_compile ...`
- `git diff --check`

### Live one-process verification run

Executed:

```bash
python3 scripts/run_headless_tablet_benchmarks.py \
  --storage-root /K3D/Knowledge3D.local/batch11_verify_world \
  --log-dir /K3D/Knowledge3D.local/batch11_verify_logs \
  --output /K3D/Knowledge3D.local/batch11_verify_results.json \
  --arc2-count 1 --arc3-count 0 --mmlu-count 1 --gsm8k-count 1 \
  --lhe-count 1 --math-count 1 --amc-aime-count 1 \
  --omni-math-count 1 --imo-count 1 --use-enriched
```

Observed before failure:

- Canonical curriculum live-load succeeded:
  - `inserted=111`
  - `updated=0`
  - galaxies:
    - `Language=32`
    - `Math=1`
    - `Reality=45`
    - `Tool=33`
- The run reached the Batch 11 warm-up gate.

Failure:

- `RuntimeError: warmup_probe_empty_recall:GAME_2D`

Artifact:

- `/K3D/Knowledge3D.local/batch11_verify_logs/warmup_probes.json`

Captured probe result:

```json
{
  "GAME_2D": {
    "specialist_lane": "answer",
    "stars_touched": 1,
    "halting_reason": "EMPTY_RECALL",
    "route_family": "GAME_2D",
    "correct": false,
    "failure_code": "no_materialized_answer"
  }
}
```

Interpretation:

- The new curriculum and ARC stars are loading into the resident world.
- The benchmark runner is now correctly refusing to continue because the `GAME_2D` route cannot materialize an answer even in the one-item warm-up.
- This is the next real blocker. It is no longer hidden behind missing knowledge or missing observability.

---

## 3. What is green

- Batch 11 observability plumbing is live and tested.
- The coverage report logic is live and tested.
- Kernel inventory cross-check is live and tested.
- The curriculum loader’s routing map now covers the previously silent-drop domains.
- High-priority Track A slices are ingested enough to be real:
  - Natural/Earth-Space
  - History/Civics/Economics
  - Applied CS/Health/Psych/Sociology
  - ARC primitives
- Real canonical write completed with `82/82` confirmation.
- Live resident-world load assertion passed far enough for the benchmark runner to reach the warm-up gate.

## 4. What is not green

- `GAME_2D` warm-up does not materialize an answer.
- Humanities is still parser-empty (`0` rows).
- Languages/Linguistics is still under-harvested (`1` row).
- Natural Sciences is still under-harvested (`1` row from the physics/chemistry/biology file, despite better coverage from Earth/Space/Environmental).
- Because the warm-up gate stops on `GAME_2D`, Batch 11 did **not** yet produce full `trace.{suite}.jsonl` sidecars from a real sweep.

---

## 5. Load-bearing conclusion

Batch 11 did its job:

1. It loaded the next knowledge waves into canonical + resident memory.
2. It added the missing observability.
3. It proved the next failure is specifically on the `GAME_2D` route family, with:
   - a specialist lane selected
   - a touched star
   - but no answer materialization

That narrows the next move substantially.

---

## 6. Recommended next move

Do **not** jump to Batch 12 math refinement yet.

The next correct slice is:

1. Fix `GAME_2D` answer materialization on the live tablet/session path.
2. Rerun the Batch 11 one-item warm-up sweep until all four route families pass.
3. Then run the next 50-slice sweep with the new trace sidecars.
4. Only after that, draft Batch 12 from the trace evidence.

Secondary follow-up after the `GAME_2D` fix:

- deepen the parser for:
  - Humanities
  - Languages/Linguistics
  - Natural Sciences narrative sections still left on disk

