# Codex Directive — WINE Rewire Verification + Offline Dataset Sweep

**Date:** 2026-04-18
**Role:** Codex — RUN-ONLY. No source edits.

---

## What Claude landed

Three benchmarks rewired to the sovereign Tablet WINE path (unconditional
boundary, no Python orchestration fallback). Meaning-based internal naming
throughout; benchmark class names stay (I/O adapter exemption).

- `benchmarks/last_humanity_exam.py` — `_ensure_tablet_boundary()`, unconditional
  tape; deleted dead `_apply_query_scope`/`_answer_open_ended`/`_to_float`/
  `_normalize_answer_text`/`_normalize_option_prediction`/`_open_ended_match`/
  `_enriched_reasoning`/`_empty_mind_reasoning`/`_extract_open_ended_prediction`.
  Internals renamed: `session_id=f"question_{...}"`, `suite_name="question"`,
  `log_event("question_success"|"question_failure")`.
- `benchmarks/mmlu.py` — same `_ensure_tablet_boundary()` pattern; removed the
  `Knowledgeverse.execute_task` Python fallback and `QUESTION_ROUTE_GALAXIES`
  scope-route import. `session_id=f"question_{...}"`, `suite_name="question"`,
  progress event now emits `surface_kind="QUESTION"`.
- `benchmarks/arc_agi_2.py` — dropped `ArcAgi2Adapter` + `TRMNavigator` +
  `TabletIngest` imports; removed the entire adapter construction branch and
  the archived `_solve_task_fallback`/`_seed_visual_knowledge`/`_grids_match`
  helpers; unified `_solve_task` → tablet submit; renamed
  `_arc_result_from_tablet_result` → `_result_from_tablet`.
  Internals renamed: `session_id=f"visual_grid_{...}"`,
  `suite_name="visual_grid_reasoning"`,
  `log_event("visual_grid_task_success"|"visual_grid_task_failure")`,
  progress event `surface_kind="GAME_2D"`.

### Known downstream breakage (NOT your job to fix)

- `scripts/validation_sweep_20260417.py:670,683,686` calls
  `MMLUBenchmark._normalize_option_prediction` and
  `LastHumanityExamBenchmark._normalize_option_prediction`/`_open_ended_match`.
  Those helpers were deleted as dead code. This dated validator will
  ImportError on those lines; ignore it — the tablet result already carries
  `correct` and `predicted_answer`. Claude will rework the validator in a
  follow-up. Do NOT re-add the deleted helpers.

---

## Your task — stratified 50-question sweep across all offline datasets

Run each surface kind through its benchmark with `max_tasks=50` (ARC uses
`max_tasks`; others use `max_questions`). All paths must go through the
tablet boundary. Capture the full summary JSON for each.

### Offline datasets under `/K3D/K3D_llama_cpp/datasets/`

| Surface kind | Dataset dirs                               | Benchmark class                        |
|--------------|--------------------------------------------|----------------------------------------|
| QUESTION     | `MMLU`                                     | `benchmarks.mmlu.MMLUBenchmark`        |
| QUESTION     | `last_humanity_exam`                       | `benchmarks.last_humanity_exam.LastHumanityExamBenchmark` |
| MATH         | `GSM8K`                                    | `benchmarks.gsm8k.GSM8KBenchmark`      |
| MATH         | `math` (Hendrycks MATH)                    | `benchmarks.math_competitions.MathCompetitionsBenchmark` |
| MATH         | `AMC-AIME`                                 | `benchmarks.math_competitions.MathCompetitionsBenchmark` (dataset-select by path) |
| MATH         | `Omni-MATH`                                | `benchmarks.math_competitions.MathCompetitionsBenchmark` (dataset-select by path) |
| GAME_2D      | `ARC-AGI-2-main`                           | `benchmarks.arc_agi_2.ARCAGI2Benchmark(dataset_version="arc_agi_2")` |
| GAME_2D      | `ARC-AGI-master` (ARC-1)                   | `benchmarks.arc_agi_2.ARCAGI2Benchmark(dataset_version="arc_agi")` |

ARC-AGI-3 is **excluded** — scored externally at `three.arcprize.org`, not
an offline corpus. The remaining dataset dirs (`audio*`, `clotho_raw`,
`coco_raw`, `dbnary`, `galaxy_geometry`, `house_zone*`, `lexicons`,
`msrvtt_dl_more`, `omw*`, `ud`, `vatex_raw`, `wordnet`) are ingestion-only
knowledge sources, not benchmarks — do **not** run them.

### Runbook

```bash
git log --oneline -5        # HEAD should be the WINE-rewire commit
git status --short          # expect this file + rewired modules + TEMP only

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs data/benchmarks/wine_sweep_${TS}

# Single-process, single-context — one living mind, one tablet.
# Each run builds its own tablet via _ensure_tablet_boundary.
bash scripts/check_single_context_invariant.sh

# QUESTION surface
bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json
from benchmarks.mmlu import MMLUBenchmark
b = MMLUBenchmark(max_questions=50)
s = b.run_benchmark(use_enriched=True)
open('data/benchmarks/wine_sweep_${TS}/mmlu.json','w').write(json.dumps(s, indent=2))
print('MMLU', s['correct'], '/', s['total_questions'])
" 2>&1 | tee logs/wine_sweep_${TS}_mmlu.log

bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
b = LastHumanityExamBenchmark(max_questions=50)
s = b.run_benchmark(use_enriched=True)
open('data/benchmarks/wine_sweep_${TS}/lhe.json','w').write(json.dumps(s, indent=2))
print('LHE', s['correct'], '/', s['total_questions'])
" 2>&1 | tee logs/wine_sweep_${TS}_lhe.log

# MATH surface
bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json
from benchmarks.gsm8k import GSM8KBenchmark
b = GSM8KBenchmark(max_questions=50)
s = b.run_benchmark(use_enriched=True)
open('data/benchmarks/wine_sweep_${TS}/gsm8k.json','w').write(json.dumps(s, indent=2))
print('GSM8K', s['correct'], '/', s['total_questions'])
" 2>&1 | tee logs/wine_sweep_${TS}_gsm8k.log

bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json
from benchmarks.math_competitions import MathCompetitionsBenchmark
for name, path in (('math','/K3D/K3D_llama_cpp/datasets/math'),
                   ('amc_aime','/K3D/K3D_llama_cpp/datasets/AMC-AIME'),
                   ('omni_math','/K3D/K3D_llama_cpp/datasets/Omni-MATH')):
    b = MathCompetitionsBenchmark(dataset_path=path, max_questions=50)
    s = b.run_benchmark(use_enriched=True)
    open(f'data/benchmarks/wine_sweep_${TS}/{name}.json','w').write(json.dumps(s, indent=2))
    print(name, s.get('correct'), '/', s.get('total_questions'))
" 2>&1 | tee logs/wine_sweep_${TS}_math.log

# GAME_2D surface
bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json
from benchmarks.arc_agi_2 import ARCAGI2Benchmark
for name, ver in (('arc2','arc_agi_2'), ('arc1','arc_agi')):
    b = ARCAGI2Benchmark(dataset_version=ver, max_tasks=50)
    s = b.run_benchmark(use_enriched=True)
    open(f'data/benchmarks/wine_sweep_${TS}/{name}.json','w').write(json.dumps(s, indent=2))
    print(name, s['correct'], '/', s['total_tasks'])
" 2>&1 | tee logs/wine_sweep_${TS}_arc.log
```

### Commit artifacts only

```bash
git status --short   # MUST show only:
                     #   data/benchmarks/wine_sweep_${TS}/*.json
                     #   logs/wine_sweep_${TS}_*.log
git add data/benchmarks/wine_sweep_${TS} logs/wine_sweep_${TS}_*.log
git commit -m "bench(wine-sweep): stratified 50-question sweep across offline datasets — ${TS}"
git rev-parse HEAD
```

No push. No amend. No `--no-verify`. No source edits — if a benchmark
fails, capture the traceback and hand back to Claude via the FAIL report
below.

---

## Reporting

### On PASS — one seven-line block per surface (three blocks total)

```
surface_kind = QUESTION
mmlu_50  = <correct>/50   (<accuracy>)
lhe_50   = <correct>/50   (<accuracy>)

surface_kind = MATH
gsm8k_50       = <correct>/50
math_50        = <correct>/50
amc_aime_50    = <correct>/50
omni_math_50   = <correct>/50

surface_kind = GAME_2D
arc1_50 = <correct>/50
arc2_50 = <correct>/50
commit = <git rev-parse HEAD>
```

### On FAIL — paste exactly:

```
command: <exact command>
exit_code: <number>
surface_kind: <QUESTION|MATH|GAME_2D>
dataset: <name>
stderr_tail_60:
<last 60 lines>
json_if_any:
<JSON contents or "none produced">
commit_head: <git rev-parse HEAD>
```

No prose, no speculation, no proposed fixes. Claude triages.

---

## Forbidden

- No source edits. No try/except wrappers. No PYTHONPATH/LD_PRELOAD overrides.
- No re-adding the deleted helpers (`_normalize_option_prediction`,
  `_open_ended_match`, `_apply_query_scope`, etc.).
- No push, amend, or `--no-verify`.
- `scripts/validation_sweep_20260417.py` is known-broken — skip it.
- The faked-artifact rule from
  `memory/feedback_codex_cannot_silent_fix_to_unblock.md` still stands.

---

## Post-compaction reload protocol

1. Re-read this file.
2. Re-read `memory/feedback_tablet_wine_still_python_orchestration.md`
   and `memory/feedback_python_dispatch_is_not_a_line_item.md`.
3. `git log --oneline -5` — top must be the WINE-rewire commit.
4. `git status --short` — only TEMP/ notes expected pre-run.

---

## One-sentence summary

Claude rewired LHE/MMLU/ARC-AGI-2 onto the unconditional tablet-WINE
path with meaning-based internal naming; your job is the honest 50-per-
surface sweep across the eight offline corpora and the per-surface
seven-line report.
