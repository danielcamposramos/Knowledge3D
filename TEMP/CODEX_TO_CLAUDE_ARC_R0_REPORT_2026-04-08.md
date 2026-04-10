# Codex to Claude: ARC R0 Report
**Date:** 2026-04-08  
**Time:** 2026-04-08 18:06:44 -0300

## Summary

ARC R0 is now closed end to end for the ARC-2 local lane:
- task JSON in
- Grammar-galaxy `arc_rule` stars seeded
- sovereign CAS/SAS build + canonicalize + match + rule-apply path exercised
- GPU-native exact-match verification kernel compiled and launched
- honest baseline score printed and saved

The ARC-3 SDK lane is also truthfully wired, but the currently installable `arc-agi` package in this Python 3.10 env does **not** expose the live ARC-AGI-3 game runtime (`Arcade` / `make`) or `arcengine.GameAction`. So that lane reports an honest unavailable result instead of faking a live ls20 completion.

## Deliverables

### R0-A
- [arc_task_galaxy_seeder.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/arc_task_galaxy_seeder.py)

What is real:
- deterministic `grid_to_rpn()`
- `pair_to_grammar_rule()` producing `MeaningCentricStar` rules with `meaning_class="arc_rule"`
- `seed_task()` / `seed_tasks_directory()`
- compact ARC grid encoding via `grid_to_program_words()` for sovereign CAS pool ingestion

### R0-B
- [arc_verification.cu](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/cranium/kernels/arc_verification.cu)

What is real:
- `ArcGrid` compact GPU struct
- `arc_verify_candidate`
- `arc_score_candidates`
- bridge methods already live in [sovereign_bridges.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/cranium/bridges/sovereign_bridges.py):
  - `launch_arc_verify_candidate()`
  - `launch_arc_score_candidates()`

### R0-C
- [arc2_local_runner.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/arc2_local_runner.py)

What is real:
- loads official local ARC evaluation JSONs
- seeds Grammar rules from task demonstrations
- builds/canonicalizes sovereign roots
- uses `k3d_pattern_match` + `k3d_rule_apply`
- decodes predicted grids back from the CAS pool
- scores with `launch_arc_verify_candidate`
- writes baseline summary + competition-style submission artifact

### R0-D
- [arc3_sdk_agent.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/arc3_sdk_agent.py)

What is real:
- `sdk_status()` inspects the actual installed package surface
- `K3DAgent` wraps the K3D ARC3 chooser around the official SDK shape **when present**
- `run_ls20_test()` now returns a truthful unavailable result if the official game runtime is not present

## Verification

### PTX / compile
- `arc_verification.cu` compiles cleanly through the live PTX helper:
  - `arc_verification_ptx_bytes = 9410`

### Focused gate
- Command:
  - `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/pytest -q tests/test_arc_r0_surface.py tests/test_sovereign_sas_surface.py tests/test_sovereign_cas_surface.py tests/test_opcode_namespace_integrity.py -x`
- Result:
  - `33 passed in 7.40s`

### ARC-2 local runner
- Command:
  - `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python benchmarks/arc2_local_runner.py --max-tasks 20 --summary-output /K3D/Knowledge3D.local/results/arc_r0_2026-04-08/arc2_local_summary.json --submission-output /K3D/Knowledge3D.local/results/arc_r0_2026-04-08/arc2_local_submission.json`
- Dataset:
  - `/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation`
- Honest score:
  - `0 / 20 = 0.0`

Artifacts:
- [arc2_local_summary.json](/K3D/Knowledge3D.local/results/arc_r0_2026-04-08/arc2_local_summary.json)
- [arc2_local_submission.json](/K3D/Knowledge3D.local/results/arc_r0_2026-04-08/arc2_local_submission.json)

### ARC-3 ls20 SDK test
- `sdk_status()`:
  - `package_available = true`
  - `game_surface_available = false`
  - `game_action_available = false`
  - `import_error = "arc_agi installed without Arcade/make runtime"`
- `run_ls20_test()` honest result:
  - `steps = 0`
  - `levels_completed = 0`
  - `score = 0.0`
  - `transport = "unavailable"`

## Truthful gaps

### ARC-2
- The current R0 rule chooser is still literal and weak:
  - every one of the first 20 tasks landed in `nearest_training_pair`
  - zero exact demonstration-rule matches were enough to solve a held-out test case
- That is acceptable for R0 because the closed loop is now real and measurable.
- The next score lift belongs to R1:
  - abstracted transformation induction
  - refinement loop over generated candidates
  - richer Grammar rules than direct demonstration-output replay

### ARC-3
- The spec’s older `arc.make(...)` expectation is stale relative to what is currently installable here.
- In this env:
  - `pip install arc-agi==0.0.7` succeeds
  - import requires a `typing.Self` compatibility patch on Python 3.10
  - the package exposes dataset surfaces (`ARC1Evaluation`, `ARC2Evaluation`)
  - it does **not** expose the ARC-AGI-3 game runtime (`Arcade` / `make`)
  - `arcengine` is not available for Python 3.10 here
- Therefore I kept the ls20 result truthful and unavailable instead of routing it through a fake “official” transport.

### Environment note
- Installing `arc-agi==0.0.7` upgraded `numpy` in the managed env to `2.2.6`.
- Focused ARC/CAS/SAS tests stayed green after that change.
- Some optional packages in this env declare `<2.0` constraints, so this should be treated as an environment risk to reconcile later.

## Protected ingest
- Still alive and untouched:
  - PID `101379`
  - status `Ssl+`
  - elapsed at recheck `03:26:14`

## Recommended next step
1. Keep this R0 baseline as the paper’s first honest number.
2. Start R1 on the ARC-2 lane:
   - abstract transformation templates instead of full-grid replay
   - candidate generation + `arc_score_candidates`
   - use SAS Grammar retrieval beyond nearest-pair fallback
3. Revisit ARC-3 SDK once a Python 3.12-compatible env with the real game runtime exists.
