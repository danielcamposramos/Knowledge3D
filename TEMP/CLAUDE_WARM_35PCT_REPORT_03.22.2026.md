# Claude Handoff — Warm-Boot 35% Validation

**Date:** 2026-03-22  
**Session ID:** `full-ee0836e235e2`  
**Mode:** Warm boot, sovereign path  
**Evidence:**
- Log: `/tmp/k3d_warm_35pct_03.22.2026.log`
- Run state: `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- Sleep-time journal: `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`

## 1. Boot / Persistence

- Warm boot confirmed:
  - `Warm boot: House loaded (247974 entries across 24 galaxies)`
- This was **not** a cold rebuild.
- Adaptive swarm checkpoints were loaded from:
  - `/K3D/Knowledge3D.local/checkpoints/adaptive_swarm`
- No `Incremental update:` line appeared in the log.
  - Under the current runner logic, that implies **0 new entries were added** during the warm-boot incremental pass.

## 2. Final 35% Results

| Suite | Score | Accuracy | Elapsed |
|------|------:|---------:|--------:|
| ARC | `2 / 42` | `4.76%` | `820.061s` |
| Math | `4 / 500` | `0.80%` | `1361.599s` |
| GSM8K | `6 / 462` | `1.30%` | `1589.590s` |
| LHE | `2 / 35` | `5.71%` | `78.562s` |
| MMLU | `1129 / 4915` | `22.97%` | `22170.304s` |

**Combined:** `1143 / 5954 = 19.20%`

## 3. Comparison vs Previous Full 35% Cold Run

Reference previous completed clean cold-start full 35%:
- ARC `2 / 42`
- Math `3 / 500`
- GSM8K `7 / 462`
- LHE `1 / 35`
- MMLU `1106 / 4915`
- Combined `1119 / 5954 = 18.79%`

Warm-boot delta:
- ARC: `2 -> 2` (`+0`)
- Math: `3 -> 4` (`+1`)
- GSM8K: `7 -> 6` (`-1`)
- LHE: `1 -> 2` (`+1`)
- MMLU: `1106 -> 1129` (`+23`)
- Combined: `1119 -> 1143` (`+24`)
- Combined accuracy: `18.79% -> 19.20%` (`+0.41 pts`)

## 4. Sleep-Time / Contrastive Outcome

Sleep-time transaction succeeded overall:
- `success: true`
- Stage B routing/Jarvis updated:
  - `updated_count: 31657`
  - `updated_specialists: ["chat", "grammar", "math", "visual"]`

Jarvis summary highlights:
- `agreements: 512`
- `contradictions: 352`
- `briefs_consolidated: 128`
- `recommended_groups_by_task`:
  - `ARC_TASK: 5`
  - `LHE_TASK: 4`
  - `MATH_TASK: 4`
  - `MMLU_TASK: 4`

### True Contrastive Status

The new contrastive collection **did** gather both positive and negative counts:

| Specialist | Positives | Negatives | Result |
|-----------|----------:|----------:|--------|
| chat | `1129` | `3786` | failed |
| grammar | `2` | `33` | failed |
| math | `10` | `523` | failed |
| visual | `2` | `1` | failed |

Failure for all four:
- `argument 2: TypeError: Don't know how to convert parameter 2`

Checkpoint outcome:
- `checkpoint: {}`
- No contrastive checkpoint was saved from this run.

## 5. Key Findings

1. **Warm boot helped overall.**
   - The system improved from `18.79%` to `19.20%`.
   - The largest gain was MMLU: `+23`.

2. **The benchmark path is stable now.**
   - The full warm 35% run completed end-to-end.
   - No silent MMLU stop on this run.

3. **True contrastive collection is live, but the trainer is broken.**
   - Positives and negatives were both collected.
   - Training failed at adapter update time with a type-conversion error.

4. **Visual negative coverage is still effectively missing.**
   - ARC finished `2/42`, so the expected visual negatives were roughly `40`.
   - Logged visual negatives were only `1`.
   - This means most wrong ARC rows are still not producing usable negative-answer payloads for contrastive training.

5. **Incremental House behaved correctly for this run.**
   - Warm boot loaded the persisted House.
   - No evidence of a destructive rebuild.
   - No new-entry update was needed.

## 6. Next Fixes Suggested

1. Debug `AdaptiveSwarmTRM.train_specialist_contrastive(...)` / adapter gradient application.
   - The exact runtime failure is:
     - `argument 2: TypeError: Don't know how to convert parameter 2`
   - This likely sits in the gradient application path, not the pair collection path.

2. Fix ARC negative-answer persistence.
   - Visual got only `1` negative despite `40` ARC misses.
   - Wrong ARC answers need to be logged consistently so contrastive has material to push away from.

3. Re-run warm 35% after the contrastive trainer fix.
   - The benchmark plumbing is stable enough now to make that next cycle meaningful.
