# GSM8K Failure Audit — 2026-03-16

Source run:
- Root: `/tmp/k3d_track_c_full_guard`
- Env: `K3D_TRM_NAVIGATE=1`, `K3D_TRM_INFLUENCE_STRENGTH=0.5`
- Result: `2/10`

Observed passes:
- `gsm8k_0`
- `gsm8k_3`

## Failure Breakdown

### gsm8k_1: robe fiber total with `half that much`
- Expected: `3`
- Got: `I don't know`
- Failure mode: `HALTING_FAILURE`
- Selection steps:
  - `GSM8K fission: hit add (4 entries)`
  - `Swarm path result: reasoning_word_problem_fission -> [Grammar] operation_pattern_multiplication`
  - `GSM8K worker preview: gsm8k_worker_1 backward_chain -> 3`
  - `GSM8K answer consensus: 3 (struct=0.71, workers=1, weight=1.30, mean=1.90)`
  - `Halting gate: continue (top=2.01, gap=0.04, agree=1, flags=1,0,1,0)`
- Root cause: The correct numeric candidate is produced, but the system does not converge because the competing symbolic and malformed numeric candidates keep agreement too low. Reference resolution improved enough to surface `3`, but not enough to win halting.
- Suggested fix: Add a GSM8K halting policy that favors a single high-structure numeric consensus over symbolic `operation_pattern_*` strings when the goal type is scalar quantity.

### gsm8k_2: house flip profit after repairs and value increase
- Expected: `70000`
- Got: `0`
- Failure mode: `STRATEGY_FAILURE`
- Selection steps:
  - `GSM8K fission: hit sub (4 entries)`
  - `GSM8K goal typing: percentage_result via typed_fusion (part, percentage)`
  - `GSM8K worker preview: gsm8k_worker_2 fusion_chain -> 70`
  - `GSM8K worker preview: gsm8k_worker_3 clause_chain -> 130`
  - `GSM8K answer consensus: 0 (struct=0.86, workers=3, weight=2.98, mean=1.44)`
  - `Halting gate: halt (top=1.86, gap=0.23, agree=3, flags=1,1,1,1)`
- Root cause: The problem is routed as a percentage-output task instead of a profit-on-resale task. The system notices `150%`, but it never composes `purchase + repairs -> new value -> profit`.
- Suggested fix: Add a dedicated `markup_profit_after_repairs` strategy that explicitly computes improved value from total basis, then subtracts basis to yield profit.

### gsm8k_4: chicken feed final meal after two prior meals
- Expected: `20`
- Got: `0`
- Failure mode: `PARSE_FAILURE`
- Selection steps:
  - `GSM8K goal typing: remaining_after_spending via typed_fusion (count, total)`
  - `Morton locate: 0 candidates, using semantic seed fallback`
  - `Swarm path result: reasoning_word_problem_fission -> [Word] word_pt_sombrinha`
  - `GSM8K answer consensus: 0 (struct=0.69, workers=2, weight=2.52, mean=1.28)`
  - `Halting gate: halt (top=1.39, gap=0.04, agree=2, flags=1,1,1,1)`
- Root cause: The parse never binds `20 chickens * 3 cups/day = 60 cups total` before subtracting the `15` and `25` cup meals. Once that total is missed, search collapses into irrelevant Word-space neighbors.
- Suggested fix: Add a flock-total parse rule that multiplies per-agent daily requirement by flock size before applying residual subtraction.

### gsm8k_5: alternating full-price / discounted glasses
- Expected: `64`
- Got: `I don't know`
- Failure mode: `COMPOSITION_FAILURE`
- Selection steps:
  - `GSM8K fission: hit add (4 entries)`
  - `Swarm path result: reasoning_word_problem_fission -> [Grammar] operation_pattern_alternating_discount_pairs`
  - `GSM8K worker preview: gsm8k_worker_0 forward_chain -> 8.39999961853`
  - `GSM8K worker preview: gsm8k_worker_5 alt_add -> 82`
  - `GSM8K answer consensus: 8.39999961853 ...`
  - `Halting gate: continue (top=2.10, gap=0.04, agree=2, flags=1,0,1,0)`
- Root cause: The correct rule family is selected, but the arithmetic composition is wrong. The system does not form `8 pairs * (5 + 3) = 64`; it mixes raw percentages and item counts directly.
- Suggested fix: Refine `operation_pattern_alternating_discount_pairs` so it groups items into price pairs first, then multiplies the pair cost by the pair count.

### gsm8k_6: chained sheep ratios across three cities
- Expected: `260`
- Got: `30`
- Failure mode: `HALTING_FAILURE`
- Selection steps:
  - `GSM8K worker preview: gsm8k_worker_0 forward_chain -> 260`
  - `GSM8K worker preview: gsm8k_worker_1 backward_chain -> 30`
  - `GSM8K answer consensus: 260 (struct=0.93, workers=1, weight=1.34, mean=1.62)`
  - `GSM8K answer consensus: 30 (struct=0.86, workers=3, weight=3.26, mean=1.34)`
  - `Halting gate: halt (top=1.68, gap=0.20, agree=3, flags=1,1,1,1)`
- Root cause: The correct composed total is found, but three weaker workers repeat `30`, so the consensus mechanism promotes the wrong candidate. This is a classic majority-over-quality failure.
- Suggested fix: Raise the weight of forward-chain totals when they satisfy the declared `total_combined_quantity` goal type and dominate by structure or dimensional consistency.

### gsm8k_7: restart-from-beginning download time
- Expected: `160`
- Got: `300`
- Failure mode: `COMPOSITION_FAILURE`
- Selection steps:
  - `GSM8K fission: hit add (4 entries)`
  - `Swarm path result: reasoning_word_problem_fission -> [Grammar] operation_pattern_restart_progress_time`
  - `GSM8K worker preview: gsm8k_worker_0 forward_chain -> 300`
  - `GSM8K worker preview: gsm8k_worker_6 alt_sub -> 138`
  - `GSM8K answer consensus: 300 (struct=0.94, workers=2, weight=2.52, mean=1.83)`
  - `Halting gate: halt (top=1.98, gap=0.06, agree=2, flags=1,1,1,1)`
- Root cause: The right restart template is selected, but its composition is wrong. The system overcounts the partial first attempt instead of treating the initial 40% progress as wasted time plus reboot delay plus full second attempt.
- Suggested fix: Tighten `operation_pattern_restart_progress_time` to model `partial_time + restart_delay + full_retry_time`, not additive blends over raw quantities.

### gsm8k_8: drive away, turn around, return through traffic
- Expected: `45`
- Got: `I don't know`
- Failure mode: `PARSE_FAILURE`
- Selection steps:
  - `GSM8K goal typing: total_combined_quantity via typed_fusion (duration, rate)`
  - `Swarm path result: reasoning_word_problem_fission -> [Number] num_4`
  - `GSM8K worker preview: gsm8k_worker_3 clause_chain -> 69`
  - `GSM8K worker preview: gsm8k_worker_4 goal_adjusted_chain -> 66`
  - `Halting gate: continue (top=1.76, gap=0.03, agree=2, flags=1,0,1,0)`
- Root cause: The parse captures only isolated durations and rates, not the directional structure. The model never forms outbound distance, return distance, then net displacement from home.
- Suggested fix: Add an outbound-return displacement rule that binds travel segments with signed direction before final distance-from-origin evaluation.

### gsm8k_9: overtime pay after 40-hour threshold
- Expected: `460`
- Got: `525`
- Failure mode: `STRATEGY_FAILURE`
- Selection steps:
  - `GSM8K fission: hit add (4 entries)`
  - `GSM8K operation anchors: operation_pattern_multiply_chain_sum, operation_pattern_overtime_total_pay, operation_pattern_base_plus_excess, operation_pattern_restart_progress_time`
  - `Swarm path result: reasoning_word_problem_fission -> [Grammar] operation_pattern_restart_progress_time`
  - `GSM8K worker preview: gsm8k_worker_0 forward_chain -> 525`
  - `Halting gate: halt (top=3.05, gap=0.90, agree=2, flags=1,1,1,1)`
- Root cause: The correct overtime template is present in the neighborhood, but routing still locks onto the restart-progress program because both problems mention thresholds and extra time. Once the wrong template wins, the arithmetic is doomed.
- Suggested fix: Strengthen typed-role routing for `threshold + excess + overtime_rate` so `operation_pattern_overtime_total_pay` outranks generic restart or base-plus-excess templates.

## Aggregate Pattern

Failure mode counts in this run:
- `HALTING_FAILURE`: 2
- `STRATEGY_FAILURE`: 2
- `COMPOSITION_FAILURE`: 2
- `PARSE_FAILURE`: 2

Highest-leverage next fixes:
1. Improve GSM8K typed-role parsing for restart/profit/return-trip/overtime families.
2. Harden halting so a structurally correct numeric candidate can beat noisy majorities.
3. Refine the two existing Grammar templates that are being selected correctly but evaluated incorrectly:
   - `operation_pattern_alternating_discount_pairs`
   - `operation_pattern_restart_progress_time`
