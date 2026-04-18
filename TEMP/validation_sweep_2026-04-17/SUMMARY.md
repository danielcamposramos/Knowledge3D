`git rev-parse HEAD`: `9db2eb2db2739b6ff236d01519997078f9b20233`
`wc -l knowledge3d/knowledgeverse/knowledgeverse.py`: `15963`

Accuracies:
- `mmlu`: `10/50` (`0.200`)
- `gsm8k`: `1/50` (`0.020`)
- `math_competitions`: `1/50` (`0.020`)
- `lhe`: `1/39` (`0.026`)
- `arc_agi_1`: `0/50` (`0.000`)

Sanity Criteria:
- ✅ `ring_used`: All live benchmark items returned query_tick/trm_io/action_buffers.
- ✅ `meaning_class_spread`: 4 distinct argmax classes across 239 routed items.
- ✅ `wire_leakage`: 5 sampled payloads contained no forbidden keys.
- ✅ `tickdriver_bounds`: ticks_total=764, produced_total=241, error_ticks=0, wall_seconds=4502.8.
- ✅ `janet_integrity`: T0=PASS, T_end=PASS.
- ✅ `token_in_set_count`: pre=1, post=1.

Stall Ledger:
- `mmlu`: stalled_at_item=`20`, wall_timeouts=`2`, wall_ceiling_hit=`False`, produced_outputs=`50`
- `gsm8k`: stalled_at_item=`None`, wall_timeouts=`0`, wall_ceiling_hit=`False`, produced_outputs=`50`
- `math_competitions`: stalled_at_item=`29`, wall_timeouts=`1`, wall_ceiling_hit=`False`, produced_outputs=`50`
- `lhe`: stalled_at_item=`39`, wall_timeouts=`10`, wall_ceiling_hit=`True`, produced_outputs=`39`
- `arc_agi_1`: stalled_at_item=`0`, wall_timeouts=`1`, wall_ceiling_hit=`False`, produced_outputs=`50`

Janet T0: PASS
Janet T_end: PASS
`wc -l TEMP/validation_sweep_2026-04-17/ring_trace.jsonl`: `482`

Daemon booted with default galaxies loaded=True and gpu_binding_total=0. The sweep preserved the live ring path where datasets were locally runnable, and the ring trace recorded `482` events. HLE remains blocked in this checkout if its payload stayed unavailable, because the local `hle-src` repo contains only the evaluation scaffold and the gated `cais/hle` split cannot be fetched anonymously.

### Round B delta

- meaning counts before: `{'COMPARATIVE_CHOICE': 0, 'DEFINITION_LOOKUP': 0, 'FACTUAL_RECALL': 32, 'GENERATIVE_COMPOSITION': 0, 'GROUNDED_DIALOG': 1, 'MULTI_HOP_INFERENCE': 0, 'NUMERIC_COMPUTE': 143, 'SPATIAL_TRANSFORM': 49, 'UNKNOWN': 14}`
- meaning counts after: `{'COMPARATIVE_CHOICE': 0, 'DEFINITION_LOOKUP': 0, 'FACTUAL_RECALL': 46, 'GENERATIVE_COMPOSITION': 0, 'GROUNDED_DIALOG': 1, 'MULTI_HOP_INFERENCE': 0, 'NUMERIC_COMPUTE': 143, 'SPATIAL_TRANSFORM': 49}`
- wall_timeouts before: `14`
- wall_timeouts after: `14`
- ARC executor verdict: `executors=['game2d_grid_materializer'], validators=['none'], trace_roles=[('router', 'executor', 'unknown', 'unknown', 'unknown', 'unknown', 'unknown', 'unknown')]`

### Round B.3 — multi-hop separability
- `multi_hop` `Janet buys 3 notebooks at $4 each and then uses a $2 coupon. How much does she pay?` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- `multi_hop` `A train travels 2 hours at 50 mph and then 1 hour at 30 mph. How far did it go altogether?` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- `multi_hop` `If all blue boxes contain 4 marbles and there are 3 blue boxes, how many marbles are there?` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- `multi_hop` `Mia reads 12 pages on Monday and twice that on Tuesday. How many pages did she read in total?` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- `multi_hop` `A baker makes 18 rolls, sells 7, and packs the rest equally into 11 bags. How many go in each bag?` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- `direct_compute` `2 + 3 = ?` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- `direct_compute` `sqrt(16)` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- `direct_compute` `14 * 6` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- `direct_compute` `100 / 4` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- `direct_compute` `9 squared` -> MULTI_HOP_INFERENCE=`0.125000`, NUMERIC_COMPUTE=`0.125000`, winner=`FACTUAL_RECALL`
- Navigator is flat: multi-hop and numeric softmax are tied across all 10 prompts.
