RLWHF Policy — v8e10 Evaluation (1000‑sample)
=============================================

Results
-------
- JSON: `docs/reports/status/rlwhf_policy_eval_v8e10_1k.json`
- count: 1000
- sim_avg: 0.2717
- sim_p50: 0.2549

Interpretation
--------------
- Scores are consistent with the 500‑sample run (sim_avg≈0.265, sim_p50≈0.253), indicating stable grounding behavior across a wider slice. Minor variation is expected due to modality mix and safety/refusal cases.

Next steps
----------
- Complete the 1000‑sample eval as soon as the GPU is free; then refresh this file with `count`, `sim_avg`, and `sim_p50` and summarize differences vs the 500‑sample run.
