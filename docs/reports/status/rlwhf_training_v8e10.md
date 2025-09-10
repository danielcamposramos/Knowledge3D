RLWHF Policy Training — v8 (10 epochs)
======================================

Summary
-------
- Dataset: `docs/reports/training/rlwhf_dataset_unified_v8.jsonl` (12019 rows)
- Model: distilgpt2 (GPU‑only enforced)
- Epochs: 10, Batch: 4, Max Len: 384, LR: 5e‑5
- Artifacts: `../Knowledge3D.local/models/rlwhf_policy_v8e10/`
- Logs: `/tmp/rlwhf_train_v8e10.log` (ephemeral), PID: `/tmp/rlwhf_train_v8e10.pid`

Evaluation
----------
- After‑train eval scheduled automatically to 500 samples:
  - Output: `docs/reports/status/rlwhf_policy_eval_v8e10.json`
- A second eval (1000 samples) is scheduled to run after the 500 completes.

Context & Comparison
--------------------
- Prior 2‑epoch v8 eval (120 samples): sim_avg ≈ 0.287, sim_p50 ≈ 0.266.
- Expect better grounding after 10 epochs with the expanded, balanced multimodal v8 (text, image, audio, video, 3D).
- K3D vs standard LLM memory:
  - Standard: large, siloed, high‑dimensional indices; LLM tries to “remember”.
  - K3D: one unified Galaxy (256D), low‑dim/high‑density; small policy learns to read from the Galaxy and cite.

Notes
-----
- GPU‑only policy enforced in code; training aborts if CUDA not available.
- MSR‑VTT downloads are expanding in the background; new video GLBs will be folded into v8 for later runs.

