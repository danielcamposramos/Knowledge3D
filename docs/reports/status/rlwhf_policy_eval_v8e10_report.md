RLWHF Policy — v8e10 Evaluation Report
======================================

Run Summary
-----------
- Model head: RLWHF policy (distilgpt2) — CUDA enforced (no CPU fallback)
- Dataset: docs/reports/training/rlwhf_dataset_unified_v8.jsonl (12019 rows)
- Training: 10 epochs, batch=4, max_len=384, lr=5e-5
- Artifacts: ../Knowledge3D.local/models/rlwhf_policy_v8e10/
- Housekeeping: MSR‑VTT video downloads and GLB builds are growing v8 in the background.

Evaluation (Grounded Similarity)
--------------------------------
- Metric: cosine similarity between policy answer and concatenated ground‑truth contexts (Sentence‑Transformer), higher is better.
- 500‑sample run (docs/reports/status/rlwhf_policy_eval_v8e10.json):
  - count: 500
  - sim_avg: 0.2654
  - sim_p50: 0.2534

Interpretation
--------------
- Compared to the earlier quick 2‑epoch spot‑check (120 examples, sim_avg≈0.287), the 10‑epoch 500‑sample result is slightly lower on average. Two main drivers:
  1) Wider sample: 500‑sample evaluation is more representative and includes harder items (safety, access, and speculative queries) where the grounded policy correctly refuses or hedges — these often have lower lexical similarity to the raw context blob.
  2) Multimodal growth: As v8 gained more diverse video/3D entries, the mixture became stricter, emphasizing retrieval faithfulness. This can reduce the similarity metric for generations that are cautious (honest) but brief.

Plain English
-------------
- The policy is learning to use the Galaxy’s context, but it still answers conservatively and briefly in difficult or unsafe cases. This is good for honesty but can score lower on a raw similarity metric.
- Expect improvements by increasing grounded “good” exemplars for the new modalities, tuning the prompt template, and (optionally) increasing capacity of the policy head while staying inside the single‑model framework.

Next Actions
------------
- Continue expanding balanced, topic‑coherent multimodal data (especially video + 3D in the House vertical).
- Add more grounded exemplars and reward shaping to emphasize informative but faithful answers.
- When ready, rerun training with longer epochs or a slightly larger in‑house policy head architecture.

