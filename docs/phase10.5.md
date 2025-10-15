# KNOWLEDGE3D — PHASE 10.5: MULTI-PARADIGM COGNITIVE BOOTSTRAPPING

## GOAL
Train the AI to master multiple learning paradigms — RLWHF, Q&A, standard RL, supervised, and a baby‑style curriculum — routed through a single Cranium head with multi‑modality fusion.

## COMPONENTS
1. Paradigm Switcher — `knowledge3d/cranium/phase10/paradigm_switcher.py`
2. Baby Curriculum — `knowledge3d/cranium/phase10/baby_curriculum.py`
3. Multi‑Modality Fusion — `knowledge3d/cranium/phase10/multi_modality_fusion.py`
4. CLI wrappers — `knowledge3d/tools/phase10/{paradigm_switcher.py,baby_curriculum.py}`

## REPRODUCTION

Use your Conda env (e.g., `k3d-testing`) and set `PYTHONPATH=.`.

1) RLWHF (honesty scoring, explanation required):

```bash
PYTHONPATH=. conda run -n k3d-testing python -m knowledge3d.tools.phase10.paradigm_switcher \
  --mode rlwhf --data '{"query": "Is the sky green?"}' \
  2>&1 | tee logs/phase10.5_rlwhf.log
```

2) Q&A (no feedback, pure correctness):

```bash
PYTHONPATH=. conda run -n k3d-testing python -m knowledge3d.tools.phase10.paradigm_switcher \
  --mode qna --data '{"input": [0.1,0.9,0.2,0.8], "label": 1}' \
  2>&1 | tee logs/phase10.5_qna.log
```

3) Baby curriculum (progressive modalities):

```bash
PYTHONPATH=. conda run -n k3d-testing python -m knowledge3d.tools.phase10.baby_curriculum --stage 1 --modality text \
  2>&1 | tee -a logs/phase10.5_baby.log
PYTHONPATH=. conda run -n k3d-testing python -m knowledge3d.tools.phase10.baby_curriculum --stage 2 --modality text_image \
  2>&1 | tee -a logs/phase10.5_baby.log
PYTHONPATH=. conda run -n k3d-testing python -m knowledge3d.tools.phase10.baby_curriculum --stage 3 --modality text_image_audio \
  2>&1 | tee -a logs/phase10.5_baby.log
PYTHONPATH=. conda run -n k3d-testing python -m knowledge3d.tools.phase10.baby_curriculum --stage 4 --modality all_modalities \
  2>&1 | tee -a logs/phase10.5_baby.log
```

## LOGS
- `logs/phase10.5_rlwhf.log` — RLWHF training steps and evaluations
- `logs/phase10.5_qna.log` — Q&A training
- `logs/phase10.5_baby.log` — Baby curriculum stages

## NOTES
- RLWHF uses `TeacherEvaluator` (Ollama `exaone3.5:latest` by default) to score responses with {-1, +0.5, +1} and maps to reward {0, 0.5, 1}.
- The student head includes a scalar `honesty_bias` influenced by RLWHF rewards for a simple end‑to‑end loop.
- Multi‑modality fusion projects each modality to a common dimension and applies a lightweight attention over modality tokens.
- Torch is optional; when unavailable, stubs run without gradient steps.
