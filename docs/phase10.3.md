# KNOWLEDGE3D — PHASE 10.3: IN-LOCO PROFESSOR — RLWHF TEACHER MODELS

## GOAL
Transform `exaone3.5:latest` and `exaone-deep:latest` into in‑loco professors that evaluate AI responses for correctness and honesty using Reinforced Learning With Honesty and Feedback (RLWHF).

## COMPONENTS
- Teacher System Prompt — `knowledge3d/tools/phase10/teacher_prompt.py`
- Teacher Evaluator — `knowledge3d/cranium/phase10/teacher_evaluator.py`
- RLWHF Training Pipeline — `knowledge3d/tools/phase10/thinking_tag_trainer.py` (RLWHF mode)
- Book Processor improvements — first-call timeout + optional system prompt

## REPRODUCTION

Adjust the Conda environment name to your setup (e.g., `k3d-testing`, `k3d-rapids`, or `k3dml`). Ensure `PYTHONPATH=.` when invoking modules.

1) Distill thinking tags with exaone‑deep (Ollama):

```bash
PYTHONPATH=. conda run -n k3d-testing python -m knowledge3d.tools.phase10.book_processor \
  --books "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to think/JSON" \
  --model exaone-deep:latest \
  --limit 20 \
  --output viewer/public/books/thinking_tags_deep.json \
  2>&1 | tee logs/phase10.3_distillation.log
```

2) Train with RLWHF teacher feedback (50 epochs):

```bash
PYTHONPATH=. conda run -n k3d-testing python -m knowledge3d.tools.phase10.thinking_tag_trainer \
  --books "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to think/JSON" \
  --model exaone-deep:latest \
  --epochs 50 \
  --limit 20 \
  --mode rlwhf \
  --output_model viewer/public/models/thinking_tag_embedder_rlwhf.pth \
  --output_tags viewer/public/models/tag_names_rlwhf.json \
  2>&1 | tee logs/phase10.3_training.log
```

3) Test `/think` live command (viewer+bridge running):

```text
/think "break problem into subproblems"
```

Note: `/think` currently looks for `viewer/public/models/thinking_tag_embedder_deep.pth` and `tag_names_deep.json`. If you trained RLWHF artifacts, either copy them to these names or update the live server logic to reference the RLWHF files.

## LOGS
- `logs/phase10.3_distillation.log` — distillation output
- `logs/phase10.3_training.log` — RLWHF training output
- `logs/phase10.3_evaluation.log` — teacher evaluations (optional: record by tee-ing evaluator prints)

## VALIDATION
- Expect stable training without shape mismatches (dynamic tag head sizing).
- Teacher scoring returns {-1, +0.5, +1} with explanations.
- No crashes on empty/short inputs; empty tag sets abort cleanly with a message.

## FILES

```
knowledge3d/
├── cranium/
│   └── phase10/
│       ├── teacher_evaluator.py
│       └── thinking_tag_embedder.py (existing)
├── tools/
│   └── phase10/
│       ├── teacher_prompt.py
│       ├── book_processor.py (updated)
│       └── thinking_tag_trainer.py (updated: RLWHF)
└── docs/
    └── phase10.3.md (this file)
```
