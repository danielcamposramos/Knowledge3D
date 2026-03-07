# Tablet Boundary Benchmark Audit (Real External Data)

Date: 2026-03-06
Mode: Headless Tablet boundary
Augmentation paused: yes

## Dataset roots
- ARC: `/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation`
- Math root: `/K3D/K3D_llama_cpp/datasets`
- Math sources:
  - `/K3D/K3D_llama_cpp/datasets/GSM8K/grade_school_math/data/test.jsonl`
  - `/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl`
- LHE local corpus: `/K3D/K3D_llama_cpp/datasets/last_humanity_exam/last_humanity_exam.json`
- LHE source dataset: `cais/hle` test split

## Run caps
- ARC tasks: 50
- Math problems: 100
- LHE questions: 50

## Results
- ARC: 0 / 50 = 0.0
- Math: 0 / 100 = 0.0
- LHE: 0 / 50 = 0.0

## Math composition
- Competition buckets: GSM8K, MATH:Algebra
- Diagnostics predicted_none_rate: 1.0

## LHE normalization note
- Local corpus contains 2500 normalized questions.
- Image-present questions: 342
- Text-only questions: 2158
- Multiple-choice questions are normalized so `correct_answer` stores option text, not raw letter labels.
- Image presence is preserved as metadata (`has_image`), but this audited headless run uses text-first local records rather than binary image payloads.

## Artifact paths
- Summary: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D.local/results/tablet_boundary_audited_20260306_231020/tablet_boundary_benchmark_summary.json`
- Metadata: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D.local/results/tablet_boundary_audited_20260306_231020/run_metadata.json`
- HLE manifest: `/K3D/K3D_llama_cpp/datasets/last_humanity_exam/manifest.json`
