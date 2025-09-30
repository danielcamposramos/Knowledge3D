from __future__ import annotations

import json
import os
from pathlib import Path

from knowledge3d.tools.phase25 import math_bench_evaluator as mbe  # type: ignore
from knowledge3d.tools import omni_bench_evaluator as obe  # type: ignore


def main() -> None:
    # Force minimal eval path to avoid heavy trainer init
    os.environ.setdefault("K3D_EVAL_MINIMAL", "1")
    os.environ.setdefault("K3D_DISABLE_TEXT_MODALITY", "1")
    os.environ.setdefault("K3D_ENABLE_RPN_POLICY", "1")
    # Small sanity limits first
    math_report = mbe.run(repos=[
        "Maxwell-Jia/AIME_2024",
        "meta-math/MetaMathQA",
        "openai/gsm8k",
    ], limit=5)
    math_out = Path("docs/benchmarks/math_bench_report.json")
    math_out.parent.mkdir(parents=True, exist_ok=True)
    math_out.write_text(json.dumps(math_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Math bench (serial) →", math_out)

    omni_report = obe.run(Path("/home/daniel/.cache/huggingface/datasets"), repos=None, split=None, limit=5)
    omni_out = Path("docs/benchmarks/omni_bench_report.json")
    omni_out.parent.mkdir(parents=True, exist_ok=True)
    omni_out.write_text(json.dumps(omni_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Omni bench (serial) →", omni_out)


if __name__ == "__main__":
    main()

