from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from knowledge3d.cranium.phase10.text_to_3d_generator import TextTo3DGenerator
from knowledge3d.cranium.ptx.ptx_ops import PTX_OPS


def _ensure_env() -> None:
    cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING")
    if cuda_launch_blocking != "1":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def generate(prompt: str, honesty_threshold: float = 0.7) -> Dict[str, Any]:
    _ensure_env()
    generator = TextTo3DGenerator()
    path = generator.generate_3d_from_text(prompt, honesty_threshold=honesty_threshold)
    metadata: Dict[str, Any] = {
        "prompt": prompt,
        "path": path,
        "manifest": generator.last_generation or {},
        "ptx": PTX_OPS.last_generated_shape() or {},
    }
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PTX-backed text-to-3D shape for Phase 10 smoke tests")
    parser.add_argument("prompt", help="Text prompt to materialise")
    parser.add_argument(
        "--honesty-threshold",
        type=float,
        default=0.7,
        help="Minimum honesty score required before generation proceeds (default: 0.7)",
    )
    args = parser.parse_args()
    result = generate(args.prompt, honesty_threshold=float(args.honesty_threshold))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
