"""
One-click training pipeline for K3D intent models + scoreboard refresh.

Usage (fast loop, seconds):
  python3 -m knowledge3d.tools.train_all --fast

Full (adds HF multilingual fine-tune, ~1 min on CPU):
  python3 -m knowledge3d.tools.train_all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    # Enforce containment on Debian hosts
    from ..utils.env_guard import enforce_containment  # type: ignore
    enforce_containment("train_all pipeline")
except Exception:
    # Soft-fail: if module not available, continue
    pass

def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Train all (sklearn + optional HF) and refresh scoreboard")
    p.add_argument("--fast", action="store_true", help="Skip HF training; do sklearn + scoreboard only")
    p.add_argument("--langs", default="en,pt,es")
    p.add_argument("--templates-synth", type=int, default=80)
    p.add_argument("--logs-synth", type=int, default=60)
    p.add_argument("--hf-epochs", type=int, default=2)
    p.add_argument("--hf-batch", type=int, default=16)
    p.add_argument("--hf-model", default="distilbert-base-multilingual-cased")
    p.add_argument("--logs", default=str((Path(__file__).resolve().parents[2].parent / (Path(__file__).resolve().parents[2].name + ".local") / "logs")))
    p.add_argument("--gltf", default="viewer/public/ai_books_basic.4k.umap.doors.glb")
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--door", type=int, default=96)
    args = p.parse_args()

    # 1) Train tiny sklearn from templates (multilingual)
    _, out = run([sys.executable, "-m", "knowledge3d.models.intent_classifier", "train-templates",
                  "--langs", args.langs, "--synth-per-label", str(args.templates_synth)])
    print("[sklearn-templates]", out)

    # 2) Train sklearn from logs + synth (fast incremental)
    _, out = run([sys.executable, "-m", "knowledge3d.models.intent_classifier", "train",
                  "--synth-per-label", str(args.logs_synth)])
    print("[sklearn-logs]", out)

    # 3) Optional HF fine-tune (stronger generalization)
    if not args.fast:
        _, out = run([sys.executable, "-m", "knowledge3d.models.intent_hf", "train",
                      "--epochs", str(args.hf_epochs), "--synth-per-label", str(args.logs_synth),
                      "--batch-size", str(args.hf_batch), "--pretrained", args.hf_model,
                      "--langs", args.langs])
        print("[hf]", out)

    # 4) Evaluate logs (confusion snapshot)
    _, out = run([sys.executable, "-m", "knowledge3d.models.eval_logs", "--logs", args.logs])
    print("[eval-logs]", out)
    try:
        summary = json.loads(out)
    except Exception:
        summary = {"pairs": 0}

    # 5) Refresh scoreboard on a representative GLB
    _, out = run([sys.executable, "-m", "knowledge3d.tools.train_session",
                  "--gltf", args.gltf, "--pairs", str(args.pairs), "--door", str(args.door)])
    print("[scoreboard]", out)

    print("\n== train_all summary ==")
    print("pairs:", summary.get("pairs"))
    print("langs:", args.langs, "fast:", args.fast)


if __name__ == "__main__":  # pragma: no cover
    main()
