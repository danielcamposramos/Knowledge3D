from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from ...cranium.phase10.paradigm_switcher import ParadigmSwitcher  # type: ignore


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run multi-paradigm training step")
    ap.add_argument("--mode", required=True, choices=["rlwhf", "qna", "standard_rl", "supervised", "baby"]) 
    ap.add_argument("--data", default="{}", help="JSON string for data payload")
    args = ap.parse_args()
    try:
        data: Dict[str, Any] = json.loads(args.data)
    except Exception:
        data = {}
    ps = ParadigmSwitcher()
    out = ps.train(args.mode, data)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()

