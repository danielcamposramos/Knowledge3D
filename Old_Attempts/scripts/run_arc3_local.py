from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc3_local import run_local_arc3
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local ARC3 benchmark without the remote API.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--storage-root", default="/K3D/Knowledge3D.local")
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()

    kv = Knowledgeverse(storage_root=args.storage_root)
    summary = run_local_arc3(
        count=args.count,
        grid_size=args.grid_size,
        max_actions=args.max_actions,
        knowledgeverse=kv,
        log_path=args.log_path or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
