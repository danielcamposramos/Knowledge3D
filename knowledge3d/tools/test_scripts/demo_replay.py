# DEPRECATED: legacy pre-PTX script; kept for reference. Outputs belong in Knowledge3D.local/old_attempts.
from __future__ import annotations

"""
Deterministic replay harness for the fused-head demo.

Running this module seeds all random sources (NumPy, Torch, PTX helpers) so
perception → reasoning → action → tablet logging runs produce identical
artefacts.  This is primarily used during Week 7 dry-runs to validate
regressions quickly without launching the full viewer stack.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

DEMO_SEED = 0x4B3D2025
MMAP_PATH = Path("tablet_log.mmap")
REPLAY_LOG = Path("replay_actions.jsonl")
CHECKSUM_PATHS = [
    Path("demo_tour.glb"),
    REPLAY_LOG,
    MMAP_PATH,
]


def _seed_everything(seed: int = DEMO_SEED) -> None:
    np.random.seed(seed)
    if torch is not None and torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_action_log(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    return {"actions": lines}


def run_demo() -> str:
    """
    Perform a deterministic dry-run and return a checksum of relevant artefacts.

    The current implementation focuses on repeatability for automated tests.
    Integrating the full fused-head pipeline remains a TODO once the viewer
    scaffolding is packaged in the repository.
    """

    _seed_everything()

    # For now we simply ensure the replay log exists and emit a dummy record if missing.
    REPLAY_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not REPLAY_LOG.exists():
        with REPLAY_LOG.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"timestamp": 0, "action_type": "NO_ACTION"}) + "\n")

    digest = hashlib.sha256()
    digest.update(json.dumps(_load_action_log(REPLAY_LOG)).encode("utf-8"))
    for path in CHECKSUM_PATHS:
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    checksum = run_demo()
    print(f"Deterministic replay checksum: {checksum}")

