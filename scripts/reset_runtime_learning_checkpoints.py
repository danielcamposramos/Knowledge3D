#!/usr/bin/env python3
"""Back up and remove active runtime-learning checkpoints.

Preserves consolidated resident knowledge while forcing a fresh boot for:
- TRM weights
- adaptive swarm weights/adapters
- navigator specialist routing bias/topology

Timestamped snapshots remain untouched except for the current ``latest`` pointers
explicitly listed below.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _move_path(path: Path, backup_root: Path, manifest: list[dict[str, Any]], dry_run: bool) -> None:
    entry: dict[str, Any] = {
        "source": str(path),
        "backup": str(backup_root / path.name),
        "exists": path.exists(),
        "type": "directory" if path.is_dir() else "file",
        "moved": False,
    }
    if not path.exists():
        manifest.append(entry)
        return
    if dry_run:
        manifest.append(entry)
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    target = backup_root / path.name
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.move(str(path), str(target))
    entry["moved"] = True
    manifest.append(entry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path("/K3D/Knowledge3D.local"),
        help="Runtime workspace root (default: /K3D/Knowledge3D.local)",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="Optional explicit backup directory. Defaults under results/runtime_learning_reset_<timestamp>/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files.",
    )
    args = parser.parse_args()

    storage_root = args.storage_root.resolve()
    checkpoints = storage_root / "checkpoints"
    backup_root = (
        args.backup_root.resolve()
        if args.backup_root is not None
        else (storage_root / "results" / f"runtime_learning_reset_{_timestamp()}").resolve()
    )

    active_paths = [
        checkpoints / "trm_weights.npz",
        checkpoints / "trm_weights_latest.npz",
        checkpoints / "adaptive_swarm",
        checkpoints / "trm_routing_state.json",
        checkpoints / "specialist_routes_latest.json",
    ]

    manifest: list[dict[str, Any]] = []
    for path in active_paths:
        _move_path(path, backup_root, manifest, args.dry_run)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "storage_root": str(storage_root),
        "checkpoints_root": str(checkpoints),
        "backup_root": str(backup_root),
        "dry_run": bool(args.dry_run),
        "paths": manifest,
    }

    if not args.dry_run:
        backup_root.mkdir(parents=True, exist_ok=True)
        (backup_root / "reset_manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
