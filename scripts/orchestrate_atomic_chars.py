#!/usr/bin/env python3
"""
Orchestrate sequential training of atomic characters with clean full-fonts.

Runs `scripts/train_atomic_character.py --char <ch>` for each character in a
provided set, sequentially (one at a time) to respect VRAM/CPU/GPU limits.

Defaults:
- Learning rate uses trainer default (0.5) with plateau scheduler.
- Epochs 1500 (extend to 3000 if <80%).
- K3D_GLYPH_STEPS=8 to speed glyph extraction.

Usage:
  PYTHONPATH=. python scripts/orchestrate_atomic_chars.py \
      --chars A..Z,a..z,0..9

"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def parse_char_ranges(spec: str) -> List[str]:
    out: List[str] = []
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    for part in parts:
        if '..' in part and len(part) >= 4:
            a, b = part.split('..', 1)
            if len(a) == 1 and len(b) == 1:
                start, end = ord(a), ord(b)
                if start <= end:
                    out.extend([chr(c) for c in range(start, end + 1)])
                else:
                    out.extend([chr(c) for c in range(end, start + 1)])
            else:
                # fall back to literal if malformed
                out.append(part)
        else:
            # accept literal multi-char chunks and iterate per codepoint
            for ch in part:
                out.append(ch)
    # de-duplicate in order
    seen = set()
    unique: List[str] = []
    for ch in out:
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential atomic char trainer orchestrator")
    parser.add_argument(
        "--chars",
        type=str,
        default="A..Z,a..z,0..9",
        help="Character ranges/list to train (e.g., 'A..Z,a..z,0..9').",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use (defaults to current).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="K3D_GLYPH_STEPS for curve sampling (default: 8).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="Target epochs (extend to max if needed).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=3000,
        help="Max epochs when extension is required.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="How many characters to train concurrently (default: 1).",
    )
    args = parser.parse_args()

    chars = parse_char_ranges(args.chars)
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = Path("/K3D/Knowledge3D.local/logs/atomic_chars") / time.strftime("%Y%m%d_%H%M%S")
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[orchestrator] Training {len(chars)} characters with parallel={args.parallel}")
    print(f"[orchestrator] Logs: {logs_dir}")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root))
    env["K3D_GLYPH_STEPS"] = str(args.steps)

    trainer_script = str(repo_root / "scripts" / "train_atomic_character.py")

    # Simple process pool
    pending = list(chars)
    running: List[tuple[str, subprocess.Popen, Path]] = []
    completed = 0

    def launch_one(ch: str) -> tuple[str, subprocess.Popen, Path]:
        log_path = logs_dir / f"char_{ord(ch)}_{ch}.log"
        print(f"[orchestrator] start '{ch}' → {log_path}")
        lf = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [args.python, trainer_script, "--char", ch, "--epochs", str(args.epochs), "--max-epochs", str(args.max_epochs)],
            cwd=str(repo_root),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
        return (ch, proc, log_path)

    # Fill initial slots
    while pending and len(running) < args.parallel:
        ch = pending.pop(0)
        running.append(launch_one(ch))

    # Main loop
    while running or pending:
        time.sleep(2)
        still_running: List[tuple[str, subprocess.Popen, Path]] = []
        for ch, proc, log_path in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((ch, proc, log_path))
                continue
            completed += 1
            if ret != 0:
                print(f"[orchestrator] WARN: '{ch}' exited code {ret} ({log_path})")
            else:
                print(f"[orchestrator] ✓ Completed '{ch}' ({completed}/{len(chars)})")
        running = still_running
        # Backfill
        while pending and len(running) < args.parallel:
            ch = pending.pop(0)
            running.append(launch_one(ch))

    print("[orchestrator] All requested characters processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
