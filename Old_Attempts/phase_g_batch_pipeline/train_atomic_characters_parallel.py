#!/usr/bin/env python3
"""
Train multiple atomic characters in parallel on single GPU.

With 12GB VRAM and FC-only training, we can train 6-8 characters simultaneously
instead of sequentially, achieving 6-8x speedup.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# All 62 characters to train
BASE_CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

# Characters already completed (update this list as training progresses)
COMPLETED = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']  # Skip characters already retrained


def train_character_batch(chars: List[str], batch_id: int, epochs: int = 1500, fonts: int = 0, fc_only: bool = True):
    """Launch parallel training for a batch of characters."""

    processes = []
    log_files = []

    for char in chars:
        # Create log file for this character
        log_path = f"/tmp/train_char_{ord(char)}_{char}_batch{batch_id}.log"
        log_files.append(log_path)

        # Build command
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_atomic_character.py"),
            "--char", char,
            "--epochs", str(epochs),
            "--fonts", str(fonts),
        ]

        if fc_only:
            cmd.append("--fc-only")

        # Start process with log redirection
        print(f"[Batch {batch_id}] Starting character '{char}' (log: {log_path})")

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env={
                    **subprocess.os.environ,
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTHONPATH": str(PROJECT_ROOT),
                }
            )

        processes.append((char, process, log_path))

    # Monitor processes
    print(f"[Batch {batch_id}] Waiting for {len(chars)} characters to complete...")

    completed_chars = []
    while processes:
        time.sleep(30)  # Check every 30 seconds

        remaining = []
        for char, process, log_path in processes:
            retcode = process.poll()

            if retcode is not None:
                # Process finished
                if retcode == 0:
                    print(f"[Batch {batch_id}] ✓ Character '{char}' completed successfully")
                    completed_chars.append(char)
                else:
                    print(f"[Batch {batch_id}] ✗ Character '{char}' failed with code {retcode}")
                    print(f"    Check log: {log_path}")
            else:
                # Still running
                remaining.append((char, process, log_path))

        processes = remaining

        if remaining:
            print(f"[Batch {batch_id}] Still training: {[c for c, _, _ in remaining]}")

    print(f"[Batch {batch_id}] All characters completed: {completed_chars}")
    return completed_chars


def main():
    """Train all characters in parallel batches."""

    # Configuration
    EPOCHS = 1500
    FONTS = 0  # 0 = use all fonts from font_db.pkl
    FC_ONLY = True
    PARALLEL_JOBS = 6  # Train 6 characters at once (safe for 12GB VRAM)

    # Filter out completed characters
    remaining_chars = [c for c in BASE_CHARACTERS if c not in COMPLETED]

    if not remaining_chars:
        print("All characters already trained!")
        return 0

    print("=" * 80)
    print(f"PARALLEL ATOMIC CHARACTER TRAINING")
    print("=" * 80)
    print(f"Total characters: {len(BASE_CHARACTERS)}")
    print(f"Already completed: {len(COMPLETED)}")
    print(f"Remaining: {len(remaining_chars)}")
    print(f"Parallel jobs: {PARALLEL_JOBS}")
    print(f"Epochs per character: {EPOCHS}")
    print(f"Fonts: {'ALL' if FONTS == 0 else FONTS}")
    print(f"Mode: {'FC-only' if FC_ONLY else 'Full CNN'}")
    print("=" * 80)
    print()

    # Split into batches
    n_batches = (len(remaining_chars) + PARALLEL_JOBS - 1) // PARALLEL_JOBS

    all_completed = []

    for batch_id in range(n_batches):
        start_idx = batch_id * PARALLEL_JOBS
        end_idx = min(start_idx + PARALLEL_JOBS, len(remaining_chars))
        batch_chars = remaining_chars[start_idx:end_idx]

        print(f"\n{'=' * 80}")
        print(f"BATCH {batch_id + 1}/{n_batches}: Training {len(batch_chars)} characters")
        print(f"Characters: {batch_chars}")
        print(f"{'=' * 80}\n")

        completed = train_character_batch(
            batch_chars,
            batch_id=batch_id + 1,
            epochs=EPOCHS,
            fonts=FONTS,
            fc_only=FC_ONLY
        )

        all_completed.extend(completed)

    print("\n" + "=" * 80)
    print("ALL BATCHES COMPLETE")
    print("=" * 80)
    print(f"Successfully trained: {len(all_completed)}/{len(remaining_chars)} characters")
    print(f"Completed: {all_completed}")
    print()

    if len(all_completed) < len(remaining_chars):
        failed = set(remaining_chars) - set(all_completed)
        print(f"Failed characters: {sorted(failed)}")
        print("Check individual log files in /tmp/train_char_*.log")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
